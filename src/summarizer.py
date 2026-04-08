import os
import shutil
import time
import json
import re
import pymupdf4llm
from google import genai
from src.utils import log_message, calculate_file_hash, load_history, save_history, parse_json_response

# 설정
CLASSIFIED_DIR = "output/classified"
SUMMARIES_DIR = "output/summaries"
PROCESSED_SUBDIR = "processed"

# 요약용 모델 (gemma_version 브랜치 전용)
MODEL_NAME = "gemma-4-31b-it"


def extract_key_sections(pdf_path: str) -> str:
    """논문에서 불필요한 섹션을 제거하고 핵심만 병합합니다. (Final Tail-cut Optimization)"""
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # 1. 후반부 일괄 절삭 (꼬리 자르기 전략)
        # 요약에 불필요한 정보가 시작되는 지점들의 키워드 리스트
        tail_keywords = [
            r'\n#+\s*References', r'\n#+\s*BIBLIOGRAPHY', r'\n#+\s*참고문헌',
            r'\n#+\s*Acknowledgments', r'\n#+\s*Acknowledgement',
            r'\n#+\s*Abbreviations', r'\n#+\s*Notes', r'\n#+\s*Declaration of Competing Interest',
            r'\n#+\s*Supporting Information', r'\n#+\s*Appendices', r'\n#+\s*Appendix',
            r'\n#+\s*Author Information', r'\n#+\s*Conflict of Interest', r'\n#+\s*Data Availability',
            r'\n#+\s*Literature Cited', r'\n#+\s*NOTES AND REFERENCES'
        ]
        
        earliest_index = len(md_text)
        
        # 블랙리스트 키워드 중 가장 먼저 등장하는 위치 찾기
        for kw_pattern in tail_keywords:
            match = re.search(kw_pattern, md_text, flags=re.IGNORECASE)
            if match:
                if match.start() < earliest_index:
                    earliest_index = match.start()
        
        # 레퍼런스 번호 리스트 감지 (예: [1] 또는 (1) 등)
        ref_list_pattern = re.search(r'\n\s*(\[\d+\]|\(\d+\)|\d+\.)\s+[A-Z]', md_text)
        if ref_list_pattern and ref_list_pattern.start() > len(md_text) * 0.2:
            if ref_list_pattern.start() < earliest_index:
                earliest_index = ref_list_pattern.start()
        
        # 발견된 가장 빠른 위치에서 텍스트를 통째로 두 동강 내고 앞부분만 취함!
        if earliest_index < len(md_text):
            md_text = md_text[:earliest_index]

        # 2. 섹션별 정밀 필터링 (중간 섹션 제거)
        # 헤더(#)를 기준으로 조각내어 블랙리스트 섹션 제거
        sections = re.split(r'(?=\n#+ )', md_text)
        blacklist = [
            'method', 'material', 'experimental', 'procedure', 'implementation',
            'author', 'conflict', 'supporting', 'appendix'
        ]
        
        cleaned_sections = []
        for section in sections:
            header_line = section.strip().split('\n')[0].lower()
            # 헤더에 블랙리스트 키워드가 있으면 스킵
            if any(bad_word in header_line for bad_word in blacklist):
                continue
            cleaned_sections.append(section)
        
        md_text = "".join(cleaned_sections)

        # 3. 최종 안전 샘플링 (토큰 한도 및 AI 집중도 유지) 상향 조정 (30,000자)
        if len(md_text) > 30000:
            key_content = md_text[:20000] + "\n\n... [중략: 핵심 결과 및 논의 보존] ...\n\n" + md_text[-10000:]
        else:
            key_content = md_text

        return key_content
    except Exception as e:
        log_message(f"텍스트 추출 및 필터링 오류 ({os.path.basename(pdf_path)}): {e}", "ERROR")
        return ""

def summarize_paper_optimized(client, filename, key_content):
    """단일 프롬프트로 서지 정보 추출과 요약을 동시에 수행합니다."""
    
    prompt = f"""당신은 전문 학술 분석가입니다. 제공된 논문 내용을 바탕으로 다음 두 가지 작업을 수행하세요.

1. **서지 정보 추출**: 파일명 생성을 위해 [발행 연도, 제1저자 성, 핵심 키워드 1개]를 추출하세요.
2. **논문 요약**: 한국어로 핵심 내용을 요약하세요. (# 요약, ## 핵심 내용, ## 결론 형식 준수)

--- 논문 내용 ({filename}) ---
{key_content}
--- 내용 끝 ---

**응답 형식**: 반드시 아래의 JSON 구조로만 응답하세요.
{{
  "metadata": {{
    "year": "YYYY",
    "author": "Surname",
    "keyword": "Keyword"
  }},
  "summary": "# 요약\\n...\\n## 핵심 내용\\n...\\n## 결론\\n..."
}}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        
        result = parse_json_response(response.text)
        if result and "metadata" in result and "summary" in result:
            return result
        else:
            log_message(f"요약 결과 파싱 실패 ({filename})", "ERROR")
            return None
    except Exception as e:
        log_message(f"요약 중 API 오류 ({filename}): {e}", "ERROR")
        return None

def sanitize_filename(name):
    """파일명으로 사용할 수 없는 문자를 제거합니다."""
    return re.sub(r'[\\/*?:"<>|]', "", name).replace(" ", "_")

def process_summaries():
    """논문으로 분류된 파일들을 최적화된 방식으로 요약하고 이름을 변경합니다."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = load_history()
    
    source_dir = os.path.join(CLASSIFIED_DIR, "논문")
    if not os.path.exists(source_dir):
        return

    processed_dir = os.path.join(source_dir, PROCESSED_SUBDIR)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(SUMMARIES_DIR, exist_ok=True)

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]
    
    if not files:
        log_message("요약할 새로운 논문이 없습니다.")
        return

    log_message(f"총 {len(files)}개의 논문을 요약 처리합니다. (단일 페이로드 최적화 모드)")

    for filename in files:
        file_path = os.path.join(source_dir, filename)
        file_hash = calculate_file_hash(file_path)

        if file_hash in history and history[file_hash].get("summarized"):
            continue

        log_message(f"요약 시작: {filename}...")
        
        # 1. 핵심 섹션 추출
        key_content = extract_key_sections(file_path)
        if not key_content:
            continue

        # [디버그용] 필터링된 텍스트 저장 및 길이 확인
        debug_path = f"output/debug_filtered_{os.path.splitext(filename)[0]}.md"
        with open(debug_path, "w", encoding="utf-8-sig") as f:
            f.write(key_content)
        log_message(f"🔍 필터링 후 텍스트 길이: {len(key_content)}자 (저장: {debug_path})")

        # 2. 통합 요약 및 메타데이터 추출 (API 1회 호출)
        result = summarize_paper_optimized(client, filename, key_content)
        if not result:
            continue

        # 3. 결과 처리
        meta = result["metadata"]
        year = meta.get("year", "0000")
        author = meta.get("author", "Unknown")
        keyword = meta.get("keyword", "Paper")
        
        new_filename = f"{year}_{sanitize_filename(author)}_{sanitize_filename(keyword)}.pdf"
        summary_filename = f"{os.path.splitext(new_filename)[0]}_summary.md"

        # 요약본 저장 (utf-8-sig)
        summary_path = os.path.join(SUMMARIES_DIR, summary_filename)
        with open(summary_path, "w", encoding="utf-8-sig") as f:
            f.write(result["summary"])

        # 원본 파일 이동 및 이름 변경
        target_path = os.path.join(processed_dir, new_filename)
        
        # 중복 이름 처리
        if os.path.exists(target_path):
            base, ext = os.path.splitext(new_filename)
            target_path = os.path.join(processed_dir, f"{base}_{int(time.time())}{ext}")

        shutil.move(file_path, target_path)

        # 히스토리 업데이트 (키가 없는 경우 대비 안전 장치 추가)
        if file_hash not in history:
            history[file_hash] = {"filename": filename, "classified": True}
            
        history[file_hash].update({
            "summarized": True,
            "new_filename": os.path.basename(target_path),
            "summary_file": summary_filename,
            "metadata": meta
        })
        save_history(history)
        
        log_message(f"✅ 요약 완료: {filename} -> {os.path.basename(target_path)}")
        time.sleep(10) # 속도 제한 준수

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    process_summaries()
