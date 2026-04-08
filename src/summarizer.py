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
    """논문에서 불필요한 섹션을 제거하고 핵심만 병합합니다. (Bulletproof Tail-cut)"""
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # 1. 후반부 일괄 절삭 (가장 확실한 키워드 검색 방식)
        # 요약에 불필요한 정보가 시작되는 지점들의 키워드 리스트
        target_words = [
            'references', 'bibliography', '참고문헌',
            'acknowledgments', 'acknowledgement',
            'abbreviations', 'notes', 'declaration of',
            'supporting information', 'associated content', 'appendices', 'appendix',
            'author information', 'conflict of interest', 'data availability',
            'literature cited', 'graphic abstract'
        ]
        
        earliest_index = len(md_text)
        
        # 각 키워드에 대해 텍스트 전체에서 위치를 찾음
        for word in target_words:
            # ## ■ [NOTES], ## **REFERENCES** 등을 모두 잡기 위해 중간에 어떤 문자든 허용
            # 단어 사이의 공백이나 기호도 유연하게 대응 (예: associated content -> associated.*content)
            word_pattern = word.replace(' ', r'.*')
            pattern = rf'\n#+.*{word_pattern}'
            match = re.search(pattern, md_text, flags=re.IGNORECASE)
            
            if match:
                if match.start() < earliest_index:
                    earliest_index = match.start()
        
        # 만약 헤더로 잡히지 않았더라도, 줄 시작 부분에 [1] 등이 나오면 레퍼런스로 간주
        ref_list_pattern = re.search(r'\n\s*[-\s]*(\[\d+\]|\(\d+\)|\d+\.)\s+[A-Z]', md_text)
        if ref_list_pattern and ref_list_pattern.start() > len(md_text) * 0.3:
            if ref_list_pattern.start() < earliest_index:
                earliest_index = ref_list_pattern.start()
        
        # 최종 절삭
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

        # 3. 최종 안전 샘플링 제거 (사용자 요청: 텍스트 전체 전달)
        key_content = md_text

        return key_content
    except Exception as e:
        log_message(f"텍스트 추출 및 필터링 오류 ({os.path.basename(pdf_path)}): {e}", "ERROR")
        return ""

def summarize_paper_optimized(client, filename, key_content):
    """단일 프롬프트로 서지 정보 추출과 요약을 동시에 수행합니다. (인용구 혼동 방지 로직)"""
    
    # 프롬프트에 본문 인용과 실제 저자를 구분하는 로직 강화
    prompt = f"""당신은 세계 최고의 학술 분석가입니다. 제공된 논문을 분석하여 정확한 정보를 추출하세요.

[작업 지시]
1. **서지 정보 정밀 추출 (중요: 본문 인용구와 혼동 금지!)**:
   - **연도(year)**: 논문의 공식 발행 연도 4자리.
   - **제1저자 성명(full_name)**: 반드시 **논문 제목(Title) 바로 아래에 적힌 저자 목록**에서 첫 번째 이름을 찾으세요.
     * 경고: 본문 중간에 언급된 `Welch et al.` 또는 `Ober et al.` 등은 인용된 타인의 논문 저자일 뿐입니다. 절대 이들을 이 논문의 저자로 적지 마세요.
     * 실제 이 논문의 저자(Primary Author)는 논문 맨 앞부분에 있습니다. (예: Wei-Liang Chen)
   - **저자 성(author)**: 위 성명에서 **'성(Surname)'**만 따로 분리하세요. (예: Chen)
     * 임의로 이름을 바꾸거나 철자를 수정하지 마세요.
   - **키워드(keyword)**: 논문의 주제를 관통하는 핵심 영문 키워드 1개.

2. **논문 요약**: 한국어로 핵심 내용을 요약하세요. (# 요약, ## 핵심 내용, ## 결론 형식 준수)

--- 논문 내용 ({filename}) ---
{key_content}
--- 내용 끝 ---

**응답 형식**: 반드시 아래의 JSON 구조로만 응답하세요.
{{
  "metadata": {{
    "year": "YYYY",
    "full_name": "First Middle Last",
    "author": "Surname",
    "keyword": "Keyword"
  }},
  "summary": "# 요약\\n...\\n## 핵심 내용\\n...\\n## 결론\\n..."
}}
"""

    try:
        # 안정적인 기본 설정으로 호출하되, 프롬프트 지시어로 사고 유도
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
    # 가장 안정적인 기본 설정으로 복구
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
        
        # 파일이 실제로 존재하는지 한 번 더 확인 (이동 중 에러 방지)
        if not os.path.exists(file_path):
            continue
            
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
