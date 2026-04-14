import os
import shutil
import time
import json
import re
import pymupdf4llm
from google import genai
from src.utils import (
    log_message, 
    calculate_file_hash, 
    load_history, 
    save_history, 
    parse_json_response,
    ensure_api_key
)

# 설정
CLASSIFIED_DIR = "output/classified"
SUMMARIES_DIR = "output/summaries"
PROCESSED_SUBDIR = "processed"

# 요약용 모델 (main 브랜치: Gemini 3.1 Flash Lite 적용)
MODEL_NAME = "models/gemini-3.1-flash-lite-preview"

# 전역 클라이언트 변수 (지연 초기화)
client = None

def get_client():
    """API 키를 확인하고 클라이언트를 반환합니다."""
    global client
    if client is None:
        api_key = ensure_api_key()
        client = genai.Client(api_key=api_key)
    return client

def extract_key_sections(pdf_path: str) -> str:
    """논문에서 불필요한 섹션을 제거하고 핵심만 병합합니다. (Bulletproof Tail-cut)"""
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)

        # 1. 후반부 일괄 절삭 (가장 확실한 키워드 검색 방식)
        target_words = [
            'references', 'bibliography', '참고문헌',
            'acknowledgments', 'acknowledgement',
            'abbreviations', 'notes', 'declaration of',
            'associated content', 'appendices', 'appendix',
            'author information', 'conflict of interest', 'data availability',
            'literature cited', 'graphic abstract'
        ]

        earliest_index = len(md_text)

        for word in target_words:
            word_pattern = word.replace(' ', r'.*')
            pattern = rf'\n#+.*{word_pattern}'
            match = re.search(pattern, md_text, flags=re.IGNORECASE)

            if match:
                if match.start() < earliest_index:
                    earliest_index = match.start()

        # 줄 시작 부분에 [1] 등이 나오면 레퍼런스로 간주 (0.3 지점 이후부터)
        ref_list_pattern = re.search(r'\n\s*[-\s]*(\[\d+\]|\(\d+\)|\d+\.)\s+[A-Z]', md_text)
        if ref_list_pattern and ref_list_pattern.start() > len(md_text) * 0.3:
            if ref_list_pattern.start() < earliest_index:
                earliest_index = ref_list_pattern.start()

        if earliest_index < len(md_text):
            md_text = md_text[:earliest_index]

        # 2. 섹션별 정밀 필터링 (중간 섹션 제거)
        sections = re.split(r'(?=\n#+ )', md_text)
        blacklist = [
            'method', 'material', 'experimental', 'procedure', 'implementation',
            'author', 'conflict', 'supporting', 'appendix'
        ]

        cleaned_sections = []
        for section in sections:
            header_line = section.strip().split('\n')[0].lower()
            if any(bad_word in header_line for bad_word in blacklist):
                continue
            cleaned_sections.append(section)

        md_text = "".join(cleaned_sections)
        return md_text
    except Exception as e:
        log_message(f"텍스트 추출 및 필터링 오류 ({os.path.basename(pdf_path)}): {e}", "ERROR")
        return ""

def summarize_paper_optimized(c, filename, key_content):
    """단일 프롬프트로 서지 정보 추출과 요약을 동시에 수행합니다. (인용구 혼동 방지 로직)"""

    prompt = f"""당신은 세계 최고의 학술 분석가입니다. 제공된 논문을 분석하여 정확한 정보를 추출하세요.

[작업 지시]
1. **서지 정보 정밀 추출 (중요: 본문 인용구와 혼동 금지!)**:
   - **연도(year)**: 논문의 공식 발행 연도 4자리.
   - **제1저자 성명(full_name)**: 반드시 **논문 제목(Title) 바로 아래에 적힌 저자 목록**에서 첫 번째 이름을 찾으세요.
     * 경고: 본문 중간에 언급된 `Welch et al.` 또는 `Ober et al.` 등은 인용된 타인의 논문 저자일 뿐입니다. 절대 이들을 이 논문의 저자로 적지 마세요.
     * 실제 이 논문의 저자(Primary Author)는 논문 맨 앞부분에 있습니다.
   - **저자 성(author)**: 위 성명에서 **'성(Surname)'**만 따로 분리하세요. (예: Chen)
   - **키워드(keyword)**: 논문의 주제를 관통하는 핵심 영문 키워드 1~2개. 2개일 경우 반드시 하이픈('-')으로 연결하세요 (예: AI-Ethics).

2. **논문 요약**: 한국어로 핵심 내용을 요약하세요. (# 요약, ## 핵심 내용, ## 결론 형식 준수)

--- 논문 내용 ({filename}) ---
{key_content}
--- 내용 끝 ---

**응답 형식**: 반드시 아래의 JSON 구조로만 응답하세요. 다른 부연 설명은 절대 생략하세요.
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
        response = c.models.generate_content(
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
    c = get_client()
    history = load_history()

    source_dir = os.path.join(CLASSIFIED_DIR, "논문")
    if not os.path.exists(source_dir):
        print("요약할 논문이 없습니다. (output/classified/논문 폴더 없음)")
        return

    processed_dir = os.path.join(source_dir, PROCESSED_SUBDIR)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(SUMMARIES_DIR, exist_ok=True)

    files = [f for f in os.listdir(source_dir) if f.lower().endswith(".pdf")]

    if not files:
        print("요약할 새로운 논문이 없습니다.")
        return

    log_message(f"총 {len(files)}개의 논문을 요약 처리합니다. (통합 페이로드 최적화 모드)")

    for filename in files:
        file_path = os.path.join(source_dir, filename)
        if not os.path.exists(file_path): continue

        file_hash = calculate_file_hash(file_path)

        if file_hash in history and history[file_hash].get("summarized"):
            # 이미 요약된 경우 processed 폴더로 이동 (원본 폴더에 남아있는 경우 대비)
            target_name = history[file_hash].get("new_filename", filename)
            shutil.move(file_path, os.path.join(processed_dir, target_name))
            continue

        print(f"🚀 요약 시작: {filename}...")

        # 1. 핵심 섹션 추출 (Tail-cut)
        key_content = extract_key_sections(file_path)
        if not key_content: continue

        # 2. 통합 요약 및 메타데이터 추출
        result = summarize_paper_optimized(c, filename, key_content)
        if not result: continue

        # 3. 결과 처리
        meta = result["metadata"]
        year = meta.get("year", "0000")
        author = meta.get("author", "Unknown")
        keyword = meta.get("keyword", "Paper")

        # 파일명: 연도_저자_키워드.pdf
        clean_author = sanitize_filename(author)
        clean_keyword = sanitize_filename(keyword)
        new_filename = f"{year}_{clean_author}_{clean_keyword}.pdf"
        summary_filename = f"{year}_{clean_author}_{clean_keyword}_summary.md"

        # 요약본 저장 (utf-8-sig)
        summary_path = os.path.join(SUMMARIES_DIR, summary_filename)
        with open(summary_path, "w", encoding="utf-8-sig") as f:
            f.write(result["summary"])

        # 원본 파일 이동 및 이름 변경
        target_path = os.path.join(processed_dir, new_filename)
        cnt = 1
        while os.path.exists(target_path):
            target_path = os.path.join(processed_dir, f"{year}_{clean_author}_{clean_keyword}_{cnt}.pdf")
            cnt += 1

        shutil.move(file_path, target_path)
        final_filename = os.path.basename(target_path)

        # 히스토리 업데이트
        if file_hash not in history:
            history[file_hash] = {"filename": filename, "classified": True}

        history[file_hash].update({
            "summarized": True,
            "new_filename": final_filename,
            "summary_file": summary_filename,
            "metadata": meta
        })
        save_history(history)

        print(f"  └ ✅ 요약 완료: {final_filename}")
        # RPM 제한 준수
        time.sleep(10)

if __name__ == "__main__":
    process_summaries()
