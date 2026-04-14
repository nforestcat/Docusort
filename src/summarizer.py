import os
import shutil
import time
import json
import re
import subprocess
import pymupdf4llm
from src.utils import log_message, calculate_file_hash, load_history, save_history, parse_json_response

# 설정
CLASSIFIED_DIR = "output/classified"
SUMMARIES_DIR = "output/summaries"
PROCESSED_SUBDIR = "processed"

# 요약용 모델 (사용자 요청: gemini-3-flash)
MODEL_NAME = "gemini-3-flash"

def get_system_instruction() -> str:
    """gemini.md 파일에서 프로젝트 가이드라인을 읽어 시스템 지시문으로 활용합니다."""
    md_path = "gemini.md"
    if os.path.exists(md_path):
        try:
            with open(md_path, "r", encoding="utf-8-sig") as f:
                return f.read()
        except Exception as e:
            log_message(f"시스템 지시문 로드 실패: {e}", "WARNING")
    
    return "당신은 논문 분석가이자 서지 정보 추출 전문가입니다."

def call_gemini_cli(prompt: str, input_text: str = None) -> str:
    """Gemini CLI를 호출합니다."""
    custom_env = os.environ.copy()
    custom_env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    custom_env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        # 시스템 지시문과 개별 작업 프롬프트를 결합
        system_info = get_system_instruction()
        full_prompt = f"{system_info}\n\n[현재 작업: 논문 요약 및 정보 추출]\n{prompt}"
        
        cmd = ["gemini.cmd", "-m", MODEL_NAME, "-p", full_prompt]
        result = subprocess.run(
            cmd, input=input_text, capture_output=True, text=True,
            env=custom_env, shell=True, errors='replace', encoding='utf-8'
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"ERROR: {result.stderr}"
    except Exception as e: return f"ERROR: {str(e)}"

def extract_key_sections(pdf_path: str) -> str:
    """논문에서 불필요한 섹션을 제거하고 핵심만 추출합니다."""
    try:
        md_text = pymupdf4llm.to_markdown(pdf_path)
        
        # 1. 후반부 절삭 (References 이후 제거)
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
            if match and match.start() < earliest_index:
                earliest_index = match.start()
        
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
        
        return "".join(cleaned_sections)
    except Exception as e:
        log_message(f"텍스트 필터링 오류 ({os.path.basename(pdf_path)}): {e}", "ERROR")
        return ""

def summarize_paper_optimized(filename, key_content):
    """구조화된 JSON 응답을 요청하여 서지 정보와 요약을 동시에 수행합니다."""
    
    # gemini.md에 상세 규칙이 있으므로, 여기서는 실행에 필요한 최소한의 데이터 구조만 강조
    prompt = f"""파일명: {filename}
위 논문을 분석하여 아래 JSON 구조로 응답하세요. 다른 부연 설명은 절대 금지입니다.

[응답 필수 형식]
{{
  "metadata": {{
    "year": "YYYY",
    "author": "Surname",
    "keyword": "Keyword-Topic"
  }},
  "summary": "# 요약\\n...\\n## 핵심 내용\\n...\\n## 결론\\n..."
}}

--- 논문 내용 ---
{key_content}
"""

    return call_gemini_cli(prompt)

def sanitize_filename(name):
    """파일명으로 사용할 수 없는 문자를 제거합니다."""
    clean_name = re.sub(r'[\\/*?:"<>|]', "", str(name))
    return clean_name.replace(" ", "_")

def process_summaries():
    """논문을 요약하고 이름을 변경합니다."""
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

    log_message(f"총 {len(files)}개의 논문을 요약 처리합니다. (CLI JSON 모드)")

    for filename in files:
        file_path = os.path.join(source_dir, filename)
        file_hash = calculate_file_hash(file_path)

        if file_hash in history and history[file_hash].get("summarized"):
            continue

        log_message(f"요약 시작: {filename}...")
        
        key_content = extract_key_sections(file_path)
        if not key_content: continue

        # 디버그용 필터링 텍스트 저장
        debug_path = f"output/debug_filtered_{os.path.splitext(filename)[0]}.md"
        with open(debug_path, "w", encoding="utf-8-sig") as f:
            f.write(key_content)

        response_text = summarize_paper_optimized(filename, key_content)
        result = parse_json_response(response_text)
        
        if not result or "metadata" not in result or "summary" not in result:
            log_message(f"요약 실패 또는 JSON 파싱 오류 ({filename})", "ERROR")
            continue

        meta = result["metadata"]
        year = sanitize_filename(meta.get("year", "0000"))
        author = sanitize_filename(meta.get("author", "Unknown"))
        keyword = sanitize_filename(meta.get("keyword", "Paper"))
        
        new_filename = f"{year}_{author}_{keyword}.pdf"
        summary_filename = f"{year}_{author}_{keyword}_summary.md"

        with open(os.path.join(SUMMARIES_DIR, summary_filename), "w", encoding="utf-8-sig") as f:
            f.write(result["summary"])

        target_path = os.path.join(processed_dir, new_filename)
        if os.path.exists(target_path):
            base, ext = os.path.splitext(new_filename)
            target_path = os.path.join(processed_dir, f"{base}_{int(time.time())}{ext}")

        shutil.move(file_path, target_path)

        if file_hash not in history:
            history[file_hash] = {"filename": filename, "classified": True}
        history[file_hash].update({
            "summarized": True,
            "new_filename": os.path.basename(target_path),
            "summary_file": summary_filename,
            "metadata": meta,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        save_history(history)
        
        log_message(f"✅ 요약 완료: {filename} -> {os.path.basename(target_path)}")
        time.sleep(10)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    process_summaries()
