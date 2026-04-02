import os
import re
import time
import json
import shutil
import subprocess
from src.utils import log_message, extract_text_from_pdf, calculate_file_hash
import fitz

def call_gemini_cli(prompt: str) -> str:
    """Gemini CLI를 직접 호출하여 결과를 반환합니다 (SSL 검증 무시 포함)."""
    custom_env = os.environ.copy()
    custom_env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    
    try:
        result = subprocess.run(
            ["gemini", prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            env=custom_env
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            log_message(f"CLI 에러: {result.stderr}", "ERROR")
            return "ERROR: CLI_FAILED"
    except Exception as e:
        log_message(f"CLI 호출 중 예외 발생: {str(e)}", "ERROR")
        return f"ERROR: {str(e)}"

HISTORY_FILE = "output/processed_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: return {}
    return {}

def save_history(history):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def clean_paper_text(text: str) -> str:
    ref_patterns = [r'\n\s*References\s*\n', r'\n\s*REFERENCES\s*\n', r'\n\s*참고문헌\s*\n']
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return text[:match.start()]
    return text

def summarize_and_rename_info(text: str):
    """CLI를 사용하여 요약 및 메타데이터 추출을 수행합니다."""
    prompt = f"""
    당신은 전문적인 학술 논문 분석가이자 번역가입니다. 주어진 논문을 깊이 있게 읽고 아래 규칙에 따라 요약본과 파일 이름 정보를 제공하세요.

    [파일 이름 규칙]
    응답 최상단에 반드시 아래 JSON 형식으로 포함하세요. 
    **주의: JSON 내의 모든 값(year, author, keywords)은 반드시 영문(English)과 숫자만 사용하세요.**
    ```json
    {{
      "year": "YYYY", 
      "author": "Surname_in_English", 
      "keywords": "English_Keywords_Separated_By_Underscore"
    }}
    ```

    [요약 및 번역 규칙]
    1. **용어 병기:** 핵심 전문 용어는 `한국어 (English)` 형식으로 병기하세요.
    2. **문장 스타일:** 학술적 톤을 유지하되, 한국어 문장 구조에 맞게 매끄럽게 의역하세요.
    3. **자체 검수 단계:** 최종 출력 전 오역이나 어색한 표현을 스스로 수정하세요.

    [출력 양식]
    # 📄 [논문 제목]
    ## 📚 1. 제목 및 서지 정보
    ## 🎯 2. 연구 배경 및 목적
    ## 💡 3. 핵심 가설 및 이론
    ## 🛠️ 4. 연구 방법
    ## 🏆 5. 핵심 결과
    ## 🏁 6. 결론 및 의의

    논문 텍스트:
    {text[:15000]}
    """
    return call_gemini_cli(prompt)

def parse_response(text: str):
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            info = json.loads(json_match.group(1))
            summary = text.replace(json_match.group(0), "").strip()
            return info, summary
        except: pass
    return None, text

def process_summaries():
    paper_dir = "output/classified/논문"
    summary_output_dir = "output/summaries"
    processed_dir = os.path.join(paper_dir, "processed")
    history = load_history()
    
    os.makedirs(summary_output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    files = [f for f in os.listdir(paper_dir) if f.lower().endswith('.pdf')]

    for filename in files:
        file_path = os.path.join(paper_dir, filename)
        file_hash = calculate_file_hash(file_path)

        if file_hash in history:
            print(f"\n⚠️ 중복 감지: {filename}")
            shutil.move(file_path, os.path.join(processed_dir, history[file_hash]))
            continue

        print(f"\n🚀 CLI 분석 시작: {filename}")
        try:
            doc = fitz.open(file_path); full_text = "".join([p.get_text() for p in doc]); doc.close()
            cleaned_text = clean_paper_text(full_text)
        except Exception as e: print(f"❌ 오류: {e}"); continue

        raw = summarize_and_rename_info(cleaned_text)
        if raw.startswith("ERROR"):
            print(f"❌ {raw}"); continue
            
        info, summary = parse_response(raw)
        if info:
            new_base = f"[{info.get('year', '0000')}]_[{info.get('author', 'Unknown')}]_[{info.get('keywords', 'Paper')}]"
            new_base = re.sub(r'[\\/*?:"<>|]', "", new_base)
            
            new_pdf_name = f"{new_base}.pdf"
            new_summary_name = f"{new_base}_summary.md"
            
            with open(os.path.join(summary_output_dir, new_summary_name), "w", encoding="utf-8") as f:
                f.write(summary)
            
            target_path = os.path.join(processed_dir, new_pdf_name)
            cnt = 1
            while os.path.exists(target_path):
                target_path = os.path.join(processed_dir, f"{new_base}_{cnt}.pdf")
                cnt += 1
            
            shutil.move(file_path, target_path)
            final_name = os.path.basename(target_path)
            print(f"✅ 완료: {final_name}")
            history[file_hash] = final_name
            save_history(history)
            time.sleep(5) # CLI 호출 간 간격
        else:
            print(f"❌ 정보 추출 실패: {filename}")

if __name__ == "__main__":
    process_summaries()
