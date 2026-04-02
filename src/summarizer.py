import os
import re
import time
import json
import shutil
from google import genai
from dotenv import load_dotenv
from src.utils import log_message, extract_text_from_pdf, calculate_file_hash
import fitz

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 사용할 모델 리스트
MODELS = ['gemini-3-flash-preview', 'gemini-3.1-flash-lite-preview']
current_model_idx = 0
client = genai.Client(api_key=api_key)

HISTORY_FILE = "output/processed_history.json"

def load_history():
    """기존 처리 이력을 로드합니다."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {}
    return {}

def save_history(history):
    """새로운 처리 이력을 저장합니다."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def switch_model():
    global current_model_idx
    if current_model_idx < len(MODELS) - 1:
        current_model_idx += 1
        return True
    return False

def clean_paper_text(text: str) -> str:
    ref_patterns = [r'\n\s*References\s*\n', r'\n\s*REFERENCES\s*\n', r'\n\s*참고문헌\s*\n']
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return text[:match.start()]
    return text

def summarize_and_rename_info(text: str):
    prompt = f"""
    당신은 전문적인 학술 논문 분석가입니다. 주어진 논문을 읽고 요약본과 파일 이름 정보를 제공하세요.
    [파일 이름 규칙] 응답 최상단에 반드시 아래 JSON 형식으로 포함하세요:
    ```json
    {{"year": "YYYY", "author": "영문성", "keywords": "키워드_언더바"}}
    ```
    [요약 규칙] 마크다운 작성, 한국어 사용.
    텍스트: {text[:15000]}
    """
    try:
        response = client.models.generate_content(model=MODELS[current_model_idx], contents=prompt)
        return response.text if response.text else "요약 실패"
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "quota" in msg:
            return "RPD_EXCEEDED" if "day" in msg else "QUOTA_EXCEEDED"
        return f"ERROR: {str(e)}"

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

        # 1. 내용 기반 중복 체크 (해시 값 비교)
        if file_hash in history:
            print(f"\n⚠️ 중복 내용 감지: {filename}")
            print(f"   (이미 '{history[file_hash]}' 라는 이름으로 처리되었습니다.)")
            # API 호출 없이 바로 이동
            target_name = history[file_hash]
            shutil.move(file_path, os.path.join(processed_dir, target_name))
            continue

        print(f"\n🚀 신규 분석 시작: {filename}")
        
        # 2. 텍스트 추출 및 분석
        try:
            doc = fitz.open(file_path); full_text = "".join([p.get_text() for p in doc]); doc.close()
            cleaned_text = clean_paper_text(full_text)
        except Exception as e: print(f"❌ 오류: {e}"); continue

        success = False
        for _ in range(3):
            raw = summarize_and_rename_info(cleaned_text)
            if raw == "RPD_EXCEEDED":
                if switch_model(): continue
                else: print("❌ 한도 초과"); return
            if raw == "QUOTA_EXCEEDED":
                time.sleep(30); continue
            
            info, summary = parse_response(raw)
            if info:
                new_base = f"[{info.get('year', '0000')}]_[{info.get('author', 'Unknown')}]_[{info.get('keywords', 'Paper')}]"
                new_base = re.sub(r'[\\/*?:"<>|]', "", new_base)
                
                new_pdf_name = f"{new_base}.pdf"
                new_summary_name = f"{new_base}_summary.md"
                
                # 저장
                with open(os.path.join(summary_output_dir, new_summary_name), "w", encoding="utf-8") as f:
                    f.write(summary)
                
                target_path = os.path.join(processed_dir, new_pdf_name)
                # 동일 이름 충돌 방지
                cnt = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(processed_dir, f"{new_base}_{cnt}.pdf")
                    cnt += 1
                
                shutil.move(file_path, target_path)
                final_name = os.path.basename(target_path)
                print(f"✅ 처리 완료: {final_name}")
                
                # 이력에 해시 저장
                history[file_hash] = final_name
                save_history(history)
                success = True; break
            else: time.sleep(5)

        if success: time.sleep(20)
        else: print(f"❌ 최종 실패: {filename}")

if __name__ == "__main__":
    process_summaries()
