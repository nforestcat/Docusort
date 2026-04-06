import os
import re
import time
import json
import shutil
import sys
import pymupdf4llm
from google import genai
from dotenv import load_dotenv
from src.utils import log_message, calculate_file_hash, ensure_api_key

# 사용할 모델 리스트
MODELS = ['gemini-3.1-flash-lite-preview', 'gemini-1.5-flash']
current_model_idx = 0
client = None

def get_client():
    """API 키를 확인하고 클라이언트를 반환합니다."""
    global client
    if client is None:
        api_key = ensure_api_key()
        client = genai.Client(api_key=api_key)
    return client

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
    global current_model_idx, client
    if current_model_idx < len(MODELS) - 1:
        current_model_idx += 1
        client = None # 클라이언트 재설정 유도
        return True
    return False

def clean_paper_text(text: str) -> str:
    """참고문헌(References) 섹션 이후의 텍스트는 분석에서 제외합니다."""
    # 특수 기호와 헤더를 포함하는 더 유연한 패턴
    ref_patterns = [
        r'\n\s*#*\s*.*References.*',
        r'\n\s*#*\s*.*REFERENCES.*',
        r'\n\s*#*\s*.*참고문헌.*'
    ]
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return text[:match.start()]
    return text

def summarize_and_rename_info(text: str):
    """Gemini SDK를 사용하여 요약 및 서지 정보를 추출합니다."""
    c = get_client()
    instruction = """당신은 전문적인 학술 논문 분석가이자 서지 정보 추출 전문가입니다. 
제시된 텍스트 전체를 분석하여 서론이나 부연 설명 없이 즉시 아래 JSON 블록으로 시작하세요. 그 뒤에 요약본을 작성하세요.

[중요 지침]
- 반드시 표준 한국어(UTF-8)를 사용하여 한글이 깨지지 않도록 작성하세요.
- JSON 블록 이후에는 마크다운 형식의 한국어 요약을 작성하세요.

[JSON 형식]
```json
{
  "year": "출판 연도 (예: 2024)",
  "author": "대표 저자의 성(Surname) (예: Smith)",
  "keywords": "핵심 키워드 2-3개를 언더바(_)로 연결 (예: AI_Ethics_Policy)"
}
```

[요약 양식] 한국어(영어 병기) 마크다운. `# 요약`, `## 핵심 내용`, `## 결론` 형식을 반드시 지키세요.
"""
    try:
        # SDK에서는 contents에 텍스트를 직접 포함
        response = c.models.generate_content(
            model=MODELS[current_model_idx], 
            contents=[instruction, text]
        )
        return response.text if response.text else "요약 실패"
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "quota" in msg:
            return "RPD_EXCEEDED" if "day" in msg else "QUOTA_EXCEEDED"
        return f"ERROR: {str(e)}"

def parse_response(text: str):
    # 0. <thought> 블록 제거
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            info = json.loads(json_match.group(1))
            summary = text.replace(json_match.group(0), "").strip()
            # 요약 양식 강제 추출
            summary_match = re.search(r'(#\s*요약.*)', summary, re.DOTALL | re.IGNORECASE)
            if summary_match:
                summary = summary_match.group(1)
            return info, summary
        except: pass
    return None, text

def process_summaries():
    paper_dir = "output/classified/논문"
    summary_output_dir = "output/summaries"
    processed_dir = os.path.join(paper_dir, "processed")
    
    # 여기서 키 체크 유도
    get_client()
    
    history = load_history()
    os.makedirs(summary_output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    files = [f for f in os.listdir(paper_dir) if f.lower().endswith('.pdf')]

    for filename in files:
        file_path = os.path.join(paper_dir, filename)
        file_hash = calculate_file_hash(file_path)

        if file_hash in history:
            print(f"\n⚠️ 중복 내용 감지: {filename}")
            target_name = history[file_hash]
            shutil.move(file_path, os.path.join(processed_dir, target_name))
            continue

        print(f"\n🚀 분석 시작 (SDK + 전체 텍스트): {filename}")
        
        try:
            # pymupdf4llm으로 교체
            full_text = pymupdf4llm.to_markdown(file_path)
            cleaned_text = clean_paper_text(full_text)
        except Exception as e: 
            print(f"❌ 추출 오류: {e}")
            continue

        success = False
        for _ in range(3):
            raw = summarize_and_rename_info(cleaned_text)
            if raw == "RPD_EXCEEDED":
                if switch_model(): 
                    print(f"🔄 모델 전환: {MODELS[current_model_idx]}")
                    continue
                else: 
                    print("❌ 모든 모델 일일 한도 초과")
                    return
            if raw == "QUOTA_EXCEEDED":
                print("⏳ 분당 한도 초과, 30초 대기...")
                time.sleep(30); continue
            
            info, summary = parse_response(raw)
            if info:
                year = str(info.get('year', '0000')).strip()
                author = str(info.get('author', 'Unknown')).strip()
                keywords = str(info.get('keywords', 'Paper')).strip()
                
                # 파일명 안전하게 처리
                new_base = f"[{year}]_[{author}]_[{keywords}]"
                new_base = re.sub(r'[\\/*?:"<>|]', "", new_base).replace(" ", "_")
                
                new_pdf_name = f"{new_base}.pdf"
                new_summary_name = f"{new_base}_summary.md"
                
                # UTF-8-SIG로 저장
                with open(os.path.join(summary_output_dir, new_summary_name), "w", encoding="utf-8-sig") as f:
                    f.write(summary)
                
                target_path = os.path.join(processed_dir, new_pdf_name)
                cnt = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(processed_dir, f"{new_base}_{cnt}.pdf")
                    cnt += 1
                
                shutil.move(file_path, target_path)
                final_name = os.path.basename(target_path)
                print(f"✅ 처리 완료: {final_name}")
                
                history[file_hash] = final_name
                save_history(history)
                success = True; break
            else: 
                print("⚠️ 응답 파싱 실패, 재시도 중...")
                time.sleep(5)

        if success: 
            # RPM 15 제한 (약 4초 이상 대기 필요, 안전하게 10초)
            time.sleep(10)
        else: 
            print(f"❌ 최종 실패: {filename}")

if __name__ == "__main__":
    process_summaries()
