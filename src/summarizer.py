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
    instruction = """당신은 세계 최고의 학술 분석가이자 서지 정보 추출 전문가입니다. 논문을 분석하여 핵심 정보를 JSON으로 추출하고 한국어 요약본을 작성하세요.

### 정보 추출 (Metadata) 규칙:
1. 연도 (year): 논문의 공식 발행 연도 4자리 (YYYY).
2. 저자 (author): 반드시 논문 제목(Title) 바로 아래에 나열된 저자 목록에서 첫 번째 저자의 '성(Surname)'만 추출하세요 (예: Chen). 본문 인용구와 혼동하지 마세요.
3. 키워드 (keyword): 논문의 주제를 관통하는 핵심 영문 키워드 1~2개. 2개일 경우 하이픈('-')으로 연결하세요 (예: AI-Ethics).

### 요약본 (Summary) 작성 규칙:
- 반드시 한국어(전문 용어는 영어 병기 가능)로 작성하세요.
- 마크다운 형식을 사용하여 아래의 구조를 반드시 지키세요:
  - # 요약: 전체적인 논문의 목적과 성과를 2~3문장으로 기술.
  - ## 핵심 내용: 주요 제안 방법, 실험 결과, 데이터 등을 불렛 포인트로 기술.
  - ## 결론: 논문이 시사하는 바와 최종 결론.

### 출력 형식 규칙:
- 반드시 아래의 JSON 구조로만 응답을 시작하세요. 부연 설명은 JSON 블록 이후에 작성하세요.
```json
{
  "metadata": {
    "year": "YYYY",
    "author": "Surname",
    "keyword": "Keyword-Topic"
  }
}
```
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
            full_json = json.loads(json_match.group(1))
            metadata = full_json.get('metadata', {})
            summary = text.replace(json_match.group(0), "").strip()
            # 요약 양식 강제 추출
            summary_match = re.search(r'(#\s*요약.*)', summary, re.DOTALL | re.IGNORECASE)
            if summary_match:
                summary = summary_match.group(1)
            return metadata, summary
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
    
    if not os.path.exists(paper_dir):
        print("요약할 논문이 없습니다. (output/classified/논문 폴더 없음)")
        return

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
                keyword = str(info.get('keyword', 'Paper')).strip()
                
                # 파일명 안전하게 처리: 연도_저자_키워드.pdf
                new_base = f"{year}_{author}_{keyword}"
                new_base = re.sub(r'[\\/*?:"<>|]', "", new_base).replace(" ", "_")
                
                new_pdf_name = f"{new_base}.pdf"
                new_summary_name = f"{new_base}_summary.md"
                
                # UTF-8-SIG로 저장
                with open(os.path.join(summary_output_dir, new_summary_name), "w", encoding="utf-8-sig") as f:
                    f.write(summary)
                
                target_path = os.path.join(processed_dir, new_pdf_name)
                cnt = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(processed_dir, f"{year}_{author}_{keyword}_{cnt}.pdf")
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
