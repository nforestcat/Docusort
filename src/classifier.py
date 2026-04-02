import os
import shutil
import time
from google import genai
from dotenv import load_dotenv
from src.utils import log_message, extract_text_from_pdf

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    log_message("GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.", "ERROR")
    raise ValueError("GEMINI_API_KEY is missing.")

# 최신 SDK 클라이언트 초기화
client = genai.Client(api_key=api_key)
MODEL_NAME = 'gemini-3.1-flash-lite-preview'

def classify_document(text: str) -> str:
    """텍스트 내용을 분석하여 카테고리를 반환합니다."""
    prompt = f"""
    당신은 전문 문서 분류 시스템입니다. 아래 문서의 내용을 분석하여 다음 중 가장 적절한 카테고리 하나만 단어로 답하세요:
    [기술, 금융, 일반, 논문]

    *주의: 제목에 'Abstract', 'References', 'Introduction' 등이 포함되거나 학술적인 형식의 문서는 반드시 '논문'으로 분류하세요.

    문서 내용 (일부):
    {text[:2000]}
    """
    
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        category = response.text.strip()
        
        valid_categories = ["기술", "금융", "일반", "논문"]
        if category not in valid_categories:
            for valid in valid_categories:
                if valid in category:
                    return valid
            return "일반"
            
        return category
    except Exception as e:
        log_message(f"Gemini 분류 중 오류 발생: {str(e)}", "ERROR")
        return "일반"

def process_all_documents():
    """input 폴더의 모든 PDF 파일을 분류합니다."""
    input_dir = "input"
    output_base_dir = "output/classified"
    
    if not os.path.exists(input_dir):
        log_message("input 폴더가 존재하지 않습니다.", "ERROR")
        return

    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not files:
        print("분류할 PDF 파일이 input/ 폴더에 없습니다.")
        return

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        print(f"처리 중: {filename}...")
        
        text = extract_text_from_pdf(file_path)
        
        if text.startswith("[ERROR]") or text.startswith("[WARNING]"):
            log_message(f"파일 스킵: {filename} ({text})", "WARNING")
            continue

        category = classify_document(text)
        log_message(f"파일 분류 완료: {filename} -> {category}")

        target_dir = os.path.join(output_base_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        
        shutil.move(file_path, os.path.join(target_dir, filename))
        print(f"이동 완료: {category} 폴더로")

        # RPM 제한 준수
        time.sleep(15)

if __name__ == "__main__":
    process_all_documents()
