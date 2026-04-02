import os
import shutil
import time
from google import genai
from dotenv import load_dotenv
from src.utils import log_message, extract_text_from_pdf, extract_zip_files, ensure_api_key

# 전역 클라이언트 변수 (지연 초기화)
client = None
MODEL_NAME = 'models/gemini-3.1-flash-lite-preview'

def get_client():
    """API 키를 확인하고 클라이언트를 반환합니다."""
    global client
    if client is None:
        api_key = ensure_api_key()
        client = genai.Client(api_key=api_key)
    return client

def classify_document(text: str) -> str:
    """텍스트 내용을 분석하여 카테고리를 반환합니다."""
    c = get_client()
    prompt = f"""
    당신은 전문 문서 분류 시스템입니다. 아래 문서의 내용을 분석하여 다음 중 가장 적절한 카테고리 하나만 단어로 답하세요:
    [기술, 금융, 일반, 논문]

    *주의: 제목에 'Abstract', 'References', 'Introduction' 등이 포함되거나 학술적인 형식의 문서는 반드시 '논문'으로 분류하세요.

    문서 내용 (일부):
    {text[:2000]}
    """
    
    try:
        response = c.models.generate_content(
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

def handle_pre_processing(input_dir: str):
    """압축 파일 등을 미리 처리합니다."""
    processed_zips_dir = os.path.join(input_dir, "processed_zips")
    
    zip_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.zip')]
    if zip_files:
        os.makedirs(processed_zips_dir, exist_ok=True)
        print(f"📦 {len(zip_files)}개의 압축 파일 발견. 압축 해제 중...")
        for zip_name in zip_files:
            zip_path = os.path.join(input_dir, zip_name)
            if extract_zip_files(zip_path, input_dir):
                # 압축 해제 성공 시 processed_zips 폴더로 이동
                shutil.move(zip_path, os.path.join(processed_zips_dir, zip_name))
        print("✅ 압축 해제 완료.")

def process_all_documents():
    """input 폴더의 모든 파일을 처리합니다."""
    input_dir = "input"
    output_base_dir = "output/classified"
    
    # 여기서 클라이언트 초기화 유도 (키 체크 포함)
    get_client()

    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)
        log_message("input 폴더 생성됨.")

    # 1. 압축 파일 전처리
    handle_pre_processing(input_dir)

    # 2. PDF 파일 리스트 확보
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
