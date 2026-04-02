import os
import shutil
import time
import subprocess
from src.utils import log_message, extract_text_from_pdf, extract_zip_files

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
            return "일반"
    except Exception as e:
        log_message(f"CLI 호출 중 예외 발생: {str(e)}", "ERROR")
        return "일반"

def classify_document(text: str) -> str:
    """CLI를 통해 카테고리를 판단합니다."""
    prompt = f"""
    당신은 전문 문서 분류 시스템입니다. 아래 문서의 내용을 분석하여 다음 중 가장 적절한 카테고리 하나만 단어로 답하세요:
    [기술, 금융, 일반, 논문]

    *주의: 제목에 'Abstract', 'References', 'Introduction' 등이 포함되거나 학술적인 형식의 문서는 반드시 '논문'으로 분류하세요.

    문서 내용 (일부):
    {text[:2000]}
    """
    
    category = call_gemini_cli(prompt)
    
    # 결과 정제 (CLI 응답에 불필요한 문자가 포함될 경우 대비)
    valid_categories = ["기술", "금융", "일반", "논문"]
    for valid in valid_categories:
        if valid in category:
            return valid
    return "일반"

def handle_pre_processing(input_dir: str):
    """압축 파일 등을 미리 처리합니다."""
    processed_zips_dir = os.path.join(input_dir, "processed_zips")
    zip_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.zip')]
    if zip_files:
        os.makedirs(processed_zips_dir, exist_ok=True)
        for zip_name in zip_files:
            zip_path = os.path.join(input_dir, zip_name)
            if extract_zip_files(zip_path, input_dir):
                shutil.move(zip_path, os.path.join(processed_zips_dir, zip_name))

def process_all_documents():
    """input 폴더의 모든 PDF 파일을 분류합니다."""
    input_dir = "input"
    output_base_dir = "output/classified"
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)

    handle_pre_processing(input_dir)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        print(f"CLI 분류 중: {filename}...")
        
        text = extract_text_from_pdf(file_path)
        if text.startswith("[ERROR]") or text.startswith("[WARNING]"):
            continue

        category = classify_document(text)
        log_message(f"CLI 분류 완료: {filename} -> {category}")

        target_dir = os.path.join(output_base_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(file_path, os.path.join(target_dir, filename))
        
        # CLI 호출 간 약간의 대기
        time.sleep(2)

if __name__ == "__main__":
    process_all_documents()
