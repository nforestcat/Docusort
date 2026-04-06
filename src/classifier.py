import os
import shutil
import time
import subprocess
import re
import sys
from src.utils import log_message, extract_zip_files

# 표준 출력 인코딩을 UTF-8로 강제 설정 (Windows 환경 대응)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def call_gemini_cli(prompt: str, file_path: str = None) -> str:
    """Gemini CLI를 사용합니다. 파일 경로가 있으면 -f 옵션을 사용합니다."""
    custom_env = os.environ.copy()
    custom_env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    custom_env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        if file_path:
            cmd = ["gemini.cmd", "-f", file_path, "-p", prompt]
        else:
            cmd = ["gemini.cmd", "-p", prompt]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=custom_env,
            shell=True,
            errors='replace',
            encoding='utf-8'
        )
        
        if result.returncode == 0:
            output = result.stdout.strip()
            valid_categories = ["논문", "기술", "금융", "일반"]
            lines = [l.strip() for l in output.split('\n') if l.strip()]
            for line in reversed(lines):
                for cat in valid_categories:
                    if cat in line: return cat
            for cat in valid_categories:
                if cat in output: return cat
            return "일반"
        else:
            log_message(f"CLI 호출 실패: {result.stderr}", "ERROR")
            return "일반"
    except Exception as e:
        log_message(f"CLI 예외 발생: {str(e)}", "ERROR")
        return "일반"

def classify_document(file_path: str) -> str:
    """CLI를 통해 카테고리를 판단합니다."""
    instruction = """당신은 고도로 정밀한 문서 분류기입니다. 제시된 파일을 보고 다음 중 하나로 분류하세요:
    [기술, 금융, 일반, 논문]
    
    결과는 반드시 다음 형식을 따르세요:
    RESULT: [카테고리]
    """
    
    response_text = call_gemini_cli(instruction, file_path=file_path)
    
    match = re.search(r'RESULT:\s*\[?(논문|기술|금융|일반)\]?', response_text)
    if match:
        return match.group(1)
    
    valid_categories = ["논문", "기술", "금융", "일반"]
    for valid in valid_categories:
        if valid in response_text:
            return valid
    return "일반"

def handle_pre_processing(input_dir: str):
    processed_zips_dir = os.path.join(input_dir, "processed_zips")
    zip_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.zip')]
    if zip_files:
        os.makedirs(processed_zips_dir, exist_ok=True)
        for zip_name in zip_files:
            zip_path = os.path.join(input_dir, zip_name)
            if extract_zip_files(zip_path, input_dir):
                shutil.move(zip_path, os.path.join(processed_zips_dir, zip_name))

def process_all_documents():
    input_dir = "input"
    output_base_dir = "output/classified"
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)

    handle_pre_processing(input_dir)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    for filename in files:
        file_path = os.path.join(input_dir, filename)
        print(f"CLI 분류 시작 (직접 파일 처리): {filename}...")
        
        category = classify_document(file_path)
        log_message(f"분류 완료: {filename} -> {category}")

        target_dir = os.path.join(output_base_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        
        dest_path = os.path.join(target_dir, filename)
        shutil.move(file_path, dest_path)
        print(f"  └ {category} 폴더로 이동 완료.")
        
        time.sleep(5)

if __name__ == "__main__":
    process_all_documents()
