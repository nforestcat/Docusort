import os
import shutil
import time
import subprocess
import re
import sys
import pymupdf4llm
from src.utils import log_message, extract_zip_files

# 표준 출력 인코딩을 UTF-8로 강제 설정 (Windows 환경 대응)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def call_gemini_cli(prompt: str, input_text: str = None) -> str:
    """Gemini CLI를 사용합니다."""
    custom_env = os.environ.copy()
    custom_env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    custom_env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        cmd = ["gemini.cmd", "-p", prompt]
        
        result = subprocess.run(
            cmd,
            input=input_text,
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

def classify_document(text: str) -> str:
    """CLI를 통해 카테고리를 판단합니다. 
    정확도를 위해 문서의 앞부분(7000자)과 뒷부분(3000자)을 조합하여 분석합니다."""
    
    # 앞부분(초록/서론)과 뒷부분(결론/참고문헌)을 샘플링하여 전달
    sample_text = text[:7000] + "\n\n...[중략]...\n\n" + text[-3000:] if len(text) > 10000 else text
    
    instruction = """당신은 고도로 정밀한 문서 분류기입니다. 제시된 텍스트(마크다운 형식)를 보고 다음 중 하나로 분류하세요:
    [기술, 금융, 일반, 논문]
    
    [분류 가이드라인]
    1. 논문: 초록(Abstract), 서론(Introduction), 저자 소속(Affiliation), 참고문헌(References) 섹션 중 2개 이상의 특징이 명확한 경우. 
       - 예: 학술지 게재용 포맷, DOI 포함, 학술적 연구 결과 보고 등.
    2. 기술: 제품 매뉴얼, 사양서, 기술 백서(Whitepaper), API 가이드, 코드 설명서 등 구체적인 기술 정보 전달이 주된 목적인 경우.
    3. 금융: 경제 리포트, 재무제표, 증권 분석, 은행/보험 안내서 등 금융 데이터나 경제 용어가 주된 경우.
    4. 일반: 그 외의 서신, 뉴스 기사, 공지사항, 홍보물 등 일상적이거나 다른 범주에 속하지 않는 경우.

    *주의: 기술적 내용이 포함된 학술 논문은 반드시 '논문'으로 분류하세요.*
    
    결과는 반드시 다음 형식을 따르세요:
    RESULT: [카테고리]
    """
    
    response_text = call_gemini_cli(instruction, input_text=sample_text)
    
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
        print(f"\n🚀 CLI 분류 시작 (PyMuPDF4LLM): {filename}...")
        
        try:
            # PyMuPDF4LLM을 사용하여 텍스트 추출
            text = pymupdf4llm.to_markdown(file_path)
        except Exception as e:
            print(f"DEBUG: Extraction Error: {e}")
            continue

        category = classify_document(text)
        log_message(f"분류 완료: {filename} -> {category}")

        target_dir = os.path.join(output_base_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        
        dest_path = os.path.join(target_dir, filename)
        shutil.move(file_path, dest_path)
        print(f"  └ {category} 폴더로 이동 완료.")
        
        time.sleep(5)

if __name__ == "__main__":
    process_all_documents()
