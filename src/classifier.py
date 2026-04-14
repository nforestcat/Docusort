import os
import shutil
import time
import subprocess
import re
import sys
import pymupdf4llm
from datetime import datetime
from src.utils import log_message, extract_zip_files, calculate_file_hash, load_history, save_history

# 표준 출력 인코딩을 UTF-8로 강제 설정 (Windows 환경 대응)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_system_instruction() -> str:
    """gemini.md 파일에서 프로젝트 가이드라인을 읽어 시스템 지시문으로 활용합니다."""
    md_path = "gemini.md"
    if os.path.exists(md_path):
        try:
            with open(md_path, "r", encoding="utf-8-sig") as f:
                content = f.read()
                # 핵심 분류 가이드라인 부분만 추출하거나 전체를 활용
                return content
        except Exception as e:
            log_message(f"시스템 지시문 로드 실패: {e}", "WARNING")
    
    # 기본 지시문 (파일 로드 실패 시)
    return "당신은 문서 분류 전문가입니다. 제시된 텍스트를 보고 [논문, 행정서식, 과제금융, 기술매뉴얼, 일반안내] 중 하나로 분류하세요."

def call_gemini_cli(prompt: str, input_text: str = None) -> str:
    """Gemini CLI를 사용하여 AI 응답을 가져옵니다."""
    custom_env = os.environ.copy()
    custom_env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    custom_env["PYTHONIOENCODING"] = "utf-8"
    
    try:
        # 시스템 지시문과 개별 파일 프롬프트를 결합
        system_info = get_system_instruction()
        full_prompt = f"{system_info}\n\n[현재 작업]\n{prompt}"
        
        cmd = ["gemini.cmd", "-p", full_prompt]
        
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
            return result.stdout.strip()
        else:
            log_message(f"CLI 호출 실패: {result.stderr}", "ERROR")
            return ""
    except Exception as e:
        log_message(f"CLI 예외 발생: {str(e)}", "ERROR")
        return ""

def classify_document(filename: str, text: str) -> str:
    """CLI를 통해 카테고리를 판단합니다."""
    
    # 앞부분(7000자)과 뒷부분(3000자) 샘플링
    sample_text = text[:7000] + "\n\n...[중략]...\n\n" + text[-3000:] if len(text) > 10000 else text
    
    instruction = f"""파일명: {filename}
위 문서를 분석하여 [논문, 행정서식, 과제금융, 기술매뉴얼, 일반안내] 중 하나로 분류하세요.
반드시 아래 형식을 포함하여 답변하세요.
RESULT: [카테고리]"""
    
    response_text = call_gemini_cli(instruction, input_text=sample_text)
    
    # 정교한 파싱 (대괄호 유무와 관계없이 추출)
    match = re.search(r'RESULT:\s*\[?(논문|행정서식|과제금융|기술매뉴얼|일반안내)\]?', response_text)
    if match:
        return match.group(1)
    
    # 키워드 직접 매칭 (Fallback)
    valid_categories = ["논문", "행정서식", "과제금융", "기술매뉴얼", "일반안내"]
    for cat in valid_categories:
        if cat in response_text:
            return cat
            
    return "일반안내"

def handle_pre_processing(input_dir: str):
    """ZIP 파일 압축 해제 및 전처리"""
    processed_zips_dir = os.path.join(input_dir, "processed_zips")
    zip_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.zip')]
    if zip_files:
        os.makedirs(processed_zips_dir, exist_ok=True)
        for zip_name in zip_files:
            zip_path = os.path.join(input_dir, zip_name)
            if extract_zip_files(zip_path, input_dir):
                shutil.move(zip_path, os.path.join(processed_zips_dir, zip_name))

def process_all_documents():
    """모든 문서를 스캔하고 분류합니다 (Hash 기반 중복 방지 적용)."""
    input_dir = "input"
    output_base_dir = "output/classified"
    history = load_history()
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)

    handle_pre_processing(input_dir)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not files:
        log_message("분류할 새로운 문서가 없습니다.")
        return

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        file_hash = calculate_file_hash(file_path)
        
        # 1. 중복 확인 (히스토리에 있고 이미 분류된 경우 삭제 후 스킵)
        if file_hash in history and history[file_hash].get("classified"):
            try:
                os.remove(file_path)
                log_message(f"🗑️ 중복 파일 삭제: {filename} (이미 처리됨)")
            except Exception as e:
                log_message(f"중복 파일 삭제 실패: {e}", "ERROR")
            continue

        print(f"\n🚀 CLI 분류 시작: {filename}...")
        
        try:
            # 마크다운 텍스트 추출
            text = pymupdf4llm.to_markdown(file_path)
        except Exception as e:
            log_message(f"텍스트 추출 오류 ({filename}): {e}", "ERROR")
            continue

        # AI 분류
        category = classify_document(filename, text)
        log_message(f"분류 완료: {filename} -> {category}")

        # 파일 이동
        target_dir = os.path.join(output_base_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        
        dest_path = os.path.join(target_dir, filename)
        
        # 이동 (이미 있으면 덮어쓰기)
        if os.path.exists(dest_path):
            os.remove(dest_path)
        shutil.move(file_path, dest_path)
        
        # 히스토리 업데이트
        history[file_hash] = {
            "filename": filename,
            "category": category,
            "classified": True,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        save_history(history)
        
        print(f"  └ {category} 폴더로 이동 및 히스토리 저장 완료.")
        time.sleep(5) # API 속도 제한 고려

if __name__ == "__main__":
    process_all_documents()
