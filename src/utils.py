import fitz  # PyMuPDF
import os
from datetime import datetime

import hashlib

def calculate_file_hash(file_path: str) -> str:
    """파일의 SHA-256 해시 값을 계산하여 고유 지문을 생성합니다."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 파일을 4K 단위로 읽어서 해시 계산 (대용량 파일 대응)
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

import zipfile
import shutil

def extract_zip_files(zip_path: str, extract_to: str):
    """ZIP 파일의 압축을 풀고 PDF 파일들을 지정된 폴더로 추출합니다."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 압축 파일 내의 모든 파일 목록 확인
            for file_info in zip_ref.infolist():
                # 폴더 구조가 있어도 파일명만 추출하여 평면적으로 저장 (중복 방지 로직 포함 권장)
                if not file_info.is_dir():
                    filename = os.path.basename(file_info.filename)
                    if not filename: continue # 디렉토리 엔트리 스킵
                    
                    target_path = os.path.join(extract_to, filename)
                    
                    # 파일명이 겹칠 경우 숫자 추가
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(target_path):
                        target_path = os.path.join(extract_to, f"{base}_{counter}{ext}")
                        counter += 1
                    
                    # 파일 추출
                    with zip_ref.open(file_info) as source, open(target_path, "wb") as target:
                        shutil.copyfileobj(source, target)
            
        log_message(f"압축 해제 완료: {zip_path}")
        return True
    except Exception as e:
        log_message(f"압축 해제 실패 ({zip_path}): {str(e)}", "ERROR")
        return False

def ensure_api_key():
    """API 키가 있는지 확인하고, 없으면 사용자에게 입력을 받아 .env를 생성합니다."""
    from dotenv import load_dotenv, set_key
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and api_key != "your_api_key_here":
        return api_key

    # 키가 없는 경우 입력 받기
    print("\n" + "!"*50)
    print("🔑 Gemini API 키가 설정되지 않았습니다.")
    print("가이드: https://aistudio.google.com/app/apikey 에서 키를 발급받으세요.")
    print("!"*50 + "\n")

    # GUI 환경인지 확인 (간이 방법)
    is_gui = False
    try:
        import __main__
        if "gui" in os.path.basename(__main__.__file__):
            is_gui = True
    except: pass

    new_key = ""
    if is_gui:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        new_key = simpledialog.askstring("API 키 설정", 
            "Google AI Studio에서 발급받은 API 키를 입력하세요:\n(https://aistudio.google.com/app/apikey)",
            parent=root)
        root.destroy()
    else:
        new_key = input("👉 API 키를 입력하고 Enter를 누르세요: ").strip()

    if new_key:
        with open(".env", "w", encoding="utf-8") as f:
            f.write(f"GEMINI_API_KEY={new_key}\n")
        print("✅ .env 파일이 성공적으로 생성되었습니다!")
        os.environ["GEMINI_API_KEY"] = new_key
        return new_key
    else:
        print("❌ API 키가 입력되지 않았습니다. 프로그램을 종료합니다.")
        os._exit(1)

def log_message(message: str, level: str = "INFO"):
    """작업 수행 내역을 automation.log 파일에 기록합니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {message}\n"
    with open("automation.log", "a", encoding="utf-8") as f:
        f.write(log_entry)

def extract_text_from_pdf(pdf_path: str) -> str:
    """PDF 파일에서 텍스트를 추출합니다."""
    text = ""
    try:
        # 파일 존재 여부 확인
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {pdf_path}")

        # PDF 열기
        doc = fitz.open(pdf_path)
        
        # 암호화 확인
        if doc.is_encrypted:
            log_message(f"PDF 암호화됨: {pdf_path}", "ERROR")
            return "[ERROR] PDF가 암호화되어 있어 텍스트를 추출할 수 없습니다."

        # 페이지별 텍스트 추출
        for page in doc:
            text += page.get_text()

        doc.close()

        # 텍스트가 비어 있는 경우 (이미지 전용 PDF 등)
        if not text.strip():
            log_message(f"텍스트가 없는 PDF(이미지 가능성): {pdf_path}", "WARNING")
            return "[WARNING] PDF에서 텍스트를 찾을 수 없습니다. (이미지 전용 PDF일 가능성)"

        return text

    except Exception as e:
        error_msg = f"PDF 처리 중 오류 발생: {str(e)}"
        log_message(error_msg, "ERROR")
        return f"[ERROR] {error_msg}"
