import os
import shutil
import time
import re
import sys
import pymupdf4llm
from google import genai
from dotenv import load_dotenv
from src.utils import log_message, extract_zip_files, ensure_api_key

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
    """텍스트 내용을 분석하여 카테고리를 반환합니다.
    문서의 앞부분(7000자)과 뒷부분(3000자)을 조합하여 분석의 정확도를 높입니다."""
    c = get_client()
    
    # 텍스트 샘플링 (앞부분 7000자 + 뒷부분 3000자)
    sample_text = text[:7000] + "\n\n...[중략]...\n\n" + text[-3000:] if len(text) > 10000 else text

    instruction = f"""당신은 고도로 정밀한 문서 분류기입니다. 제시된 텍스트(마크다운 형식)를 보고 다음 중 하나로 분류하세요:
    [기술, 금융, 일반, 논문]
    
    [분류 가이드라인]
    1. 논문: 초록(Abstract), 서론(Introduction), 저자 소속(Affiliation), 참고문헌(References) 섹션 중 2개 이상의 특징이 명확한 경우. 
       - 예: 학술지 게재용 포맷, DOI 포함, 학술적 연구 결과 보고 등.
    2. 기술: 제품 매뉴얼, 사양서, 기술 백서(Whitepaper), API 가이드, 코드 설명서 등 구체적인 기술 정보 전달이 주된 목적인 경우.
    3. 금융: 경제 리포트, 재무제표, 증권 분석, 은행/보험 안내서 등 금융 데이터나 경제 용어가 주된 경우.
    4. 일반: 그 외의 서신, 뉴스 기사, 공지사항, 홍보물 등 일상적이거나 다른 범주에 속하지 않는 경우.

    *주의: 기술적 내용이 포함된 학술 논문은 반드시 '논문'으로 분류하세요.*
    
    결과는 반드시 다음 형식으로 답변하세요:
    RESULT: [카테고리]

    문서 내용:
    {sample_text}
    """
    
    try:
        response = c.models.generate_content(
            model=MODEL_NAME,
            contents=instruction
        )
        response_text = response.text.strip()
        
        # RESULT: [카테고리] 형식에서 추출 시도
        match = re.search(r'RESULT:\s*\[?(논문|기술|금융|일반)\]?', response_text)
        if match:
            return match.group(1)
        
        # 형식이 틀린 경우 텍스트에서 카테고리 포함 여부 확인
        valid_categories = ["논문", "기술", "금융", "일반"]
        for valid in valid_categories:
            if valid in response_text:
                return valid
        return "일반"
            
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
        print(f"\n🚀 문서 분석 중: {filename}...")
        
        try:
            # pymupdf4llm을 사용하여 마크다운 텍스트 추출
            text = pymupdf4llm.to_markdown(file_path)
        except Exception as e:
            log_message(f"텍스트 추출 실패: {filename} ({str(e)})", "ERROR")
            continue

        category = classify_document(text)
        log_message(f"파일 분류 완료: {filename} -> {category}")

        target_dir = os.path.join(output_base_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        
        shutil.move(file_path, os.path.join(target_dir, filename))
        print(f"  └ ✅ {category} 폴더로 이동 완료.")

        # RPM 15 제한 준수 (최소 4초 대기 필요, 안전하게 5초 설정)
        time.sleep(5)

if __name__ == "__main__":
    process_all_documents()
