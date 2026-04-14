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

    instruction = f"""당신은 대학원 연구실 및 행정 환경의 문서 분류 전문가입니다. 제시된 텍스트(마크다운 형식)를 분석하여 다음 5가지 카테고리 중 가장 적합한 하나를 판별하세요.

    [카테고리 후보 및 판별 기준]
    1. 논문 (Paper): 학술지 논문, 컨퍼런스 발표 자료, 학위 논문 등 학술적 연구 성과물. 
       - 특징: 초록(Abstract), 서론(Introduction), 결론, 참고문헌(References) 형식이 뚜렷함.
    2. 행정서식 (Admin Forms): 각종 신청서, 결과 보고서 양식, 동의서, 서약서 등 행정 처리에 필요한 서식.
    3. 과제금융 (Grant/Finance): 연구 과제 예산, 연구비 집행 내역서, 견적서, 지출 증빙, 과제 제안서(RFP) 등.
    4. 기술매뉴얼 (Tech Manual): 실험 장비 매뉴얼, 소프트웨어 사용 가이드, MSDS, 기술 사양서.
    5. 일반안내 (General Info): 학과 공지사항, 세미나 포스터, 강의 계획서(Syllabus), 학사 일정 등.

    결과는 반드시 다음 형식으로만 답변하세요:
    RESULT: [카테고리명] (예: RESULT: 논문)

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
        match = re.search(r'RESULT:\s*(논문|행정서식|과제금융|기술매뉴얼|일반안내)', response_text)
        if match:
            return match.group(1)
        
        # 형식이 틀린 경우 텍스트에서 카테고리 포함 여부 확인
        valid_categories = ["논문", "행정서식", "과제금융", "기술매뉴얼", "일반안내"]
        for valid in valid_categories:
            if valid in response_text:
                return valid
        return "일반안내"
            
    except Exception as e:
        log_message(f"Gemini 분류 중 오류 발생: {str(e)}", "ERROR")
        return "일반안내"

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
