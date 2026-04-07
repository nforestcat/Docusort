import os
import shutil
import time
import re
import sys
import pymupdf4llm
from google import genai
from dotenv import load_dotenv
from src.utils import log_message, extract_zip_files, ensure_api_key, calculate_file_hash, load_history, save_history

# 사용할 모델 리스트 (gemma_version 브랜치: Gemma 4 전담)
MODELS = ['gemma-4-31b-it']
current_model_idx = 0
client = None

def get_client():
    """API 키를 확인하고 클라이언트를 반환합니다."""
    global client
    if client is None:
        api_key = ensure_api_key()
        client = genai.Client(api_key=api_key)
    return client

def switch_model():
    """한도 초과 시 다음 모델로 전환합니다."""
    global current_model_idx, client
    if current_model_idx < len(MODELS) - 1:
        current_model_idx += 1
        client = None # 클라이언트 재설정 유도
        return True
    return False

def classify_document(text: str) -> str:
    """텍스트 내용을 분석하여 카테고리를 반환합니다."""
    c = get_client()
    
    # 텍스트 샘플링 (앞부분 7000자 + 뒷부분 3000자)
    sample_text = text[:7000] + "\n\n...[중략]...\n\n" + text[-3000:] if len(text) > 10000 else text

    instruction = f"""당신은 고도로 정밀한 문서 분류기입니다. 제시된 텍스트(마크다운 형식)를 보고 다음 중 하나로 분류하세요:
    [기술, 금융, 일반, 논문]
    
    [분류 가이드라인]
    1. 논문: 초록(Abstract), 서론(Introduction), 저자 소속(Affiliation), 참고문헌(References) 섹션 중 2개 이상의 특징이 명확한 경우. 
    2. 기술: 제품 매뉴얼, 사양서, 기술 백서(Whitepaper), API 가이드 등 기술 정보 전달이 주된 목적인 경우.
    3. 금융: 경제 리포트, 재무제표, 증권 분석 등 금융 데이터나 경제 용어가 주된 경우.
    4. 일반: 그 외의 홍보물, 서신, 뉴스 기사 등 일상적이거나 다른 범주에 속하지 않는 경우.

    *주의: 기술적 내용이 포함된 학술 논문은 반드시 '논문'으로 분류하세요.*
    
    결과는 반드시 다음 형식으로 답변하세요:
    RESULT: [카테고리]

    문서 내용:
    {sample_text}
    """
    
    try:
        response = c.models.generate_content(
            model=MODELS[current_model_idx],
            contents=instruction
        )
        response_text = response.text.strip()
        
        match = re.search(r'RESULT:\s*\[?(논문|기술|금융|일반)\]?', response_text)
        if match:
            return match.group(1)
        
        valid_categories = ["논문", "기술", "금융", "일반"]
        for valid in valid_categories:
            if valid in response_text:
                return valid
        return "일반"
            
    except Exception as e:
        msg = str(e).lower()
        if ("429" in msg or "quota" in msg) and switch_model():
            log_message(f"한도 초과로 모델 전환: {MODELS[current_model_idx]}")
            return classify_document(text) # 전환된 모델로 재시도
        
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
                shutil.move(zip_path, os.path.join(processed_zips_dir, zip_name))
        print("✅ 압축 해제 완료.")

def process_all_documents():
    """input 폴더의 모든 파일을 처리합니다."""
    input_dir = "input"
    output_base_dir = "output/classified"
    
    # 이력 로드
    history = load_history()
    
    if not os.path.exists(input_dir):
        os.makedirs(input_dir, exist_ok=True)

    # 1. 압축 파일 전처리
    handle_pre_processing(input_dir)

    # 2. PDF 파일 리스트 확보
    files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not files:
        print("분류할 PDF 파일이 input/ 폴더에 없습니다.")
        return

    for filename in files:
        file_path = os.path.join(input_dir, filename)
        file_hash = calculate_file_hash(file_path)

        # 중복 체크
        if file_hash in history:
            entry = history[file_hash]
            # 구버전 이력(문자열만 있는 경우) 대응
            if isinstance(entry, str):
                category = "논문" if "논문" in entry or "_" in entry else "일반"
            else:
                category = entry.get("category", "일반")
            
            print(f"\n⚠️ 중복 내용 감지: {filename} (이미 {category}로 분류됨)")
            target_dir = os.path.join(output_base_dir, category)
            os.makedirs(target_dir, exist_ok=True)
            
            # 논문인 경우 이미 이름이 바뀌었을 수 있으므로 이력의 파일명 활용 고려
            # 여기서는 단순히 해당 카테고리 폴더로 이동만 처리 (요약기에서 최종 정리)
            shutil.move(file_path, os.path.join(target_dir, filename))
            continue

        print(f"\n🚀 문서 분석 중: {filename}...")
        
        try:
            text = pymupdf4llm.to_markdown(file_path)
        except Exception as e:
            log_message(f"텍스트 추출 실패: {filename} ({str(e)})", "ERROR")
            continue

        category = classify_document(text)
        log_message(f"파일 분류 완료: {filename} -> {category}")

        # 이력에 임시 기록 (요약 단계에서 최종 파일명으로 업데이트됨)
        history[file_hash] = {"category": category, "original_filename": filename}
        save_history(history)

        target_dir = os.path.join(output_base_dir, category)
        os.makedirs(target_dir, exist_ok=True)
        
        shutil.move(file_path, os.path.join(target_dir, filename))
        print(f"  └ ✅ {category} 폴더로 이동 완료.")

        # RPM 15 제한 준수 (최소 4초 대기 필요, 안전하게 5초 설정)
        time.sleep(5)

if __name__ == "__main__":
    process_all_documents()
