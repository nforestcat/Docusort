import os
import shutil
import time
from datetime import datetime
from google import genai
from src.utils import log_message, calculate_file_hash, load_history, save_history, parse_json_response, extract_text_from_pdf

# 설정
INPUT_DIR = "input"
CLASSIFIED_DIR = "output/classified"
CATEGORIES = ["논문", "기술", "금융", "일반"]
BATCH_SIZE = 5 # 한 번에 분류할 문서 개수

# AI 모델 설정 (gemma_version 브랜치 전용)
MODEL_NAME = "gemma-4-31b-it"

def classify_documents_batch(client, doc_batch):
    """여러 문서를 한 번에 분류합니다."""
    # doc_batch: list of (filename, sample_text)
    
    prompt = "당신은 문서 분류 전문가입니다. 아래 제공된 여러 문서(1번부터 N번까지)의 내용을 분석하여 각각의 카테고리를 판별하세요.\n\n"
    prompt += "카테고리 후보: [논문, 기술, 금융, 일반]\n"
    prompt += "응답 형식: 반드시 아래와 같은 JSON 리스트 형식으로만 답변하세요. 다른 설명은 생략하세요.\n"
    prompt += '[{"filename": "문서1.pdf", "category": "논문"}, {"filename": "문서2.pdf", "category": "기술"}]\n\n'
    
    for i, (filename, text) in enumerate(doc_batch):
        # 문서별로 앞 7,000자, 뒤 3,000자 샘플링 (토큰 절약 및 특징 추출)
        sample = text[:7000] + "\n... (중략) ...\n" + text[-3000:] if len(text) > 10000 else text
        prompt += f"--- [문서 {i+1}: {filename}] ---\n{sample}\n\n"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        
        results = parse_json_response(response.text)
        if isinstance(results, list):
            return results
        else:
            log_message(f"배치 분류 결과가 리스트 형식이 아닙니다: {response.text[:200]}", "ERROR")
            return None
    except Exception as e:
        log_message(f"배치 분류 중 API 오류: {e}", "ERROR")
        return None

def process_all_documents():
    """Input 폴더의 모든 새로운 문서를 배치 단위로 분류합니다."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history = load_history()
    
    # 1. 처리할 파일 목록 확보 (이미 처리된 해시는 스킵)
    files_to_process = []
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(INPUT_DIR, filename)
            file_hash = calculate_file_hash(file_path)
            
            if file_hash in history and history[file_hash].get("classified"):
                continue
                
            files_to_process.append((filename, file_path, file_hash))

    if not files_to_process:
        log_message("새로 분류할 문서가 없습니다.")
        return

    log_message(f"총 {len(files_to_process)}개의 새로운 문서를 분류합니다. (배치 크기: {BATCH_SIZE})")

    # 2. 배치 단위로 루프 실행
    for i in range(0, len(files_to_process), BATCH_SIZE):
        batch = files_to_process[i : i + BATCH_SIZE]
        doc_data_batch = []
        
        # 텍스트 미리 추출
        for filename, path, fhash in batch:
            text = extract_text_from_pdf(path)
            doc_data_batch.append((filename, text))
        
        log_message(f"배치 실행 중 ({i//BATCH_SIZE + 1}/{(len(files_to_process)-1)//BATCH_SIZE + 1})...")
        batch_results = classify_documents_batch(client, doc_data_batch)
        
        if not batch_results:
            log_message("배치 처리에 실패했습니다. 다음 배치로 넘어갑니다.", "WARNING")
            continue

        # 3. 결과 적용 (파일 이동 및 히스토리 업데이트)
        # LLM 응답 파일명과 실제 파일명 매칭
        result_map = {res.get("filename"): res.get("category") for res in batch_results if "filename" in res}
        
        for filename, path, fhash in batch:
            category = result_map.get(filename, "일반") # 결과 없으면 '일반'
            if category not in CATEGORIES: category = "일반"
            
            target_dir = os.path.join(CLASSIFIED_DIR, category)
            os.makedirs(target_dir, exist_ok=True)
            
            # 이동 (이미 있으면 덮어쓰기 대신 이름 변경 가능하나 여기선 단순 이동)
            shutil.move(path, os.path.join(target_dir, filename))
            
            # 히스토리 업데이트
            history[fhash] = {
                "filename": filename,
                "category": category,
                "classified": True,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            log_message(f"분류 완료: {filename} -> {category}")

        save_history(history)
        time.sleep(5) # API 속도 제한 준수

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    process_all_documents()
