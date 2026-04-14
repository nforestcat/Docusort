import os
import shutil
import time
from datetime import datetime
from google import genai
from src.utils import log_message, calculate_file_hash, load_history, save_history, parse_json_response, extract_text_from_pdf

# 설정
INPUT_DIR = "input"
CLASSIFIED_DIR = "output/classified"
CATEGORIES = ["논문", "행정서식", "과제금융", "기술매뉴얼", "일반안내"]
BATCH_SIZE = 5 # 한 번에 분류할 문서 개수

# AI 모델 설정 (gemma_version 브랜치 전용)
MODEL_NAME = "gemma-4-31b-it"

def classify_documents_batch(client, doc_batch):
    """여러 문서를 한 번에 분류합니다. (Gemma 4 System Prompt & ID 매칭 적용)"""
    
    # 1. 시스템 프롬프트: 역할, 규칙, 출력 형식만 엄격하게 지정
    system_instruction = """당신은 대학원 연구실 환경의 문서 분류 전문가입니다.
제공된 여러 문서의 내용을 분석하여 각각의 카테고리를 판별하세요.

카테고리 후보 및 판별 기준:
1. 논문 (Paper): 학술지 논문, 컨퍼런스 발표 자료, 학위 논문 등 학술적 연구 성과물. (분류 후 요약 프로세스로 전달됨)
2. 행정서식 (Admin Forms): 입학/졸업 관련 신청서, 휴학/복학 신청서, 각종 보고서 양식, 서약서 등 행정 처리에 필요한 빈 서식 또는 작성된 서류.
3. 과제금융 (Grant/Finance): 연구비 집행 내역서, 견적서, 지출 증빙, 과제 제안서(RFP), 과제 협약서 등 연구 과제 및 예산 관련 문서.
4. 기술매뉴얼 (Tech Manual): 실험 장비 매뉴얼, 소프트웨어 사용법, MSDS(물질안전보건자료), 기술 사양서 등 도구 사용 및 안전 관련 문서.
5. 일반안내 (General Info): 학과 공지사항, 세미나 포스터, 강의 계획서(Syllabus), 학사 일정안내 등 일반적인 정보 전달 문서.

응답 형식: 반드시 아래와 같은 JSON 리스트 형식으로만 답변하세요. 다른 설명은 절대 생략하세요.
[
  {"id": 0, "category": "논문"},
  {"id": 1, "category": "행정서식"}
]"""

    # 2. 사용자 프롬프트: 순수한 데이터만 주입 (ID 기반)
    user_prompt = "아래 제공된 문서들을 분석하고 JSON으로 결과를 반환해 줘.\n\n"
    
    for i, (filename, text) in enumerate(doc_batch):
        # 문서별로 앞 7,000자, 뒤 3,000자 샘플링 (토큰 절약 및 특징 추출)
        sample = text[:7000] + "\n... (중략) ...\n" + text[-3000:] if len(text) > 10000 else text
        # 파일명 대신 명확한 고유 인덱스(id) 사용
        user_prompt += f"--- [문서 ID: {i}] ---\n파일이름: {filename}\n{sample}\n\n"

    try:
        # 3. API 호출 (시스템 프롬프트 적용 및 저온도 설정)
        from google.genai import types
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )
        
        results = parse_json_response(response.text)
        
        # 4. ID를 다시 파일명으로 안전하게 매핑
        if isinstance(results, list):
            mapped_results = []
            for res in results:
                doc_id = res.get("id")
                category = res.get("category")
                
                # doc_id가 정상적인 인덱스인지 확인 후 원래 파일명과 결합
                if isinstance(doc_id, int) and 0 <= doc_id < len(doc_batch):
                    original_filename = doc_batch[doc_id][0]
                    mapped_results.append({"filename": original_filename, "category": category})
            
            return mapped_results
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
    
    # 1. 처리할 파일 목록 확보 (이미 처리된 해시는 삭제 및 스킵)
    files_to_process = []
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(INPUT_DIR, filename)
            file_hash = calculate_file_hash(file_path)
            
            # 중복 파일 처리 (이미 히스토리에 있는 경우)
            if file_hash in history and history[file_hash].get("classified"):
                try:
                    os.remove(file_path)
                    log_message(f"🗑️ 중복 파일 감지 및 삭제: {filename} (이미 처리됨)")
                except Exception as e:
                    log_message(f"중복 파일 삭제 실패 ({filename}): {e}", "ERROR")
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
            category = result_map.get(filename, "일반안내") # 결과 없으면 '일반안내'
            if category not in CATEGORIES: category = "일반안내"
            
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
