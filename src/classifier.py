import os
import shutil
import time
import re
from datetime import datetime
from google import genai
from google.genai import types
from src.utils import (
    log_message, 
    calculate_file_hash, 
    load_history, 
    save_history, 
    parse_json_response, 
    extract_text_from_pdf,
    ensure_api_key
)

# 설정
INPUT_DIR = "input"
CLASSIFIED_DIR = "output/classified"
CATEGORIES = ["논문", "행정서식", "과제금융", "기술매뉴얼", "일반안내"]
BATCH_SIZE = 5 # 한 번에 분류할 문서 개수

# AI 모델 설정 (main 브랜치: Gemini 3.1 Flash Lite 적용)
MODEL_NAME = "models/gemini-3.1-flash-lite-preview"

# 전역 클라이언트 변수 (지연 초기화)
client = None

def get_client():
    """API 키를 확인하고 클라이언트를 반환합니다."""
    global client
    if client is None:
        api_key = ensure_api_key()
        client = genai.Client(api_key=api_key)
    return client

def classify_documents_batch(c, doc_batch):
    """여러 문서를 한 번에 분류합니다. (Batch processing with System Instruction)"""

    # 1. 시스템 프롬프트: 역할, 규칙, 출력 형식 지정
    system_instruction = """당신은 대학원 연구실 및 행정 환경의 문서 분류 전문가입니다.
제공된 여러 문서의 내용을 분석하여 각각의 카테고리를 판별하세요.

카테고리 후보 및 판별 기준:
1. 논문 (Paper): 학술지 논문, 컨퍼런스 발표 자료, 학위 논문 등 학술적 연구 성과물. 
2. 행정서식 (Admin Forms): 각종 신청서, 결과 보고서 양식, 동의서, 서약서 등 행정 처리에 필요한 서식.
3. 과제금융 (Grant/Finance): 연구 과제 예산, 연구비 집행 내역서, 견적서, 지출 증빙, 과제 제안서(RFP) 등.
4. 기술매뉴얼 (Tech Manual): 실험 장비 매뉴얼, 소프트웨어 사용 가이드, MSDS, 기술 사양서.
5. 일반안내 (General Info): 학과 공지사항, 세미나 포스터, 강의 계획서(Syllabus), 학사 일정 등.

응답 형식: 반드시 아래와 같은 JSON 리스트 형식으로만 답변하세요. 다른 설명은 절대 생략하세요.
[
  {"id": 0, "category": "논문"},
  {"id": 1, "category": "행정서식"}
]"""

    # 2. 사용자 프롬프트: 데이터 주입 (ID 기반)
    user_prompt = "아래 제공된 문서들을 분석하고 JSON으로 결과를 반환해 줘.\n\n"

    for i, (filename, text) in enumerate(doc_batch):
        # 문서별 샘플링 (앞 7,000자, 뒤 3,000자)
        sample = text[:7000] + "\n... (중략) ...\n" + text[-3000:] if len(text) > 10000 else text
        user_prompt += f"--- [문서 ID: {i}] ---\n파일이름: {filename}\n{sample}\n\n"

    try:
        # 3. API 호출
        response = c.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1
            )
        )

        results = parse_json_response(response.text)

        # 4. ID를 다시 파일명으로 매핑
        if isinstance(results, list):
            mapped_results = []
            for res in results:
                doc_id = res.get("id")
                category = res.get("category")

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
    c = get_client()
    history = load_history()

    # 1. 처리할 파일 목록 확보
    files_to_process = []
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        return

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(INPUT_DIR, filename)
            file_hash = calculate_file_hash(file_path)

            # 중복 체크 (히스토리 참조)
            if file_hash in history and history[file_hash].get("classified"):
                # 이미 분류된 경우 원본 폴더에서 삭제하거나 건너뜀 (여기서는 안전하게 건너뛰기만 할 수도 있음)
                # shutil.move로 이미 이동되었어야 함.
                continue

            files_to_process.append((filename, file_path, file_hash))

    if not files_to_process:
        print("새로 분류할 문서가 없습니다.")
        return

    log_message(f"총 {len(files_to_process)}개의 문서를 배치 분류합니다. (배치 크기: {BATCH_SIZE})")

    # 2. 배치 단위 실행
    for i in range(0, len(files_to_process), BATCH_SIZE):
        batch = files_to_process[i : i + BATCH_SIZE]
        doc_data_batch = []

        # 텍스트 추출
        for filename, path, fhash in batch:
            text = extract_text_from_pdf(path)
            doc_data_batch.append((filename, text))

        print(f"📦 배치 처리 중 ({(i//BATCH_SIZE)+1}/{(len(files_to_process)-1)//BATCH_SIZE+1})...")
        batch_results = classify_documents_batch(c, doc_data_batch)

        if not batch_results:
            print("❌ 배치 처리에 실패했습니다. 다음으로 넘어갑니다.")
            continue

        # 3. 결과 적용
        result_map = {res.get("filename"): res.get("category") for res in batch_results if "filename" in res}

        for filename, path, fhash in batch:
            category = result_map.get(filename, "일반안내")
            if category not in CATEGORIES: category = "일반안내"

            target_dir = os.path.join(CLASSIFIED_DIR, category)
            os.makedirs(target_dir, exist_ok=True)

            try:
                shutil.move(path, os.path.join(target_dir, filename))
                
                # 히스토리 업데이트
                history[fhash] = {
                    "filename": filename,
                    "category": category,
                    "classified": True,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                print(f"  └ ✅ {filename} -> {category}")
            except Exception as e:
                log_message(f"파일 이동 실패 ({filename}): {e}", "ERROR")

        save_history(history)
        # RPM 제한 준수
        time.sleep(5)

if __name__ == "__main__":
    process_all_documents()
