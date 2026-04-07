import os
import re
import time
import json
import shutil
import sys
import pymupdf4llm
from google import genai
from dotenv import load_dotenv
from src.utils import log_message, calculate_file_hash, ensure_api_key, load_history, save_history

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

def split_and_filter_sections(text: str):
    """섹션을 분할하고 알짜배기 섹션만 골라내어 그룹화합니다."""
    # ## 헤더 기준으로 분할
    raw_sections = re.split(r'(\n##\s+)', text)
    
    # 헤더와 내용을 매핑
    sections_dict = {"Intro_Abstract": []}
    current_header = "Intro_Abstract"
    
    if len(raw_sections) > 1:
        # 첫 부분 (보통 Abstract)
        sections_dict["Intro_Abstract"].append(raw_sections[0].strip())
        
        for i in range(1, len(raw_sections), 2):
            header = raw_sections[i].strip().lower()
            content = raw_sections[i+1].strip() if i+1 < len(raw_sections) else ""
            
            # 버릴 섹션 필터링
            skip_keywords = ['reference', 'related work', 'acknowledgement', 'appendix', 'conflict of interest', 'supporting information', 'author contribution']
            if any(k in header for k in skip_keywords):
                continue
            
            # 핵심 그룹화 (Map 1: 배경 및 결론)
            core_keywords = ['abstract', 'introduction', 'conclusion', 'summary']
            # 연구 그룹화 (Map 2: 본문 및 결과)
            research_keywords = ['result', 'discussion', 'method', 'experiment', 'analysis', 'implementation']
            
            if any(k in header for k in core_keywords):
                sections_dict["Intro_Abstract"].append(f"## {header}\n{content}")
            elif any(k in header for k in research_keywords):
                if "Research_Body" not in sections_dict: sections_dict["Research_Body"] = []
                sections_dict["Research_Body"].append(f"## {header}\n{content}")
            else:
                # 기타 분류되지 않은 중요할 수 있는 본문
                if "Research_Body" not in sections_dict: sections_dict["Research_Body"] = []
                sections_dict["Research_Body"].append(f"## {header}\n{content}")
    else:
        sections_dict["Intro_Abstract"].append(text)

    return sections_dict

def summarize_group(group_name: str, content_list: list):
    """그룹화된 텍스트를 한 번의 API 호출로 요약합니다."""
    c = get_client()
    combined_content = "\n\n".join(content_list)
    
    if group_name == "Intro_Abstract":
        role = "논문의 연구 배경, 목적, 그리고 최종 결론을 분석하는 전략 분석가"
        focus = "이 연구가 왜 시작되었으며, 어떤 문제를 해결했고, 최종적으로 어떤 성과를 거두었는지에 집중하세요."
    else:
        role = "논문의 실험 방법론과 결과 데이터를 정밀 분석하는 기술 심사역"
        focus = "구체적인 연구 방법, 실험 결과, 데이터의 의미 및 논의 사항에 집중하여 상세히 요약하세요."

    instruction = f"""당신은 {role}입니다.
제시된 텍스트는 논문의 핵심 섹션들입니다. 내용을 상세하고 전문적으로 요약하세요.

[중요 지침]
- 반드시 한국어(UTF-8)를 사용하세요.
- 마크다운 헤더(##)를 기준으로 내용을 파악하되, 원본에 있는 주요 섹션의 정보가 누락되지 않도록 하세요.
- {focus}
"""
    try:
        response = c.models.generate_content(
            model=MODELS[current_model_idx],
            contents=[instruction, combined_content]
        )
        return response.text if response.text else f"{group_name} 요약 실패"
    except Exception as e:
        return f"Error in Map phase ({group_name}): {str(e)}"

def reduce_final_review(mapped_summaries: list):
    """(Reduce Phase) 개별 그룹 요약본을 합쳐 최종 리뷰와 서지 정보를 생성합니다."""
    c = get_client()
    combined_text = "\n\n---\n\n".join(mapped_summaries)
    
    instruction = """당신은 논문의 전체 구조와 세부 연구 결과를 통합하여 고품질 학술 리뷰를 작성하는 수석 편집자입니다.
제시된 요약본들을 하나로 합쳐 깊이 있는 최종 리뷰 문서를 작성하고 서지 정보를 추출하세요.

[중요 지침]
- 반드시 한국어(UTF-8)를 사용하세요.
- 전체적인 연구의 흐름(배경 -> 방법 -> 결과 -> 결론)이 매끄럽게 연결되도록 하세요.
- 결과물은 반드시 아래의 JSON 블록으로 시작하고, 그 뒤에 마크다운 리뷰를 작성하세요.

[JSON 형식]
```json
{
  "year": "출판 연도 (예: 2024)",
  "author": "대표 저자의 성(Surname) (예: Smith)",
  "keywords": "핵심 키워드 2-3개를 언더바(_)로 연결"
}
```

[리뷰 양식] 마크다운 형식. `# 최종 리뷰`, `## 연구 배경 및 목적`, `## 주요 연구 방법 및 결과`, `## 학술적 가치 및 결론` 형식을 갖추세요.
"""
    try:
        response = c.models.generate_content(
            model=MODELS[current_model_idx],
            contents=[instruction, combined_text]
        )
        return response.text if response.text else "최종 리듀스 실패"
    except Exception as e:
        return f"Error in Reduce phase: {str(e)}"

def parse_response(text: str):
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        try:
            info = json.loads(json_match.group(1))
            summary = text.replace(json_match.group(0), "").strip()
            return info, summary
        except: pass
    return None, text

def process_summaries():
    paper_dir = "output/classified/논문"
    summary_output_dir = "output/summaries"
    processed_dir = os.path.join(paper_dir, "processed")
    
    get_client()
    history = load_history()
    os.makedirs(summary_output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    files = [f for f in os.listdir(paper_dir) if f.lower().endswith('.pdf')]

    for filename in files:
        file_path = os.path.join(paper_dir, filename)
        file_hash = calculate_file_hash(file_path)

        if file_hash in history:
            entry = history[file_hash]
            if isinstance(entry, dict) and entry.get("filename"):
                target_name = entry.get("filename")
                print(f"\n⚠️ 이미 요약 완료된 내용 감지: {filename}")
                shutil.move(file_path, os.path.join(processed_dir, target_name))
                continue

        print(f"\n🚀 선택적 맵핑(Selective Mapping) 분석 시작 (Gemma 4): {filename}")
        
        try:
            full_text = pymupdf4llm.to_markdown(file_path)
            
            # 1. Selective Split & Filter
            section_groups = split_and_filter_sections(full_text)
            print(f"  └ 📄 섹션 필터링 및 그룹화 완료 ({len(section_groups)}개 그룹)")
            
            # 2. Map Phase (고정 2회 호출 예상)
            mapped_summaries = []
            for group_name, content_list in section_groups.items():
                if not content_list: continue
                print(f"    - [{group_name}] 그룹 분석 중...")
                summary_part = summarize_group(group_name, content_list)
                mapped_summaries.append(summary_part)
                time.sleep(5)
            
            # 3. Reduce Phase
            print("  └ 🔄 최종 통합 리뷰 생성 중...")
            final_raw = reduce_final_review(mapped_summaries)
            
            info, final_summary = parse_response(final_raw)
            if info:
                year = str(info.get('year', '0000')).strip()
                author = str(info.get('author', 'Unknown')).strip()
                keywords = str(info.get('keywords', 'Paper')).strip()
                
                new_base = f"{year}_{author}_{keywords}"
                new_base = re.sub(r'[\\/*?:"<>|\[\]]', "", new_base).replace(" ", "_")
                
                new_pdf_name = f"{new_base}.pdf"
                new_summary_name = f"{new_base}_summary.md"
                
                with open(os.path.join(summary_output_dir, new_summary_name), "w", encoding="utf-8-sig") as f:
                    f.write(final_summary)
                
                target_path = os.path.join(processed_dir, new_pdf_name)
                cnt = 1
                while os.path.exists(target_path):
                    target_path = os.path.join(processed_dir, f"{new_base}_{cnt}.pdf")
                    cnt += 1
                
                shutil.move(file_path, target_path)
                final_name = os.path.basename(target_path)
                print(f"✅ 최종 처리 완료: {final_name}")
                
                history[file_hash] = {
                    "category": "논문",
                    "filename": final_name,
                    "model": "gemma-4-31b-it-selective-mapreduce",
                    "processed_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                save_history(history)
            else:
                print("⚠️ 최종 리듀스 응답 파싱 실패")

        except Exception as e:
            print(f"❌ 분석 중 오류: {e}")
            continue

        time.sleep(5)

if __name__ == "__main__":
    process_summaries()
