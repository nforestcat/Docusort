import os
import re
import time
import json
import shutil
import subprocess
import sys
from src.utils import log_message, extract_text_from_pdf, calculate_file_hash
import fitz

def call_gemini_cli(prompt: str, input_text: str = None) -> str:
    """Gemini CLI를 호출합니다."""
    custom_env = os.environ.copy()
    custom_env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    custom_env["PYTHONIOENCODING"] = "utf-8"
    try:
        cmd = ["gemini.cmd", "-p", prompt]
        result = subprocess.run(
            cmd, input=input_text, capture_output=True, text=True,
            env=custom_env, shell=True, errors='replace', encoding='utf-8'
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"ERROR: {result.stderr}"
    except Exception as e: return f"ERROR: {str(e)}"

HISTORY_FILE = "output/processed_history.json"
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f: return json.load(f)
        except: return {}
    return {}

def save_history(history):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

def clean_paper_text(text: str) -> str:
    """참고문헌(References) 섹션 이후의 텍스트는 분석에서 제외합니다."""
    ref_patterns = [r'\n\s*References\s*\n', r'\n\s*REFERENCES\s*\n', r'\n\s*참고문헌\s*\n']
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return text[:match.start()]
    return text

def summarize_and_rename_info(text: str):
    instruction = """당신은 논문 분석가이자 서지 정보 추출 전문가입니다. 
서론이나 부연 설명 없이 즉시 아래 JSON 블록으로 시작하세요. 그 뒤에 요약본을 작성하세요.

[JSON 형식]
{
  "year": "출판 연도 (예: 2024)",
  "author": "대표 저자의 성(Surname) (예: Smith)",
  "keywords": "핵심 키워드 2-3개를 언더바(_)로 연결 (예: AI_Ethics_Policy)"
}

[요약 양식] 한국어(영어 병기) 마크다운. `# 요약`, `## 핵심 내용`, `## 결론` 형식을 반드시 지키세요.
"""
    return call_gemini_cli(instruction, input_text=text[:10000])

def parse_response(text: str):
    # 0. <thought> 블록 제거
    text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 1. JSON 블록 추출 시도
    json_str = None
    json_block_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_block_match:
        json_str = json_block_match.group(1)
    else:
        json_match = re.search(r'(\{.*?\})', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)

    refined_info = {"year": "0000", "author": "Unknown", "keywords": "Paper"}
    summary = text

    if json_str:
        try:
            # 우선 json.loads로 시도
            data = json.loads(json_str)
            
            # 연도 추출
            year = data.get("year")
            if not year and "publication_date" in data:
                year_match = re.search(r'(\d{4})', str(data["publication_date"]))
                if year_match: year = year_match.group(1)
            refined_info["year"] = str(year) if year else "0000"
            
            # 저자 추출
            author = data.get("author") or data.get("authors")
            if isinstance(author, list) and author:
                author = author[0]
            # 저자 이름에서 성만 추출 시도 (공백 기준 마지막 단어)
            if author and isinstance(author, str) and " " in author.strip():
                author = author.strip().split(" ")[-1]
            refined_info["author"] = str(author) if author else "Unknown"
            
            # 키워드 추출
            keywords = data.get("keywords") or data.get("tags") or data.get("topic")
            if isinstance(keywords, list):
                keywords = "_".join(map(str, keywords[:3]))
            refined_info["keywords"] = str(keywords) if keywords else "Paper"
            
            # 요약 본문은 JSON 이후 부분
            summary = text.split(json_str)[-1].strip()
            summary = re.sub(r'^```json|^```|^\s*\}', '', summary).strip()
        except:
            # JSON 파싱 실패 시 기존 regex 기반 추출 (최후의 수단)
            year_match = re.search(r'"year":\s*"?(\d{4})"?', json_str)
            if not year_match: year_match = re.search(r'"publication_date":\s*"?(\d{4})"?', json_str)
            
            author_match = re.search(r'"authors?":\s*(?:\[\s*"?(.*?)"?[,\]]|"?(.*?)"?[,}\n])', json_str)
            
            keywords_match = re.search(r'"(?:keywords|tags|topic)":\s*(?:\[\s*"?(.*?)"?[,\]]|"?(.*?)"?[,}\n])', json_str)

            refined_info["year"] = year_match.group(1) if year_match else "0000"
            
            auth = (author_match.group(1) or author_match.group(2) or "Unknown") if author_match else "Unknown"
            if " " in auth.strip(): auth = auth.strip().split(" ")[-1]
            refined_info["author"] = auth
            
            refined_info["keywords"] = (keywords_match.group(1) or keywords_match.group(2) or "Paper") if keywords_match else "Paper"
            
            summary = text.split(json_str)[-1].strip()

    # 요약 양식 강제 추출
    summary_match = re.search(r'(#\s*요약.*)', summary, re.DOTALL | re.IGNORECASE)
    if summary_match:
        summary = summary_match.group(1)

    return refined_info, summary

def sanitize_filename(name: str) -> str:
    name = str(name).replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    name = re.sub(r'[^a-zA-Z0-9_\-\[\]\s]', '', name)
    name = name.strip().replace(" ", "_")
    name = re.sub(r'_+', '_', name)
    return name if name else "Unknown"

def process_summaries():
    paper_dir = "output/classified/논문"
    summary_output_dir = "output/summaries"
    processed_dir = os.path.join(paper_dir, "processed")
    history = load_history()
    
    os.makedirs(summary_output_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    files = [f for f in os.listdir(paper_dir) if f.lower().endswith('.pdf')]

    for filename in files:
        file_path = os.path.join(paper_dir, filename)
        file_hash = calculate_file_hash(file_path)

        if file_hash in history:
            shutil.move(file_path, os.path.join(processed_dir, history[file_hash]))
            continue

        print(f"\n🚀 분석 및 요약 중: {filename}...")
        try:
            doc = fitz.open(file_path); full_text = "".join([p.get_text() for p in doc]); doc.close()
            cleaned_text = clean_paper_text(full_text)
        except: continue

        raw = summarize_and_rename_info(cleaned_text)
        print(f"DEBUG: Raw response: {raw}")
        info, summary = parse_response(raw)
        
        if info:
            year = sanitize_filename(info.get('year', '0000'))
            author = sanitize_filename(info.get('author', 'Unknown'))
            keywords = sanitize_filename(info.get('keywords', 'Paper'))
            
            new_base = f"[{year}]_[{author}]_[{keywords}]"
            new_pdf_name = f"{new_base}.pdf"
            new_summary_name = f"{new_base}_summary.md"
            
            with open(os.path.join(summary_output_dir, new_summary_name), "w", encoding="utf-8") as f:
                f.write(summary)
            
            target_path = os.path.join(processed_dir, new_pdf_name)
            cnt = 1
            while os.path.exists(target_path):
                target_path = os.path.join(processed_dir, f"{new_base}_{cnt}.pdf")
                cnt += 1
            
            shutil.move(file_path, target_path)
            final_name = os.path.basename(target_path)
            print(f"  └ ✅ 요약 완료: {final_name}")
            history[file_hash] = final_name
            save_history(history)
            time.sleep(10) # 쿼터 방지
        else:
            print(f"  └ ❌ 추출 실패")
            time.sleep(5)

if __name__ == "__main__":
    process_summaries()
