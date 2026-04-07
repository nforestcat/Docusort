import os
import re
import subprocess
import pymupdf4llm
import sys

# 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

def call_gemini_cli(prompt: str, input_text: str = None) -> str:
    custom_env = os.environ.copy()
    custom_env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    custom_env["PYTHONIOENCODING"] = "utf-8"
    try:
        cmd = ["gemini.cmd", "-p", prompt]
        result = subprocess.run(
            cmd, input=input_text, capture_output=True, text=True,
            env=custom_env, shell=True, errors='replace', encoding='utf-8'
        )
        return result.stdout.strip()
    except Exception as e: return f"ERROR: {str(e)}"

def clean_paper_text(text: str) -> str:
    ref_patterns = [r'\n\s*References\s*\n', r'\n\s*REFERENCES\s*\n', r'\n\s*참고문헌\s*\n']
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return text[:match.start()]
    return text

instruction = """당신은 논문 분석가입니다. 제시된 텍스트 전체를 분석하여 아래 양식으로 요약하세요.
[요약 양식] 한국어 마크다운. `# 요약`, `## 핵심 내용`, `## 결론` 형식을 지키세요.
마지막에 '분석 모델: [현재 사용 중인 모델명]' 형식을 반드시 포함하세요.
"""

# 파일 경로 찾기
base_dir = "output/classified"
paper_cat = next((d for d in os.listdir(base_dir) if "논문" in d), None)
target_dir = os.path.join(base_dir, paper_cat, "processed")
target_file = next((f for f in os.listdir(target_dir) if "Wei-Liang_Chen" in f and f.endswith(".pdf")), None)
file_path = os.path.abspath(os.path.join(target_dir, target_file))

print(f"📄 대상 파일: {file_path}")
full_text = pymupdf4llm.to_markdown(file_path)
cleaned_text = clean_paper_text(full_text)
total_len = len(cleaned_text)
print(f"📏 전체 텍스트 길이 (References 제외): {total_len:,}자")

print(f"\n--- [전체 텍스트 요약 시작] ---")
summary = call_gemini_cli(instruction, input_text=cleaned_text)

print("\n[요약 결과]")
print(summary)

# 결과 저장
with open("summary_full_text.md", "w", encoding="utf-8") as f:
    f.write(summary)
