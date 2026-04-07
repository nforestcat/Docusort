import os
import re
import pymupdf4llm
import sys

# 인코딩 설정
if sys.stdout.encoding != 'utf-8':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

def clean_paper_text(text: str) -> str:
    ref_patterns = [r'\n\s*References\s*\n', r'\n\s*REFERENCES\s*\n', r'\n\s*참고문헌\s*\n']
    for pattern in ref_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match: return text[:match.start()]
    return text

# 파일 경로 찾기
base_dir = "output/classified"
paper_cat = next((d for d in os.listdir(base_dir) if "논문" in d), None)
target_dir = os.path.join(base_dir, paper_cat, "processed")
target_file = next((f for f in os.listdir(target_dir) if "Wei-Liang_Chen" in f and f.endswith(".pdf")), None)
file_path = os.path.abspath(os.path.join(target_dir, target_file))

full_text = pymupdf4llm.to_markdown(file_path)
cleaned_text = clean_paper_text(full_text)

# 50,000자 지점 주변의 텍스트 확인 (49,900 ~ 50,100)
cutoff_point = 50000
context_range = 200
snippet = cleaned_text[cutoff_point - context_range : cutoff_point + context_range]

print(f"--- [50,000자 지점 주변 텍스트] ---")
print(snippet)
print(f"\n--- [끝] ---")
