import subprocess
import os
import re

def call_gemini_cli(prompt: str, file_path: str = None) -> str:
    custom_env = os.environ.copy()
    custom_env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    custom_env["PYTHONIOENCODING"] = "utf-8"
    try:
        if file_path:
            cmd = ["gemini.cmd", "-f", file_path, "-p", prompt]
        else:
            cmd = ["gemini.cmd", "-p", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, env=custom_env, shell=True, errors='replace', encoding='utf-8')
        return result.stdout.strip()
    except Exception as e: return str(e)

file_path = os.path.abspath("input/[0000]_[Wei-Liang_Chen]_[Unknown].pdf")
print(f"Testing with path: {file_path}")
instruction = "당신은 고도로 정밀한 문서 분류기입니다. 제시된 파일을 보고 다음 중 하나로 분류하세요: [논문, 행정서식, 과제금융, 기술매뉴얼, 일반안내]. 결과는 반드시 다음 형식을 따르세요: RESULT: [카테고리]"

response = call_gemini_cli(instruction, file_path)
print(f"Response: {response}")
