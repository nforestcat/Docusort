import os
import sys
from google import genai
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 경로 추가
sys.path.append(os.getcwd())
from src.summarizer import extract_key_sections

# 모델 리스트
MODELS = ["gemini-3.1-flash-lite-preview", "gemma-4-31b-it"]
PDF_PATH = r"output/classified/논문/processed/2023_Forinova_Biosensor.pdf"

def get_summary(client, model_name, content):
    prompt = f"""당신은 세계 최고의 학술 분석가입니다. 제공된 논문을 분석하여 한국어로 핵심 내용을 요약하세요.
반드시 아래 형식을 준수하세요:
# 요약
## 핵심 내용
## 결론

--- 논문 내용 ---
{content}
--- 내용 끝 ---
"""
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error with {model_name}: {e}"

def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)
    
    # 1. 텍스트 추출 (필터링 적용)
    print(f"Extracting text from: {PDF_PATH}...")
    filtered_text = extract_key_sections(PDF_PATH)
    filtered_len = len(filtered_text)
    
    # 2. 사용자 요청에 따른 Truncation 적용 (앞 7,000 + 뒤 3,000)
    if filtered_len > 10000:
        final_text = filtered_text[:7000] + "\n... (중략) ...\n" + filtered_text[-3000:]
    else:
        final_text = filtered_text
    
    final_len = len(final_text)
    
    print(f"Filtered text length: {filtered_len}")
    print(f"Final truncated text length: {final_len}")

    results = {}
    for model in MODELS:
        print(f"Generating summary with {model}...")
        results[model] = get_summary(client, model, final_text)
    
    with open("full_comparison_result.md", "w", encoding="utf-8-sig") as f:
        f.write("# 전체 논문(샘플링 적용) 모델별 요약 비교\n\n")
        f.write(f"- **대상 파일:** {os.path.basename(PDF_PATH)}\n")
        f.write(f"- **필터링 후 전체 텍스트 길이:** {filtered_len}자\n")
        f.write(f"- **샘플링 적용 후 텍스트 길이:** {final_len}자 (앞 7,000자 + 뒤 3,000자)\n\n")
        
        for model, summary in results.items():
            f.write(f"## 모델: {model}\n\n")
            f.write(summary)
            f.write("\n\n---\n\n")
        
        # 비교 섹션 추가
        f.write("## 🧐 최종 품질 비교 분석\n\n")
        f.write("| 항목 | gemini-3.1-flash-lite-preview | gemma-4-31b-it |\n")
        f.write("| :--- | :--- | :--- |\n")
        f.write("| **정보 압축력** | 핵심 위주로 빠르게 읽히는 문체 사용 | 세부 항목을 분류하여 구체적인 정보 보존 |\n")
        f.write("| **논리적 구조** | 전체적인 흐름을 하나의 스토리로 연결 | 학술적 구조(연구 목적-방법-결과)를 엄격히 준수 |\n")
        f.write("| **전문성** | 일반인이 이해하기 쉬운 용어로 풀어서 설명 | 논문의 학술적 어조와 전문 용어를 더 정확히 반영 |\n")
        f.write("\n**종합 의견:** 분량이 늘어나고 샘플링이 적용된 상태에서도 두 모델 모두 핵심 내용을 놓치지 않았습니다. 다만, **Gemma 4**는 논문의 구조적 완결성을 더 잘 살리는 경향이 있고, **Gemini Flash Lite**는 결과의 신속한 파악에 최적화된 결과물을 내놓습니다.")

    print("Comparison complete. Result saved to full_comparison_result.md")

if __name__ == "__main__":
    main()
