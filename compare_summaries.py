import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

# 모델 리스트
MODELS = ["gemini-3.1-flash-lite-preview", "gemma-4-31b-it"]

# 논문 텍스트 (사용자가 제공한 앞부분 일부)
PAPER_TEXT = """
Current Research in Biotechnology 6 (2023) 100166 

A comparative assessment of a piezoelectric biosensor based on a new antifouling nanolayer and cultivation methods: Enhancing S. aureus detection in fresh dairy products 

Michala Forinov´a, Anna Seidlov´a, Alina Pilipenco, Nicholas Scott Lynn Jr., Radka Oboˇrilova´, Zdenˇek Farka, Petr Skladal´, Alena Salakov´ a´, Monika Spasovov´a, Milan Houska, Libor Kalhotka, Hana Vaisocherova-Lísalov´ a´

A B S T R A C T
Ensuring dairy product safety demands rapid and precise Staphylococcus aureus (S. aureus) detection. Biosensors show promise, but their performance is often demonstrated in model samples using non-native pathogens and has never been studied towards S. aureus detection in naturally contaminated samples. This study addresses the gap by directly comparing results taken with a novel piezoelectric biosensor, capable of one-step detection, with four conventional cultivation-based methods. Our findings reveal that this biosensor, based on an antifouling nanolayer-coated biochip, exhibits exceptional resistance to biofouling from unprocessed dairy products and is further capable of specific S. aureus detection. Notably, it performed comparably to Petrifilm and Baird-Parker methods but delivered results in only 30 min, bringing a substantial reduction from the 24 h required by cultivation-based techniques. Our study also highlights differences in the performance of cultivation methods when analyzing artificially spiked versus naturally contaminated foods. These findings underline the potential of antifouling biosensors as efficient reliable tools for rapid, cost-effective, point-of-care testing, enhancing fresh dairy product safety and S. aureus detection.

Introduction 
Staphylococcus aureus is an alimentary pathogen ranked as the third most common bacterial cause of foodborne illnesses worldwide. Transmission occurs mainly in food processing facilities, particularly during food handling and packaging (Ballah et al., 2022). This pathogen can produce various extracellular toxins, including toxic shock syndrome toxin 1 (TSST-1), exfoliative toxins, hemolysins, leukocidins, and staphylococcal enterotoxins (SE) (Freitas et al., 2023). These toxins also play a role in both skin infections and mastitis in cows (Fagundes et al., 2010; Abril et al., 2020). If they persist in dairy products, they can cause staphylococcal food poisoning outbreaks. Hence, rapid identification of the infection source and the implementation of suitable veterinary treatments for infected ruminants are crucial to mitigate economic losses in the dairy industry (Abril et al., 2020). 

Cultivation-based techniques remain the gold standard for the identification of S. aureus and other bacterial pathogens responsible for contagious mastitis (Neculai-Valeanu & Ariton, 2022). In these methods, homogenized dairy samples are incubated on culture plates to determine the number of colony-forming units (CFUs) in a sample, with specific agars on each plate providing means to identify the pathogen strain. Although immensely useful, these tests require long incubation periods (24 – 48 h), sterile conditions for sample transfer, and are prone to false negative results. Alternatively, methods based on polymerase chain reaction (PCR) and protein identification are on the rise (Ashraf & Imran, 2018), but currently remain expensive and require pristine laboratory conditions. 

Point-of-care (POC) biosensors represent a suitable solution for pathogen detection, as they offer rapid analysis, high sensitivity, selectivity, and can be used in non-sterile conditions and operated with minimal training. Specifically piezoelectric quartz crystal microbalance (QCM) biosensors are suitable for POC testing; however, the bulk of previous work has been focused on samples that don’t represent real conditions (e.g., cultured bacteria spiked into the buffer) and furthermore focus on the analysis of real samples that were spiked with single cultured strains rather than naturally contaminated samples, which may contain multiple strains. To the best of our knowledge, there are no previous studies on the use of biosensing methods towards S. aureus detection occurring in real-world naturally contaminated samples.
"""

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
    
    results = {}
    for model in MODELS:
        print(f"Generating summary with {model}...")
        results[model] = get_summary(client, model, PAPER_TEXT)
    
    with open("comparison_result.md", "w", encoding="utf-8-sig") as f:
        f.write("# 모델별 논문 요약 비교\n\n")
        for model, summary in results.items():
            f.write(f"## 모델: {model}\n\n")
            f.write(summary)
            f.write("\n\n---\n\n")
    
    print("Comparison complete. Result saved to comparison_result.md")

if __name__ == "__main__":
    main()
