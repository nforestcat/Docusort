import sys
import os
from src.classifier import process_all_documents
from src.summarizer import process_summaries
from src.utils import log_message

def main():
    print("="*50)
    print("🚀 DocuSort AI: 통합 문서 분류 및 요약 시스템")
    print("="*50)
    
    log_message("통합 프로세스 시작")

    try:
        # 1. 문서 분류 단계 (ZIP 해제 포함)
        print("\n[1단계: 문서 분류 및 전처리]")
        process_all_documents()
        
        # 2. 논문 요약 단계 (이름 변경 및 정리 포함)
        print("\n[2단계: 논문 요약 및 이름 변경]")
        process_summaries()
        
        print("\n" + "="*50)
        print("✅ 모든 작업이 완료되었습니다!")
        print("결과물 확인: output/summaries/ 및 output/classified/논문/processed/")
        print("="*50)
        log_message("통합 프로세스 완료")

    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 작업이 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 치명적 오류 발생: {str(e)}")
        log_message(f"통합 프로세스 오류: {str(e)}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
