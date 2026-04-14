import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.classifier import process_all_documents
from src.summarizer import process_summaries
from src.utils import log_message

class NewFileHandler(FileSystemEventHandler):
    """파일 생성 이벤트 감지 시 작업을 수행합니다."""
    def __init__(self):
        self.processing = False # 중복 실행 방지

    def on_created(self, event):
        if not event.is_directory:
            ext = os.path.splitext(event.src_path)[1].lower()
            if ext in ['.pdf', '.zip']:
                if not self.processing:
                    print(f"\n🔔 새 파일 감지: {os.path.basename(event.src_path)}")
                    self.run_process()

    def run_process(self):
        self.processing = True
        # 파일이 완전히 복사/생성될 때까지 충분히 대기 (대용량 PDF 고려)
        time.sleep(3)
        print("\n🚀 자동 프로세스를 시작합니다 (분류 -> 요약)...")
        log_message("자동 감시 프로세스 시작")
        try:
            # 1. 문서 분류 단계
            print("[1단계: 문서 분류 및 전처리]")
            process_all_documents()

            # 2. 논문 요약 단계
            print("\n[2단계: 논문 요약 및 이름 변경]")
            process_summaries()

            print("\n✅ 자동 프로세스 완료. 감시를 계속합니다.")
            print("결과물 확인: output/summaries/ 및 output/classified/논문/processed/")
            log_message("자동 감시 프로세스 완료")
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            log_message(f"자동 감시 프로세스 오류: {str(e)}", "ERROR")
        finally:
            self.processing = False

def start_monitoring():
    input_dir = "input"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)

    # [중요] 감시 시작 전 현재 input 폴더에 있는 파일들을 즉시 한 번 처리합니다.
    print("="*50)
    print("🚀 DocuSort AI: 초기 작업 및 실시간 모니터링 준비 중")
    print("="*50)

    # 임시 핸들러를 생성하여 현재 파일들을 처리
    initial_handler = NewFileHandler()
    initial_handler.run_process()

    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, input_dir, recursive=False)

    print("\n" + "="*50)
    print("🔍 실시간 모니터링 모드 가동 중")
    print(f"📍 감시 폴더: {os.path.abspath(input_dir)}")
    print("💡 이 창을 띄워두면 input 폴더에 파일을 넣는 즉시 처리됩니다.")
    print("   (종료하려면 Ctrl+C를 누르세요.)")
    print("="*50)

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    start_monitoring()
