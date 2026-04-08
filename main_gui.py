import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import os
import subprocess
import time
from src.classifier import process_all_documents
from src.summarizer import process_summaries
from src.utils import log_message

class DocuSortGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DocuSort AI - 문서 자동 분류 및 요약")
        self.root.geometry("600x450")
        
        self.monitoring = False
        self.observer = None

        # UI 구성
        self.create_widgets()

    def create_widgets(self):
        # 상단 제목
        title_label = tk.Label(self.root, text="📄 DocuSort AI Dashboard", font=("Helvetica", 16, "bold"))
        title_label.pack(pady=10)

        # 메인 버튼 프레임
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.start_btn = tk.Button(btn_frame, text="🚀 즉시 전체 실행", command=self.run_process_async, 
                                   width=20, height=2, bg="#4CAF50", fg="white", font=("Helvetica", 10, "bold"))
        self.start_btn.grid(row=0, column=0, padx=10)

        self.monitor_btn = tk.Button(btn_frame, text="🔍 실시간 감시 시작", command=self.toggle_monitoring, 
                                     width=20, height=2, bg="#2196F3", fg="white", font=("Helvetica", 10, "bold"))
        self.monitor_btn.grid(row=0, column=1, padx=10)

        # 폴더 열기 프레임
        folder_frame = tk.Frame(self.root)
        folder_frame.pack(pady=5)

        tk.Button(folder_frame, text="📂 Input 폴더", command=lambda: self.open_folder("input")).grid(row=0, column=0, padx=5)
        tk.Button(folder_frame, text="📂 분류 결과", command=lambda: self.open_folder("output/classified")).grid(row=0, column=1, padx=5)
        tk.Button(folder_frame, text="📂 요약본 폴더", command=lambda: self.open_folder("output/summaries")).grid(row=0, column=2, padx=5)

        # 로그 출력 영역
        tk.Label(self.root, text="실행 로그:").pack(anchor="w", padx=20)
        self.log_area = scrolledtext.ScrolledText(self.root, height=15, width=70, font=("Consolas", 9))
        self.log_area.pack(pady=5, padx=20)

    def log(self, message):
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.see(tk.END)
        # 콘솔 로그 및 파일 로그와도 연동
        if "❌" in message or "ERROR" in message:
            log_message(message.strip(), "ERROR")
        else:
            log_message(message.strip())

    def open_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        subprocess.Popen(f'explorer "{os.path.abspath(path)}"')

    def run_process_async(self):
        self.start_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        self.log("🚀 전체 프로세스를 시작합니다 (분류 -> 요약)...")
        try:
            # 1. 문서 분류 단계
            self.log("[1단계: 문서 분류 및 전처리]")
            process_all_documents()
            
            # 2. 논문 요약 단계
            self.log("\n[2단계: 논문 요약 및 이름 변경]")
            process_summaries()
            
            self.log("\n✅ 모든 작업이 완료되었습니다!")
            self.log("결과물 확인: output/summaries/ 및 output/classified/논문/processed/")
            messagebox.showinfo("완료", "모든 문서의 분류 및 요약이 완료되었습니다.")
        except Exception as e:
            self.log(f"❌ 오류 발생: {str(e)}")
            messagebox.showerror("오류", f"작업 중 치명적 오류가 발생했습니다:\n{e}")
        finally:
            self.start_btn.config(state=tk.NORMAL)

    def toggle_monitoring(self):
        if not self.monitoring:
            self.start_monitor_thread()
        else:
            self.stop_monitoring()

    def start_monitor_thread(self):
        from watchdog.observers import Observer
        from main_monitor import NewFileHandler

        self.monitoring = True
        self.monitor_btn.config(text="🛑 감시 중지", bg="#f44336")
        self.log("🔍 실시간 감시 모드 활성화 (input 폴더)...")

        # GUI용 커스텀 핸들러 (로그 연동)
        class GUIFileHandler(NewFileHandler):
            def __init__(self, gui):
                super().__init__()
                self.gui = gui
            def run_process(self):
                self.processing = True
                self.gui.log(f"\n🔔 새 파일 감지! 자동 프로세스를 시작합니다...")
                time.sleep(3) # 파일 쓰기 완료 대기
                try:
                    self.gui.log("[1단계: 문서 분류 및 전처리]")
                    process_all_documents()
                    self.gui.log("[2단계: 논문 요약 및 이름 변경]")
                    process_summaries()
                    self.gui.log("✅ 자동 프로세스 완료. 결과물을 확인하세요.")
                except Exception as e:
                    self.gui.log(f"❌ 자동 처리 중 오류: {e}")
                finally:
                    self.processing = False

        self.observer = Observer()
        handler = GUIFileHandler(self)
        self.observer.schedule(handler, "input", recursive=False)
        self.observer.start()

    def stop_monitoring(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
        self.monitoring = False
        self.monitor_btn.config(text="🔍 실시간 감시 시작", bg="#2196F3")
        self.log("🛑 실시간 감시가 중지되었습니다.")

    def on_closing(self):
        """창을 닫을 때 모든 감시를 중단하고 안전하게 종료합니다."""
        if self.monitoring:
            self.stop_monitoring()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DocuSortGUI(root)
    # 창 닫기(X 버튼) 이벤트와 핸들러 연결
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
