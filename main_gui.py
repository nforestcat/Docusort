import tkinter as tk
from tkinter import messagebox, scrolledtext
import threading
import os
import subprocess
from src.classifier import process_all_documents
from src.summarizer import process_summaries

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

    def open_folder(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        subprocess.Popen(f'explorer "{os.path.abspath(path)}"')

    def run_process_async(self):
        self.start_btn.config(state=tk.DISABLED)
        threading.Thread(target=self.run_process, daemon=True).start()

    def run_process(self):
        self.log("🚀 프로세스를 시작합니다...")
        try:
            process_all_documents()
            process_summaries()
            self.log("✅ 모든 작업이 완료되었습니다!")
            messagebox.showinfo("완료", "문서 분류 및 요약이 완료되었습니다.")
        except Exception as e:
            self.log(f"❌ 오류: {str(e)}")
            messagebox.showerror("오류", f"작업 중 오류 발생: {e}")
        finally:
            self.start_btn.config(state=tk.NORMAL)

    def toggle_monitoring(self):
        if not self.monitoring:
            self.start_monitor_thread()
        else:
            self.stop_monitoring()

    def start_monitor_thread(self):
        from main_monitor import NewFileHandler
        from watchdog.observers import Observer

        self.monitoring = True
        self.monitor_btn.config(text="🛑 감시 중지", bg="#f44336")
        self.log("🔍 실시간 감시를 시작합니다 (input 폴더)...")

        # 모니터링 로직 (GUI용 커스텀 핸들러)
        class GUIFileHandler(NewFileHandler):
            def __init__(self, gui):
                super().__init__()
                self.gui = gui
            def run_process(self):
                self.gui.log(f"🔔 새 파일 감지! 프로세스 가동...")
                super().run_process()
                self.gui.log("✅ 대기 중...")

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

if __name__ == "__main__":
    root = tk.Tk()
    app = DocuSortGUI(root)
    root.mainloop()
