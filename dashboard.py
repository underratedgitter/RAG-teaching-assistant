import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import threading
import subprocess
import sys
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests


class RAGDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Teaching Assistant")
        self.root.geometry("700x650")
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.videos_dir = os.path.join(self.base_dir, "videos")
        self.audios_dir = os.path.join(self.base_dir, "audios")
        self.jsons_dir = os.path.join(self.base_dir, "jsons")
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.audios_dir, exist_ok=True)
        os.makedirs(self.jsons_dir, exist_ok=True)
        self._clear_work_dirs_on_start()
        
        self.df = None
        self.embedding_matrix = None
        self.load_embeddings()
        
        self.create_ui()

    def _clear_work_dirs_on_start(self):
        """Remove all videos, audios, jsons, and embedding cache on launch."""
        dirs = [self.videos_dir, self.audios_dir, self.jsons_dir]
        for path in dirs:
            try:
                for name in os.listdir(path):
                    file_path = os.path.join(path, name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            except Exception:
                pass

        for cache_name in ("embeddings.joblib", "embedding_matrix.npy"):
            try:
                cache_path = os.path.join(self.base_dir, cache_name)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            except Exception:
                pass
        
    def create_ui(self):
        # Video section
        tk.Label(self.root, text="1. Video Management").pack(anchor="w", padx=10, pady=(10,5))
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill="x", padx=10)
        
        tk.Button(btn_frame, text="Upload Videos", command=self.upload_videos).pack(side="left", padx=(0,5))
        self.process_btn = tk.Button(btn_frame, text="Process Videos", command=self.process_videos)
        self.process_btn.pack(side="left", padx=(0,5))
        tk.Button(btn_frame, text="Open Folder", command=lambda: os.startfile(self.videos_dir)).pack(side="left", padx=(0,5))
        tk.Button(btn_frame, text="Clear Terminal", command=self.clear_terminal).pack(side="left")
        
        self.status_label = tk.Label(self.root, text="", fg="gray")
        self.status_label.pack(anchor="w", padx=10, pady=5)
        
        # Separator
        tk.Frame(self.root, height=1, bg="gray").pack(fill="x", padx=10, pady=5)
        
        # Question section
        tk.Label(self.root, text="2. Ask Question").pack(anchor="w", padx=10, pady=(0,5))
        
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=10)
        
        self.question_entry = tk.Entry(input_frame)
        self.question_entry.pack(side="left", fill="x", expand=True, padx=(0,5))
        self.question_entry.bind("<Return>", lambda e: self.ask_question())
        
        self.ask_btn = tk.Button(input_frame, text="Ask", command=self.ask_question)
        self.ask_btn.pack(side="right")
        
        # Response section
        tk.Label(self.root, text="Response:").pack(anchor="w", padx=10, pady=(10,5))
        
        self.response_text = tk.Text(self.root, height=8, wrap="word")
        self.response_text.pack(fill="x", padx=10)
        
        # Separator
        tk.Frame(self.root, height=1, bg="gray").pack(fill="x", padx=10, pady=10)
        
        # Terminal section
        tk.Label(self.root, text="Terminal Output:").pack(anchor="w", padx=10, pady=(0,5))
        
        terminal_frame = tk.Frame(self.root)
        terminal_frame.pack(fill="both", expand=True, padx=10, pady=(0,10))
        
        self.terminal = tk.Text(terminal_frame, height=12, bg="black", fg="lime", 
                                font=("Consolas", 9), wrap="word")
        self.terminal.pack(side="left", fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(terminal_frame, command=self.terminal.yview)
        scrollbar.pack(side="right", fill="y")
        self.terminal.config(yscrollcommand=scrollbar.set)
        
        self.update_status()
        self.log("Ready. Upload videos and click 'Process Videos' to start.")
        
    def log(self, text):
        """Add text to terminal"""
        self.terminal.insert("end", text + "\n")
        self.terminal.see("end")
        
    def log_safe(self, text):
        """Thread-safe logging"""
        self.root.after(0, lambda: self.log(text))
        
    def clear_terminal(self):
        self.terminal.delete("1.0", "end")
        
    def load_embeddings(self):
        embeddings_path = os.path.join(self.base_dir, "embeddings.joblib")
        matrix_path = os.path.join(self.base_dir, "embedding_matrix.npy")
        
        if os.path.exists(embeddings_path):
            try:
                self.df = joblib.load(embeddings_path)
                if os.path.exists(matrix_path):
                    self.embedding_matrix = np.load(matrix_path)
                else:
                    self.embedding_matrix = np.vstack(self.df['embedding'].values)
            except:
                self.df = None
                
    def update_status(self):
        videos = len([f for f in os.listdir(self.videos_dir) if f.endswith(('.mp4','.avi','.mkv','.mov'))])
        chunks = len(self.df) if self.df is not None else 0
        self.status_label.config(text=f"{videos} videos | {chunks} chunks indexed")
        
    def upload_videos(self):
        files = filedialog.askopenfilenames(
            title="Select Videos",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")]
        )
        if files:
            for f in files:
                dest = os.path.join(self.videos_dir, os.path.basename(f))
                if not os.path.exists(dest):
                    shutil.copy2(f, dest)
                    self.log(f"Uploaded: {os.path.basename(f)}")
            self.update_status()
            
    def process_videos(self):
        videos = [f for f in os.listdir(self.videos_dir) if f.endswith(('.mp4','.avi','.mkv','.mov'))]
        if not videos:
            messagebox.showwarning("No Videos", "Upload videos first!")
            return
            
        self.process_btn.config(state="disabled", text="Processing...")
        self.ask_btn.config(state="disabled")
        self.clear_terminal()
        threading.Thread(target=self._process, daemon=True).start()
        
    def run_script(self, script_name, step_name):
        """Run a script and stream output to terminal"""
        self.log_safe(f"\n{'='*40}")
        self.log_safe(f"  {step_name}")
        self.log_safe(f"{'='*40}")
        
        process = subprocess.Popen(
            [sys.executable, script_name],
            cwd=self.base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in process.stdout:
            self.log_safe(line.rstrip())
            
        process.wait()
        return process.returncode == 0
        
    def _process(self):
        try:
            self._status("Step 1/3: Converting videos...")
            if not self.run_script("video_to_mp3.py", "Step 1/3: Converting Videos"):
                self._status("Error in video conversion")
                return
            
            self._status("Step 2/3: Transcribing (may take minutes)...")
            if not self.run_script("mp3_to_json.py", "Step 2/3: Transcribing Audio"):
                self._status("Error in transcription")
                return
            
            self._status("Step 3/3: Creating embeddings...")
            if not self.run_script("preprocess_json.py", "Step 3/3: Creating Embeddings"):
                self._status("Error in embedding creation")
                return
            
            self.load_embeddings()
            chunks = len(self.df) if self.df is not None else 0
            self._status(f"Done! {chunks} chunks indexed")
            self.log_safe(f"\n[SUCCESS] Processing complete! {chunks} chunks indexed.")
            
        except Exception as e:
            self._status(f"Error: {e}")
            self.log_safe(f"Error: {e}")
        finally:
            self.root.after(0, lambda: self.process_btn.config(state="normal", text="Process Videos"))
            self.root.after(0, lambda: self.ask_btn.config(state="normal"))
            self.root.after(0, self.update_status)
            
    def _status(self, text):
        self.root.after(0, lambda: self.status_label.config(text=text))
        
    def ask_question(self):
        question = self.question_entry.get().strip()
        if not question:
            return
        if self.df is None:
            messagebox.showwarning("No Data", "Process videos first!")
            return
            
        self.ask_btn.config(state="disabled")
        self.response_text.delete("1.0", "end")
        self.response_text.insert("1.0", "Thinking...")
        self.log(f"\nQuestion: {question}")
        threading.Thread(target=self._answer, args=(question,), daemon=True).start()
        
    def _answer(self, question):
        try:
            self.log_safe("Creating embedding...")
            r = requests.post("http://localhost:11434/api/embed", 
                json={"model": "nomic-embed-text", "input": [question]},
                timeout=30)
            r.raise_for_status()
            q_embed = r.json()["embeddings"][0]
            
            self.log_safe("Searching... (top 5 results)")
            matrix = self.embedding_matrix if self.embedding_matrix is not None else np.vstack(self.df['embedding'].values)
            sims = cosine_similarity(matrix.astype(np.float32), 
                                    np.array([q_embed], dtype=np.float32)).flatten()
            top_idx = sims.argsort()[::-1][:5]
            chunks = self.df.loc[top_idx]
            
            self.log_safe("Generating response...")
            # Use simpler prompt for faster response
            prompt = f'''Video subtitles:
{chunks[["title", "number", "start", "end", "text"]].to_json(orient="records")}

Q: {question}
Provide a concise answer with video/timestamp references only.'''
            
            r = requests.post("http://localhost:11434/api/generate", 
                json={
                    "model": "qwen2.5:1.5b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 100, "num_gpu": 99, "temperature": 0.3}
                },
                timeout=60)
            r.raise_for_status()
            response = r.json()["response"]
            
            self.log_safe("Done!")
            self.root.after(0, lambda: self._show_response(response))
        except requests.exceptions.Timeout:
            self.log_safe("Error: Request timeout (is Ollama running?)")
            self.root.after(0, lambda: self._show_response("Error: Connection timeout. Make sure Ollama is running: ollama serve"))
        except requests.exceptions.ConnectionError:
            self.log_safe("Error: Cannot connect to Ollama")
            self.root.after(0, lambda: self._show_response("Error: Cannot connect to Ollama. Start it with: ollama serve"))
        except Exception as e:
            self.log_safe(f"Error: {e}")
            self.root.after(0, lambda: self._show_response(f"Error: {e}"))
            
    def _show_response(self, text):
        self.response_text.delete("1.0", "end")
        self.response_text.insert("1.0", text)
        self.ask_btn.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    app = RAGDashboard(root)
    root.mainloop()
