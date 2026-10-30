# RAG Teaching Assistant Dashboard - Updated April 2026
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import shutil
import threading
import subprocess
import sys
import time
import platform
import json
import numpy as np
import joblib
import requests
from requests.adapters import HTTPAdapter

# ── Retrieval tuning ─────────────────────────────────────────────────────────
# Raise TOP_K to give the model more context at the cost of a longer prompt;
# raise the threshold to be stricter about relevance.
TOP_K = 8                   # chunks considered for the answer
SIMILARITY_THRESHOLD = 0.3  # below this a chunk is treated as unrelated
FALLBACK_K = 3              # used when nothing clears the threshold

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "qwen2.5:1.5b")


def _has_gpu_offload():
    """Whether Ollama can offload layers to a GPU on this machine.

    Ollama uses Metal on Apple Silicon and CUDA/ROCm on Linux and Windows.
    An Intel Mac has neither, so asking for 99 GPU layers there is at best
    ignored — it is not a portable default.
    """
    if sys.platform == "darwin":
        return platform.machine() == "arm64"      # Metal, Apple Silicon only
    return True                                    # let Ollama decide elsewhere


GENERATION_OPTIONS = {"num_predict": 300}


def _recorded_models(df, meta):
    """Model identities recorded on the dataframe and the matrix sidecar."""
    attrs = getattr(df, "attrs", None) or {}
    df_model = attrs.get("embedding_model") or attrs.get("model")
    meta_model = meta.get("model") if isinstance(meta, dict) else None
    return df_model, meta_model


def _dataframe_embeddings_match_model(df, meta, current_model):
    """Whether the dataframe's embedding column came from ``current_model``.

    A dataframe with no model identity is untrusted: it could have been
    produced by any model, and querying it with the current one silently
    compares vectors from different embedding spaces. Artifacts written before
    the dataframe carried this metadata can still prove their identity through
    the matrix sidecar, which has recorded the model since it was introduced.
    """
    df_model, meta_model = _recorded_models(df, meta)
    if df_model is not None:
        return df_model == current_model
    corpus_id = getattr(df, "attrs", {}).get("corpus_id")
    return bool(corpus_id) and meta.get("corpus_id") == corpus_id and meta_model == current_model


def _matrix_matches_dataframe(df, meta, current_model=None):
    """Whether a matrix sidecar belongs to this dataframe's corpus and model.

    ``current_model`` is optional only for callers that inspect legacy
    artifacts; the dashboard always passes it, so a matrix with no recorded
    model is rejected rather than trusted by default.
    """
    if not isinstance(meta, dict):
        return False
    corpus_id = getattr(df, "attrs", {}).get("corpus_id")
    if not (bool(corpus_id) and meta.get("corpus_id") == corpus_id and meta.get("rows") == len(df)):
        return False
    if current_model is None:
        return True
    return meta.get("model") == current_model


if _has_gpu_offload():
    GENERATION_OPTIONS["num_gpu"] = 99

# RAG Teaching Assistant Dashboard Interface
class RAGDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("RAG Teaching Assistant")
        self.root.geometry("700x650")

        try:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
            self.videos_dir = os.path.join(self.base_dir, "videos")
            self.audios_dir = os.path.join(self.base_dir, "audios")
            self.jsons_dir = os.path.join(self.base_dir, "jsons")

            # Create directories with error handling
            for dir_path in [self.videos_dir, self.audios_dir, self.jsons_dir]:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except Exception as e:
                    print(f"Warning: Could not create {dir_path}: {e}")

            self._clear_work_dirs_on_start()

            self.df = None
            self.embedding_matrix = None
            self.load_embeddings()

            # Reusable HTTP session with connection pooling
            self.session = requests.Session()
            adapter = HTTPAdapter(pool_connections=2, pool_maxsize=2)
            self.session.mount('http://', adapter)

            self.create_ui()

            # Auto-start Ollama if not running
            threading.Thread(target=self._ensure_ollama_running, daemon=True).start()

            # Pre-warm Ollama models in background so first query is fast
            threading.Thread(target=self._warm_models, daemon=True).start()

        except Exception as e:
            print(f"Error during initialization: {e}")
            self.create_ui()  # Still create UI so app doesn't crash

    def _clear_work_dirs_on_start(self):
        """Remove all videos, audios, jsons, and embedding cache on launch.

        Off by default. video_to_mp3.py and mp3_to_json.py both skip files
        they have already produced, so that work is meant to be reused —
        wiping it every launch made that skip logic unreachable and forced a
        full re-transcription each session, which is minutes of GPU time per
        lecture.

        Set CLEAR_ON_START=1 to restore the old behaviour for a clean demo.
        """
        if os.environ.get("CLEAR_ON_START") != "1":
            return

        dirs = [self.videos_dir, self.audios_dir, self.jsons_dir]
        for path in dirs:
            try:
                if os.path.exists(path):
                    for name in os.listdir(path):
                        file_path = os.path.join(path, name)
                        try:
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception:
                            pass
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
        tk.Button(btn_frame, text="Open Folder", command=self.open_videos_folder).pack(side="left", padx=(0,5))
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

    def open_videos_folder(self):
        """Reveal the videos folder in the system file manager.

        os.startfile does not exist outside Windows, so this button raised
        AttributeError on macOS and Linux the moment it was clicked.
        """
        path = self.videos_dir
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif sys.platform.startswith("win"):
                os.startfile(path)          # noqa: S606 — Windows only
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            self.log(f"Could not open {path}: {e}")

    def _ensure_ollama_running(self):
        """Check if Ollama is running, if not start it automatically."""
        max_attempts = 30  # Try for 30 seconds
        attempt = 0
        ollama_started = False

        while attempt < max_attempts:
            try:
                # Try to connect to Ollama
                response = self.session.get(f"{OLLAMA_URL}/api/tags", timeout=2)
                if response.status_code == 200:
                    self.log_safe("✓ Connected to Ollama server")
                    return True
            except Exception:
                pass

            attempt += 1
            if attempt == 1:
                self.log_safe("⏳ Ollama not detected. Attempting to start Ollama...")
                try:
                    # Start Ollama in background
                    subprocess.Popen("ollama serve", shell=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
                    ollama_started = True
                except Exception as e:
                    self.log_safe(f"⚠ Could not auto-start Ollama: {str(e)}")

            time.sleep(1)

        if ollama_started:
            self.log_safe("⚠ Ollama started but not ready yet. Please try processing after a moment.")
        else:
            self.log_safe("⚠ Ollama not responding. Make sure Ollama is installed and run: ollama serve")
        return False

    def _warm_models(self):
        """Pre-warm Ollama models to avoid cold-start latency."""
        try:
            # Wait a bit for Ollama to be ready
            time.sleep(2)

            self.session.post(f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": ["warmup"]}, timeout=30)
            self.session.post(f"{OLLAMA_URL}/api/generate",
                json={"model": ANSWER_MODEL, "prompt": "hi", "stream": False,
                      "options": {"num_predict": 1}}, timeout=30)
            self.log_safe("✓ Ollama models warmed up - ready for queries.")
        except Exception:
            self.log_safe("⚠ Model warm-up in progress... (this is normal)")

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
        """Load cached embeddings, rejecting another corpus or embedding model."""
        embeddings_path = os.path.join(self.base_dir, "embeddings.joblib")
        matrix_path = os.path.join(self.base_dir, "embedding_matrix.npy")
        meta_path = os.path.join(self.base_dir, "embedding_matrix.meta.json")

        try:
            if not os.path.exists(embeddings_path):
                return

            self.df = joblib.load(embeddings_path)
            meta = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, encoding="utf-8") as handle:
                        meta = json.load(handle)
                except (OSError, ValueError, UnicodeDecodeError):
                    meta = {}

            if not _dataframe_embeddings_match_model(self.df, meta, EMBED_MODEL):
                # Never mix vectors from two models. The dataframe's own
                # embeddings are unusable here, and rebuilding the matrix from
                # them would just preserve the mismatch, so discard both: the
                # next Process Videos run re-embeds from the transcripts.
                df_model, meta_model = _recorded_models(self.df, meta)
                recorded = df_model or meta_model or "unknown model"
                print(
                    f"[WARNING] Cached embeddings were created with {recorded!r}, "
                    f"not {EMBED_MODEL!r}; ignoring them so preprocessing can rebuild."
                )
                self.df = None
                self.embedding_matrix = None
                return

            # Older artifacts have the identity only in the sidecar. Stamp the
            # dataframe in memory so later checks do not fall back again.
            if getattr(self.df, "attrs", None) is not None:
                self.df.attrs.setdefault("embedding_model", EMBED_MODEL)

            matrix_ok = False
            if os.path.exists(matrix_path) and _matrix_matches_dataframe(
                self.df, meta, EMBED_MODEL
            ):
                try:
                    matrix = np.load(matrix_path)
                    expected_dim = len(self.df["embedding"].iloc[0])
                    if (
                        matrix.ndim == 2
                        and matrix.shape[0] == len(self.df)
                        and matrix.shape[1] == expected_dim
                    ):
                        self.embedding_matrix = matrix
                        matrix_ok = True
                except Exception:
                    matrix_ok = False

            if not matrix_ok:
                # The dataframe is authoritative: rebuild the matrix from its
                # own embedding column rather than trusting a stale .npy file.
                self.embedding_matrix = np.vstack(self.df["embedding"].values)

            self._prepare_matrix()
        except Exception:
            self.df = None
            self.embedding_matrix = None

    def _prepare_matrix(self):
        """Normalise the corpus once, at load, instead of on every question.

        With unit-length rows, cosine similarity is just a dot product, so a
        query no longer recomputes norms for the whole corpus. This also stops
        the per-query `.astype(np.float32)` that copied the entire matrix
        (~30 MB at 10k chunks) every time someone asked something.
        """
        if self.embedding_matrix is None:
            return
        m = np.ascontiguousarray(self.embedding_matrix, dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0          # a zero vector would divide by zero
        self.embedding_matrix = m / norms

    def update_status(self):
        """Update status label with error handling"""
        try:
            videos = 0
            chunks = 0

            try:
                if os.path.exists(self.videos_dir):
                    videos = len([f for f in os.listdir(self.videos_dir) if f.endswith(('.mp4','.avi','.mkv','.mov'))])
            except Exception:
                videos = 0

            chunks = len(self.df) if self.df is not None else 0
            self.status_label.config(text=f"{videos} videos | {chunks} chunks indexed")
        except Exception:
            self.status_label.config(text="Ready")

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
            errors="replace",   # a non-UTF-8 byte in tool output must not kill the pipeline thread
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
        # The <Return> binding bypasses the disabled Ask button, which let a
        # question start mid-processing (while embeddings reload) or overlap
        # an answer already in flight.
        if str(self.ask_btn["state"]) == "disabled":
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
            r = self.session.post(f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": [question]},
                timeout=30)
            r.raise_for_status()
            q_embed = r.json()["embeddings"][0]

            self.log_safe("Searching... (top 8 results)")
            matrix = self.embedding_matrix
            if matrix is None:
                matrix = np.vstack(self.df['embedding'].values)
                self.embedding_matrix = matrix
                self._prepare_matrix()
                matrix = self.embedding_matrix

            # Rows are unit-length (see _prepare_matrix), so cosine similarity
            # reduces to a dot product — no per-query norm computation.
            q = np.asarray(q_embed, dtype=np.float32)
            qn = np.linalg.norm(q)
            sims = matrix @ (q / (qn if qn else 1.0))

            # argpartition is O(n); a full argsort was sorting every chunk to
            # read the top 8.
            k = min(TOP_K, sims.shape[0])
            cand = np.argpartition(-sims, k - 1)[:k]
            cand = cand[np.argsort(-sims[cand])]

            top_idx = [int(i) for i in cand if sims[i] > SIMILARITY_THRESHOLD]
            if not top_idx:
                top_idx = [int(i) for i in cand[:FALLBACK_K]]

            # iloc, not loc: these are positional indices from argsort. They
            # only coincide with labels while the frame has a clean RangeIndex.
            chunks = self.df.iloc[top_idx]

            self.log_safe("Generating response...")
            # Build readable context with timestamps
            context_parts = []
            for _, row in chunks.iterrows():
                mins_s, secs_s = divmod(int(row['start']), 60)
                mins_e, secs_e = divmod(int(row['end']), 60)
                context_parts.append(
                    f"[Video {row['number']}: {row['title']} | {mins_s}:{secs_s:02d}-{mins_e}:{secs_e:02d}]\n{row['text']}"
                )
            context_str = "\n\n".join(context_parts)

            prompt = f'''You are a teaching assistant. Answer the student's question using ONLY the lecture transcript excerpts below. Be accurate and helpful.

--- LECTURE EXCERPTS ---
{context_str}
--- END EXCERPTS ---

Student Question: {question}

Instructions:
- Answer based strictly on the provided excerpts.
- Reference the video number and timestamp (e.g., "In Video 1 at 2:30...").
- If the excerpts don't contain enough info, say so.
- Be concise but thorough.'''

            r = self.session.post(f"{OLLAMA_URL}/api/generate",
                json={
                    "model": ANSWER_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {**GENERATION_OPTIONS, "temperature": 0.2, "top_p": 0.9}
                },
                timeout=120)
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
            # Bind the message now: Python unbinds `e` when the except block
            # ends, before Tk runs the callback.
            message = f"Error: {e}"
            self.root.after(0, lambda: self._show_response(message))

    def _show_response(self, text):
        self.response_text.delete("1.0", "end")
        self.response_text.insert("1.0", text)
        self.ask_btn.config(state="normal")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = RAGDashboard(root)
        root.mainloop()
    except Exception as e:
        print(f"Fatal error: {e}")
        # Try to show error in GUI
        try:
            messagebox.showerror("Error", f"Application error: {e}\n\nCheck that all dependencies are installed:\npip install -r requirements.txt")
        except Exception:
            print(f"Could not display error dialog: {e}")
        sys.exit(1)
