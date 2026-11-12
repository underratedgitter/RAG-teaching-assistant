import requests
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
# Batches in flight at once. Ollama serialises work internally, so a few is
# plenty — this is about not leaving the GPU idle between requests.
EMBED_CONCURRENCY = int(os.environ.get("EMBED_CONCURRENCY", "3"))

# Exit code when some inputs failed but the rest were processed.
PARTIAL_FAILURE = 2

CACHE_DIR = ".embed_cache"

# nomic-embed-text is trained with task prefixes: a document has to be
# embedded as "search_document: <text>" (and a query as "search_query: <text>")
# or the vectors are ranked badly. Other embedding models take the raw text.
DOCUMENT_PREFIX = "search_document: " if EMBED_MODEL.startswith("nomic-embed-text") else ""


def cache_key(texts):
    """Identity of a file's chunked text, plus the model that embedded it."""
    h = hashlib.sha256()
    h.update(EMBED_MODEL.encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def load_cached(json_file, texts):
    """Embeddings from a previous run, if this file's text is unchanged."""
    path = os.path.join(CACHE_DIR, f"{json_file}.npz")
    if not os.path.exists(path):
        return None
    try:
        # allow_pickle=False: a cached file must never be able to run code.
        with np.load(path, allow_pickle=False) as blob:
            key = blob["key"].item()
            embeddings = blob["embeddings"]
            rows = embeddings.shape[0] if embeddings.ndim else 0
            if key == cache_key(texts) and rows == len(texts):
                return embeddings.tolist()
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        pass
    return None


def save_cached(json_file, texts, embeddings):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{json_file}.npz")
    tmp = path + ".partial"
    try:
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(len(texts), -1) if array.size else np.zeros((0, 0), dtype=np.float32)
        # The file object is passed to np.savez on purpose: given a filename,
        # numpy would append ".npz" to the temp path.
        with open(tmp, "wb") as handle:
            np.savez(handle, key=np.array(cache_key(texts)), embeddings=array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except Exception as e:
        print(f"   [WARNING] Could not cache embeddings for {json_file}: {e}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _plain(value):
    """numpy scalars are not JSON types; unwrap them before dumping."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_artifacts(df, corpus_id):
    """Write the chunk records and the embedding matrix as one validated pair.

    ``chunks.json`` carries the corpus id, the embedding model and the task
    prefix; the matrix sidecar records the same identity. The dashboard refuses
    artifacts whose corpus or model does not match, so a crash between writes
    cannot leave row numbers pointing at the wrong chunks, and changing
    EMBED_MODEL cannot silently reuse vectors from the previous model.

    Nothing is pickled: reading a tampered artifact would otherwise execute
    code.
    """
    tmp_chunks = "chunks.json.partial"
    tmp_matrix = "embedding_matrix.npy.partial"
    tmp_meta = "embedding_matrix.meta.json.partial"

    try:
        matrix = np.vstack(df["embedding"].values).astype(np.float32)
        with open(tmp_matrix, "wb") as handle:
            np.save(handle, matrix)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_matrix, "embedding_matrix.npy")

        meta = {"corpus_id": corpus_id, "rows": len(df), "model": EMBED_MODEL}
        with open(tmp_meta, "w", encoding="utf-8") as handle:
            json.dump(meta, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_meta, "embedding_matrix.meta.json")

        columns = [c for c in df.columns if c != "embedding"]
        chunks = [
            {name: _plain(value) for name, value in row.items()}
            for row in df[columns].to_dict(orient="records")
        ]
        payload = {
            "corpus_id": corpus_id,
            "model": EMBED_MODEL,
            "document_prefix": DOCUMENT_PREFIX,
            "chunks": chunks,
        }
        # Written last: the chunk records are what the dashboard reads to
        # decide the artifacts are current, so they must not exist before the
        # matrix they describe.
        with open(tmp_chunks, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_chunks, "chunks.json")
    finally:
        for path in (tmp_chunks, tmp_matrix, tmp_meta):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass

# Session with retry logic and connection pooling.
#
# allowed_methods matters: urllib3 only retries idempotent verbs by default
# (GET, PUT, HEAD, DELETE, OPTIONS, TRACE), so without naming POST this whole
# retry block did nothing — every embedding call had zero retries.
session = requests.Session()
retry = Retry(
    total=4,
    connect=3,
    read=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=frozenset(["POST", "GET"]),
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
session.mount('http://', adapter)
session.mount('https://', adapter)   # a remote OLLAMA_URL over TLS gets the same retries

def _embed_batch(batch):
    """Embed one batch, falling back to one-at-a-time if the batch fails."""
    try:
        r = session.post(f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": batch},
            timeout=120)
        r.raise_for_status()
        return r.json()["embeddings"]
    except requests.exceptions.RequestException as e:
        print(f"   [WARNING] Embedding request failed: {e}")
        if len(batch) == 1:
            raise
        # Retrying one at a time isolates a transient or size-related batch
        # failure. A text that still fails alone raises, so the file is
        # reported as failed rather than saved with misaligned embeddings.
        out = []
        for text in batch:
            r = session.post(f"{OLLAMA_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": [text]},
                timeout=120)
            r.raise_for_status()
            out.extend(r.json()["embeddings"])
        return out


def create_embedding(text_list, batch_size=128):
    """Embed a list of texts, several batches in flight at once.

    Ollama serves concurrent requests, and a single batch leaves the GPU idle
    while the next one is being serialised and sent. Overlapping a few keeps
    it fed. Results are reassembled in the original order — retrieval maps
    embeddings to chunks positionally, so order is not optional.
    """
    if not text_list:
        return []

    batches = [text_list[i:i + batch_size] for i in range(0, len(text_list), batch_size)]
    if len(batches) == 1:
        return _embed_batch(batches[0])

    results = [None] * len(batches)
    with ThreadPoolExecutor(max_workers=EMBED_CONCURRENCY) as pool:
        futures = {pool.submit(_embed_batch, b): i for i, b in enumerate(batches)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()   # raises if a batch failed

    return [vec for batch in results for vec in batch]


def merge_segments(chunks, target_words=150, overlap_words=30):
    """Merge small Whisper segments into larger overlapping chunks for retrieval.

    The buffer holds whole segments rather than bare strings, so a chunk's
    start/end span every segment it contains — including the overlap carried
    over from the previous chunk. Previously `start` was reset to the segment
    that triggered the flush, which put every overlapped chunk's timestamp
    ~12 seconds later than the words it actually quotes.

    The running word count also replaces a `sum(...)` that re-split every
    buffered segment on every iteration, which made this quadratic.
    """
    if not chunks:
        return chunks

    title = chunks[0].get("title", "")
    number = chunks[0].get("number", "1")

    merged = []
    buf = []       # segment dicts, so overlap keeps its own timing
    buf_words = 0  # running total

    def flush():
        merged.append({
            "number": number, "title": title,
            "start": buf[0]["start"],
            "end": buf[-1]["end"],
            "text": " ".join(s["text"] for s in buf).strip(),
        })

    for seg in chunks:
        if buf_words >= target_words:
            flush()
            # Carry the tail forward as overlap, timestamps included. The
            # overlap must stay below the target: a carried segment that
            # already fills a chunk would trigger a flush on every following
            # segment and repeat itself in each of those chunks.
            tail, running = [], 0
            for s in reversed(buf):
                words = len(s["text"].split())
                if running >= overlap_words or running + words >= target_words:
                    break
                running += words
                tail.insert(0, s)
            buf, buf_words = tail, running
        buf.append(seg)
        buf_words += len(seg["text"].split())

    if buf:
        flush()
    return merged


print("="*50)
print("  Generating Embeddings")
print("="*50)

start_time = time.time()

os.makedirs("jsons", exist_ok=True)
# Sorted so chunk_ids are stable from run to run; listdir order is arbitrary.
jsons = sorted(f for f in os.listdir("jsons") if f.endswith('.json'))
print(f"\n[*] Found {len(jsons)} JSON files")

my_dicts = []
chunk_id = 0
total_chunks = 0
failed = []
cached_files = 0

for json_idx, json_file in enumerate(jsons, 1):
    try:
        with open(f"jsons/{json_file}", encoding="utf-8") as f:
            content = json.load(f)

        raw_chunks = content['chunks']
        # Merge small segments into bigger overlapping chunks
        merged_chunks = merge_segments(raw_chunks, target_words=150, overlap_words=30)
        chunk_count = len(merged_chunks)
        print(f"\n[+] [{json_idx}/{len(jsons)}] {json_file}: {len(raw_chunks)} segs -> {chunk_count} chunks")

        # Embedding is the expensive stage, and adding one lecture used to
        # re-embed the entire library. Cached by a hash of this file's chunk
        # text plus the model name, so unchanged files cost nothing and a
        # changed transcript or a different model invalidates itself. The
        # hash covers the prefixed text, so switching the task prefix cannot
        # reuse vectors that were embedded without it.
        texts = [DOCUMENT_PREFIX + c['text'] for c in merged_chunks]
        embeddings = load_cached(json_file, texts)
        if embeddings is None:
            embeddings = create_embedding(texts, batch_size=128)
            save_cached(json_file, texts, embeddings)
        else:
            cached_files += 1
            print("   [=] Reusing cached embeddings")

        for i, chunk in enumerate(merged_chunks):
            chunk['chunk_id'] = chunk_id
            chunk['embedding'] = embeddings[i]
            chunk_id += 1
            my_dicts.append(chunk)

        # Counted only once the chunks are actually embedded and kept, so a
        # skipped file cannot inflate the total.
        total_chunks += chunk_count
        print(f"   [OK] Embedded {chunk_count} chunks")
    except Exception as e:
        # Embedding is the slow part of this pipeline. Aborting the whole run
        # for one unreadable file threw away every file already embedded, so
        # record it and keep going — the failures are reported at the end.
        print(f"   [ERROR] Failed to process {json_file}: {e}")
        failed.append((json_file, str(e)))
        continue

if not my_dicts:
    print("\n[ERROR] No chunks to process!")
    if failed:
        print("    every file failed:")
        for name, err in failed:
            print(f"      {name}: {err}")
    sys.exit(1)

print(f"\n[>] Creating DataFrame with {len(my_dicts)} total chunks...")
df = pd.DataFrame.from_records(my_dicts)

# Tie the chunk records and the matrix together. The id covers every chunk
# text (with its task prefix) and the embedding model, so a matrix from
# another run cannot be mistaken for this one even when both happen to have
# the same number of rows.
corpus_id = cache_key([DOCUMENT_PREFIX + chunk["text"] for chunk in my_dicts])

print("[*] Saving embeddings...")
save_artifacts(df, corpus_id)

elapsed = time.time() - start_time
print("\n[SUCCESS] Complete!")
print(f"   Files embedded: {len(jsons) - len(failed)}/{len(jsons)}"
      f"  ({cached_files} reused from cache)")
print(f"   Total chunks: {total_chunks}")
print(f"   Time: {elapsed:.1f}s")
if elapsed > 0:
    print(f"   Speed: {total_chunks/elapsed:.1f} chunks/sec")

if failed:
    print(f"\n[WARNING] {len(failed)} file(s) were skipped:")
    for name, err in failed:
        print(f"     {name}: {err}")
    print("   Everything else was embedded and saved.")

session.close()

if failed:
    sys.exit(PARTIAL_FAILURE)

