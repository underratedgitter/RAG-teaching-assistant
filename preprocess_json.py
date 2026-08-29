import requests
import os
import json
import numpy as np
import pandas as pd
import joblib
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

def create_embedding(text_list, batch_size=128):
    """Create embeddings with optimized batching and error handling"""
    all_embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        try:
            r = session.post("http://localhost:11434/api/embed", 
                json={"model": "nomic-embed-text", "input": batch},
                timeout=60)
            r.raise_for_status()
            all_embeddings.extend(r.json()["embeddings"])
        except requests.exceptions.RequestException as e:
            print(f"   [WARNING] Embedding request failed: {e}")
            # Retry with smaller batch
            if len(batch) > 1:
                for text in batch:
                    try:
                        r = session.post("http://localhost:11434/api/embed",
                            json={"model": "nomic-embed-text", "input": [text]},
                            timeout=60)
                        r.raise_for_status()
                        all_embeddings.extend(r.json()["embeddings"])
                    except Exception as e2:
                        print(f"    [ERROR] Failed to embed text: {e2}")
                        raise
            else:
                raise
    
    return all_embeddings


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
            # carry the tail forward as overlap, timestamps included
            tail, running = [], 0
            for s in reversed(buf):
                running += len(s["text"].split())
                tail.insert(0, s)
                if running >= overlap_words:
                    break
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

jsons = [f for f in os.listdir("jsons") if f.endswith('.json')]
print(f"\n[*] Found {len(jsons)} JSON files")

my_dicts = []
chunk_id = 0
total_chunks = 0
failed = []

for json_idx, json_file in enumerate(jsons, 1):
    try:
        with open(f"jsons/{json_file}") as f:
            content = json.load(f)
        
        raw_chunks = content['chunks']
        # Merge small segments into bigger overlapping chunks
        merged_chunks = merge_segments(raw_chunks, target_words=150, overlap_words=30)
        chunk_count = len(merged_chunks)
        print(f"\n[+] [{json_idx}/{len(jsons)}] {json_file}: {len(raw_chunks)} segs -> {chunk_count} chunks")

        # Create embeddings with optimized batch size
        texts = [c['text'] for c in merged_chunks]
        embeddings = create_embedding(texts, batch_size=128)

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
    exit(1)

print(f"\n[>] Creating DataFrame with {len(my_dicts)} total chunks...")
df = pd.DataFrame.from_records(my_dicts)

# Pre-compute the embedding matrix for faster queries
print(f"[*] Saving embeddings...")
joblib.dump(df, 'embeddings.joblib', compress=3)

# Also save the pre-computed matrix for faster similarity search (float32 for memory efficiency)
embedding_matrix = np.vstack(df['embedding'].values).astype(np.float32)
np.save('embedding_matrix.npy', embedding_matrix)

elapsed = time.time() - start_time
print(f"\n[SUCCESS] Complete!")
print(f"   Files embedded: {len(jsons) - len(failed)}/{len(jsons)}")
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

