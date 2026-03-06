import requests
import os
import json
import numpy as np
import pandas as pd
import joblib
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Session with retry logic and connection pooling
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
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
    """Merge small Whisper segments into larger overlapping chunks for better retrieval."""
    if not chunks:
        return chunks
    merged = []
    buf_texts, buf_start, buf_end = [], chunks[0]["start"], chunks[0]["end"]
    title, number = chunks[0].get("title", ""), chunks[0].get("number", "1")
    overlap_buf = []  # texts to prepend as overlap from previous chunk

    for seg in chunks:
        word_count = sum(len(t.split()) for t in buf_texts)
        if word_count >= target_words:
            merged.append({
                "number": number, "title": title,
                "start": buf_start, "end": buf_end,
                "text": " ".join(buf_texts).strip()
            })
            # keep last few texts as overlap for next chunk
            overlap_buf = []
            running = 0
            for t in reversed(buf_texts):
                running += len(t.split())
                overlap_buf.insert(0, t)
                if running >= overlap_words:
                    break
            buf_texts = list(overlap_buf)
            buf_start = seg["start"]
        buf_texts.append(seg["text"])
        buf_end = seg["end"]

    if buf_texts:
        merged.append({
            "number": number, "title": title,
            "start": buf_start, "end": buf_end,
            "text": " ".join(buf_texts).strip()
        })
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

for json_idx, json_file in enumerate(jsons, 1):
    try:
        with open(f"jsons/{json_file}") as f:
            content = json.load(f)
        
        raw_chunks = content['chunks']
        # Merge small segments into bigger overlapping chunks
        merged_chunks = merge_segments(raw_chunks, target_words=150, overlap_words=30)
        chunk_count = len(merged_chunks)
        total_chunks += chunk_count
        print(f"\n[+] [{json_idx}/{len(jsons)}] {json_file}: {len(raw_chunks)} segs -> {chunk_count} chunks")
        
        # Create embeddings with optimized batch size
        texts = [c['text'] for c in merged_chunks]
        embeddings = create_embedding(texts, batch_size=128)
           
        for i, chunk in enumerate(merged_chunks):
            chunk['chunk_id'] = chunk_id
            chunk['embedding'] = embeddings[i]
            chunk_id += 1
            my_dicts.append(chunk)
        
        print(f"   [OK] Embedded {chunk_count} chunks")
    except Exception as e:
        print(f"   [ERROR] Failed to process {json_file}: {e}")
        raise

if not my_dicts:
    print("\n[ERROR] No chunks to process!")
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
print(f"   Total chunks: {total_chunks}")
print(f"   Time: {elapsed:.1f}s")
if elapsed > 0:
    print(f"   Speed: {total_chunks/elapsed:.1f} chunks/sec")

session.close()

