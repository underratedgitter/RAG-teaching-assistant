import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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
        
        chunk_count = len(content['chunks'])
        total_chunks += chunk_count
        print(f"\n[+] [{json_idx}/{len(jsons)}] {json_file}: {chunk_count} chunks")
        
        # Create embeddings with optimized batch size
        texts = [c['text'] for c in content['chunks']]
        embeddings = create_embedding(texts, batch_size=128)
           
        for i, chunk in enumerate(content['chunks']):
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

