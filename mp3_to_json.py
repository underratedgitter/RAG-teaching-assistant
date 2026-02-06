import whisper
import json
import os
import torch
import time

# Setup for performance and reliability
os.makedirs("jsons", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using: {device.upper()}")

# Use base model for 3-5x faster transcription
try:
    model = whisper.load_model("base", device=device)
except Exception as e:
    print(f"[WARNING] Failed to load 'base' model: {e}")
    print(f"[INFO] Falling back to 'tiny' model")
    model = whisper.load_model("tiny", device=device)

audios = [f for f in os.listdir("audios") if f.endswith('.mp3')]
existing_jsons = set(os.listdir("jsons"))
print(f"Found {len(audios)} audio files")

start_total = time.time()
results = {"success": 0, "skipped": 0, "failed": 0}

for idx, audio in enumerate(audios, 1):
    json_filename = f"{audio}.json"
    if json_filename in existing_jsons:
        print(f"[{idx}/{len(audios)}] Skipping {audio} (exists)")
        results["skipped"] += 1
        continue
    
    # Extract number and title from filename: "1_VideoName.mp3"
    name = os.path.splitext(audio)[0]  # Remove .mp3
    if "_" in name:
        parts = name.split("_", 1)
        number = parts[0]
        title = parts[1] if len(parts) > 1 else name
    else:
        number = "1"
        title = name
    
    print(f"[{idx}/{len(audios)}] Transcribing: {audio}")
    start = time.time()
    
    try:
        # Validate file exists and is readable
        audio_path = f"audios/{audio}"
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            print(f"  [!] Invalid audio file (empty or missing)")
            results["failed"] += 1
            continue
            
        result = model.transcribe(
            audio=audio_path,
            language="en",
            fp16=False,
            verbose=False,
            task="transcribe"
        )
        
        # Validate transcription result
        if not result or "segments" not in result:
            print(f"  [!] Invalid transcription result")
            results["failed"] += 1
            continue
        
        chunks = []
        for seg in result.get("segments", []):
            try:
                chunk = {
                    "number": number,
                    "title": title,
                    "start": float(seg.get("start", 0)),
                    "end": float(seg.get("end", 0)),
                    "text": str(seg.get("text", "")).strip()
                }
                if chunk["text"]:  # Only add non-empty chunks
                    chunks.append(chunk)
            except (ValueError, KeyError, TypeError) as e:
                print(f"  [WARNING] Skipped malformed segment: {e}")
                continue
        
        if not chunks:
            print(f"  [!] No valid chunks extracted")
            results["failed"] += 1
            continue
        
        # Save with error handling
        try:
            with open(f"jsons/{json_filename}", "w", encoding="utf-8") as f:
                json.dump({
                    "chunks": chunks, 
                    "text": result.get("text", ""),
                    "language": result.get("language", "en")
                }, f, ensure_ascii=False)
            
            elapsed = time.time() - start
            print(f"  [+] Created {len(chunks)} chunks ({elapsed:.1f}s)")
            results["success"] += 1
        except Exception as e:
            print(f"  [!] Failed to save JSON: {e}")
            results["failed"] += 1
        
        # Clear GPU cache
        if device == "cuda":
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"  [!] Transcription failed: {e}")
        results["failed"] += 1
        # Continue with next file instead of crashing
        if device == "cuda":
            try:
                torch.cuda.empty_cache()
            except:
                pass

elapsed_total = time.time() - start_total
print(f"\nDone! ({elapsed_total:.1f}s total)")
print(f"  Success: {results['success']}, Skipped: {results['skipped']}, Failed: {results['failed']}")

if results["success"] > 0 or results["skipped"] > 0:
    print("[STATUS] Ready for next step")
