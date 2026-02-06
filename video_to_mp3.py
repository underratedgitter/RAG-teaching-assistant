# Converts videos to mp3 - Optimized for speed
import os 
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

os.makedirs("audios", exist_ok=True)
existing_audios = set(os.listdir("audios"))

files = [f for f in os.listdir("videos") if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
print(f"Found {len(files)} video files")

def convert_video(args):
    """Convert a single video file"""
    i, file = args
    name = os.path.splitext(file)[0]
    output_name = f"{i}_{name}.mp3"
    
    if output_name in existing_audios:
        return f"Skipping {file} (exists)"
    
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", f"videos/{file}", 
             "-vn", "-acodec", "libmp3lame", "-q:a", "4", "-threads", "0",
             f"audios/{output_name}"],
            capture_output=True,
            timeout=3600
        )
        
        if result.returncode == 0:
            return f"Created {output_name}"
        else:
            return f"Error converting {file}"
    except subprocess.TimeoutExpired:
        return f"Timeout converting {file}"
    except Exception as e:
        return f"Error converting {file}: {e}"

start = time.time()
max_workers = min(4, os.cpu_count() or 2)

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(convert_video, (i, f)) for i, f in enumerate(sorted(files), 1)]
    for future in as_completed(futures):
        print(f"  {future.result()}")

elapsed = time.time() - start
print(f"Done! ({elapsed:.1f}s)")
