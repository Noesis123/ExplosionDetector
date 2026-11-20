import shutil
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.file_loader import load_audio_file, validate_audio_file

UPLOADS = [
    (
        Path(r"C:\Users\Aman Yazeen\.gemini\antigravity\brain\ad076906-e1fe-4046-9a40-92f5f2339702\.user_uploaded\uploaded_media_0_1795077722079.m4a"),
        "Recording (7).m4a",
        "p24_room_ambient_ac_rec07.m4a"
    ),
    (
        Path(r"C:\Users\Aman Yazeen\.gemini\antigravity\brain\ad076906-e1fe-4046-9a40-92f5f2339702\.user_uploaded\uploaded_media_1_1795077722079.m4a"),
        "Recording (8).m4a",
        "p24_room_ambient_ac_rec08.m4a"
    ),
    (
        Path(r"C:\Users\Aman Yazeen\.gemini\antigravity\brain\ad076906-e1fe-4046-9a40-92f5f2339702\.user_uploaded\uploaded_media_2_1795077722079.m4a"),
        "Recording (9).m4a",
        "p24_room_ambient_ac_rec09.m4a"
    )
]

DEST_DIR = Path("data/phase24_approved/room_ambient")
DEST_DIR.mkdir(parents=True, exist_ok=True)

print("Importing Phase 24 Approved Room Ambient Recordings...")
for src_path, original_name, dest_name in UPLOADS:
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    
    dest_path = DEST_DIR / dest_name
    shutil.copy2(src_path, dest_path)
    
    # Calculate sha256
    sha256 = hashlib.sha256(dest_path.read_bytes()).hexdigest()
    
    # Validate audio
    report = validate_audio_file(dest_path)
    audio, sr, dur, ch = load_audio_file(dest_path)
    
    print(f"\nImported: {dest_name} (from {original_name})")
    print(f"  Destination: {dest_path}")
    print(f"  Size: {dest_path.stat().st_size:,} bytes")
    print(f"  SHA256: {sha256[:16]}...")
    print(f"  Sample Rate: {sr} Hz, Channels: {ch}, Duration: {dur:.3f} s")
    print(f"  Peak: {report['peak']}, RMS: {report['rms']}")

print("\nImport complete successfully!")
