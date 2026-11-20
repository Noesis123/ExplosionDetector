import av
import numpy as np
from pathlib import Path

f = Path(r"C:\Users\Aman Yazeen\.gemini\antigravity\brain\ad076906-e1fe-4046-9a40-92f5f2339702\.user_uploaded\uploaded_media_0_1795077722079.m4a")

container = av.open(str(f))
stream = container.streams.audio[0]
frames = []
for frame in container.decode(stream):
    frames.append(frame.to_ndarray())

# frame.to_ndarray() has shape (channels, samples) for planar or (1, samples*channels) depending on format
audio = np.concatenate(frames, axis=-1)
print(f"Decoded successfully! Shape: {audio.shape}, dtype: {audio.dtype}, sample_rate: {stream.rate}")
