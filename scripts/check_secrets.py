from pathlib import Path
import re

pat = re.compile(r"(?i)(api[_-]?key|secret[_-]?key|password|token)\s*[:=]\s*['\"]([^'\"]{8,})['\"]")
findings = []

for p in Path(".").rglob("*"):
    if not p.is_file():
        continue
    if any(x in p.parts for x in [".venv312", ".git", "__pycache__", ".pytest_cache"]):
        continue
    if p.suffix.lower() in [".mp3", ".wav", ".pt", ".pth", ".npy", ".pkl", ".bin", ".db", ".png", ".jpg", ".svg", ".m4a"]:
        continue
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(txt.splitlines(), 1):
            m = pat.search(line)
            if m:
                val = m.group(2)
                if not any(ph in val.lower() for ph in ["dummy", "placeholder", "xxx", "none", "your_", "change"]):
                    findings.append((str(p), line_no, m.group(1), val[:4] + "****"))
    except Exception:
        pass

print(f"Total potential secret findings: {len(findings)}")
for f in findings:
    print(f"  {f[0]}:{f[1]} [{f[2]}] = {f[3]}")
