from pathlib import Path
from datetime import datetime

def save_logbook(text, html, trace, out_dir="output"):
    run_dir = Path(out_dir) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "logbook.txt").write_text(text, encoding="utf-8")
    (run_dir / "logbook.html").write_text(html, encoding="utf-8")
    (run_dir / "trace.raw").write_text("\n".join(map(str, trace)))
