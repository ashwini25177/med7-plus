"""
Direct launcher for the Streamlit web dashboard.
Run with: python run_app.py
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
app_path = PROJECT_ROOT / "web_app" / "app_streamlit.py"

if __name__ == "__main__":
    print(f"[*] Starting Med7-Plus Streamlit UI from {app_path} ...")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd)
