"""
Direct launcher for the FastAPI microservice.
Run with: python run_api.py
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

if __name__ == "__main__":
    print(f"[*] Starting Med7-Plus FastAPI server on http://127.0.0.1:8000 ...")
    print(f"[*] Swagger UI available at http://127.0.0.1:8000/docs")
    uvicorn.run("web_app.api:app", host="127.0.0.1", port=8000, reload=True)
