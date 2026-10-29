"""Run the API and the UI in one container (for single-container hosts like Hugging Face Spaces)."""

import os
import subprocess
import sys
import time
import urllib.request

API_PORT = 8000
UI_PORT = os.getenv("PORT", "7860")

api = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)]
)

# Wait for the API to finish loading the embedding model before starting the UI.
for _ in range(300):
    if api.poll() is not None:
        sys.exit("API process exited during startup")
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{API_PORT}/health", timeout=2)
        break
    except OSError:
        time.sleep(1)

ui = subprocess.Popen(
    [
        sys.executable, "-m", "streamlit", "run", "ui/streamlit_app.py",
        "--server.port", UI_PORT,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        # Spaces serves the app inside an iframe; without these, file uploads fail with 403.
        "--server.enableXsrfProtection", "false",
        "--server.enableCORS", "false",
    ]
)

# If either process dies, exit so the platform restarts the container.
while api.poll() is None and ui.poll() is None:
    time.sleep(5)
sys.exit(1)
