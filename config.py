import os
from pathlib import Path

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
A1111_URL = os.getenv("A1111_URL", "http://127.0.0.1:7860").rstrip("/")

BASE_DIR = Path(__file__).resolve().parent
WORKFLOW_DIR = Path(os.getenv("COMFYUI_WORKFLOW_DIR", str(BASE_DIR / "workflows")))
COMFYUI_OUTPUT_DIR = Path(os.getenv("COMFYUI_OUTPUT_DIR", str(BASE_DIR / "output")))

ASPECT_RATIOS = {
    "16:9": (1344, 768),
    "16:9_full": (1920, 1080),
    "9:16": (768, 1344),
    "9:16_full": (1080, 1920),
    "1:1": (1024, 1024),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1216, 832),
    "2:3": (832, 1216),
    "21:9": (1536, 640),
    "9:21": (640, 1536),
}
