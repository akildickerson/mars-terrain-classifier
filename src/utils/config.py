import yaml
from pathlib import Path
import os


def load_config(path: str):
    with open(path, "r") as file:
        cfg = yaml.safe_load(file)
    if os.environ.get("NASA_API_KEY"):
        cfg.setdefault("nasa", {})["api_key"] = os.environ.get("NASA_API_KEY")
    return cfg
