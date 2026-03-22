from pathlib import Path
import json

CONFIG_PATH = Path(__file__).parent.parent / "config" / "defaults.json"

def load_defaults():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def save_defaults(data):
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)
