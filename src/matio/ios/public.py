import os
import json
import yaml

from matio.ios.path import PUBLIC

def load(file: str) -> str | dict | list:
    """
    Load a file from PUBLIC directory.

    Args:
        file (str): relative path under PUBLIC

    Returns:
        str | dict | list: 
            - .jsonl -> list of dicts/objects
            - .json  -> dict or list
            - .yaml/.yml -> dict or list
            - others -> raw string
    """
    filepath = PUBLIC / file
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Public file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        if filepath.suffix == ".json":
            content = json.load(f)
        elif filepath.suffix in {".yaml", ".yml"}:
            content = yaml.safe_load(f)
        elif filepath.suffix == ".jsonl":
            content = [json.loads(line) for line in f if line.strip()]
        else:
            content = f.read()
    return content
