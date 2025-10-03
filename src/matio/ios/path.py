from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent
ROOT = SOURCE.parent.parent

ASSETS = SOURCE / "assets"
PUBLIC = SOURCE / "public"

MODELS = ROOT / "models"

BANNER = ASSETS / "banner.png"