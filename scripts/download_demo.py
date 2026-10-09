"""Download official demonstration assets and make an illustrative support crop."""
from pathlib import Path
import requests
from PIL import Image

root = Path(__file__).resolve().parents[1]
folder = root / "data" / "demo"
folder.mkdir(parents=True, exist_ok=True)
url = "https://raw.githubusercontent.com/timojl/clipseg/master/example_image.jpg"
response = requests.get(url, timeout=60)
response.raise_for_status()
path = folder / "example_image.jpg"
path.write_bytes(response.content)
image = Image.open(path).convert("RGB")
# Demonstration only: same-query crop, not the disjoint support protocol used by E2.
image.crop((160, 145, 330, 235)).save(folder / "support.jpg")
print(f"Saved {path} and illustrative spoon crop support.jpg")
