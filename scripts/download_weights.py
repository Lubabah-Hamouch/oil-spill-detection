import argparse
import urllib.request
from pathlib import Path

WEIGHTS = {
    "classification_best.pth": "https://github.com/Lubabah-Hamouch/oil-spill-detection/releases/download/v1.0.0/classification_best.pth",
    "segmentation_best.pth": "https://github.com/Lubabah-Hamouch/oil-spill-detection/releases/download/v1.0.0/segmentation_best.pth",
}


def download_weights(force: bool = False) -> None:
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("OIL SPILL DETECTION — DOWNLOADING MODEL WEIGHTS")
    print("=" * 60)

    for filename, url in WEIGHTS.items():
        destination = models_dir / filename
        if destination.exists() and not force:
            print(f"[{filename}] Already exists locally at {destination}. Skipping.")
            continue

        print(f"[{filename}] Downloading from {url} ...")
        try:
            # Custom Request to handle redirect and User-Agent header
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(destination, "wb") as out_file:
                total_size = response.getheader("Content-Length")
                total_bytes = int(total_size) if total_size else None
                downloaded = 0
                block_size = 8192

                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)

                    if total_bytes:
                        pct = (downloaded / total_bytes) * 100
                        print(f"\r  Progress: {downloaded}/{total_bytes} bytes ({pct:.1f}%)", end="", flush=True)

            print(f"\n[{filename}] Download complete! Saved to {destination}")

        except Exception as e:
            if destination.exists():
                destination.unlink()  # Clean up incomplete download
            print(f"\n[ERROR] Failed to download {filename}: {e}")

    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights for Oil Spill Detection.")
    parser.add_argument("--force", action="store_true", help="Force redownload even if weights already exist.")
    args = parser.parse_args()

    download_weights(force=args.force)
