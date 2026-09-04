import os
import sys
import urllib.request
import tarfile
import zipfile

# ------------------------------------------------------------------
# 1. MODEL DOWNLOAD DEFINITIONS
# ------------------------------------------------------------------
MODELS = {
    "sherpa-onnx-nemo-ctc-en-citrinet-512": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2",
        "archive_name": "sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2",
        "extract_type": "tar.bz2"
    },
    "sherpa-onnx-zipformer-en-2023-06-26": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2",
        "archive_name": "sherpa-onnx-zipformer-en-2023-06-26.tar.bz2",
        "extract_type": "tar.bz2"
    },
    "vosk-model-small-en-us-0.15": {
        "url": "https://alphacephei.com/kaldi/models/vosk-model-small-en-us-0.15.zip",
        "archive_name": "vosk-model-small-en-us-0.15.zip",
        "extract_type": "zip"
    }
}

DEST_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------------
def download_progress_hook(count, block_size, total_size):
    """Displays a CLI progress bar during download."""
    downloaded = count * block_size
    percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024) if total_size > 0 else 0
    
    sys.stdout.write(
        f"\r   Progress: [{ '=' * (percent // 2) }{ ' ' * (50 - percent // 2) }] "
        f"{percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)"
    )
    sys.stdout.flush()

def extract_archive(archive_path, extract_dir, archive_type):
    """Extracts tar.bz2 or zip archives safely."""
    print(f"\n   📦 Extracting {os.path.basename(archive_path)}...")
    if archive_type == "tar.bz2":
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(path=extract_dir)
    elif archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(path=extract_dir)
    print("   ✅ Extraction complete.")

# ------------------------------------------------------------------
# 3. MAIN DOWNLOAD ENGINE
# ------------------------------------------------------------------
def main():
    print("🚀 POPI SPEECH ENGINE MODEL DOWNLOADER")
    print(f"Target Directory: {DEST_DIR}\n" + "=" * 60)

    for folder_name, info in MODELS.items():
        model_path = os.path.join(DEST_DIR, folder_name)
        archive_path = os.path.join(DEST_DIR, info["archive_name"])

        # Check if extracted model folder already exists
        if os.path.exists(model_path) and os.path.isdir(model_path):
            print(f"✔️ Model '{folder_name}' already exists. Skipping.")
            continue

        print(f"\n⏳ Downloading '{folder_name}'...")
        try:
            # Download file with progress hook
            urllib.request.urlretrieve(
                info["url"],
                archive_path,
                reporthook=download_progress_hook
            )
            
            # Extract downloaded file
            extract_archive(archive_path, DEST_DIR, info["extract_type"])

            # Clean up raw archive file to save disk space
            if os.path.exists(archive_path):
                os.remove(archive_path)
                print("   🧹 Cleaned up temporary archive file.")

        except Exception as e:
            print(f"\n❌ Failed to download {folder_name}: {e}")

    print("\n" + "=" * 60)
    print("🎉 All speech engine models are ready!")

if __name__ == "__main__":
    main()