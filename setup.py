import os
import urllib.request
import bz2
import shutil
import sys
import time

def setup_aegis_environment():
    # --- Konfigurasi ---
    BASE_DIR = os.getcwd()
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    # URL resmi model dlib (68 landmarks)
    DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    DLIB_FILENAME_BZ2 = "shape_predictor_68_face_landmarks.dat.bz2"
    DLIB_FILENAME_DAT = "shape_predictor_68_face_landmarks.dat"

    print("🛡️  INITIALIZING AEGIS VERITY ENVIRONMENT SETUP 🛡️")
    print("====================================================")

    # 1. Buat Direktori Models
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        print(f"[+] Folder 'models/' created at: {MODELS_DIR}")
    else:
        print(f"[i] Folder 'models/' already exists.")

    # 2. Cek apakah model sudah ada
    target_path = os.path.join(MODELS_DIR, DLIB_FILENAME_DAT)
    temp_bz2_path = os.path.join(MODELS_DIR, DLIB_FILENAME_BZ2)

    if os.path.exists(target_path):
        print(f"[✓] Model '{DLIB_FILENAME_DAT}' detected. Skipping download.")
    else:
        print(f"[*] Model missing. Starting download for Layer 3 (AV Consistency)...")
        print(f"    Source: {DLIB_MODEL_URL}")
        print(f"    Target: {target_path}")
        print("    (Size: ~100MB. Please wait...)")

        try:
            # Fungsi untuk menampilkan progress bar sederhana
            def progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\r    Downloading... {percent}%")
                sys.stdout.flush()

            # Download File
            start_time = time.time()
            urllib.request.urlretrieve(DLIB_MODEL_URL, temp_bz2_path, reporthook=progress_hook)
            print(f"\n[+] Download complete in {round(time.time() - start_time, 2)} seconds.")

            # Ekstrak File
            print("[*] Extracting .bz2 archive...")
            with bz2.BZ2File(temp_bz2_path) as fr, open(target_path, "wb") as fw:
                shutil.copyfileobj(fr, fw)
            print(f"[+] Extraction complete.")

            # Bersihkan file sampah
            os.remove(temp_bz2_path)
            print("[+] Cleanup temporary files done.")

        except Exception as e:
            print(f"\n[X] CRITICAL ERROR: {str(e)}")
            print("    Tips: Check your internet connection or try downloading manually.")
            # Hapus file parsial jika gagal agar tidak corrupt
            if os.path.exists(temp_bz2_path):
                os.remove(temp_bz2_path)
            sys.exit(1)

    print("====================================================")
    print("✅ AEGIS VERITY IS READY.")
    print(f"   Run pipeline using: python main.py")

if __name__ == "__main__":
    setup_aegis_environment()