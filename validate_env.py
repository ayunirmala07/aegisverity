import importlib

modules = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "opencv-python": "cv2",
    "mediapipe": "mediapipe",
    "librosa": "librosa",
    "grad-cam": "pytorch_grad_cam",
    "dlib-bin": "dlib",
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas"
}

print("=== Environment Validation ===")
for name, module in modules.items():
    try:
        m = importlib.import_module(module)
        version = getattr(m, "__version__", "No __version__ attr")
        print(f"{name}: {version}")
    except Exception as e:
        print(f"{name}: ERROR -> {e}")
