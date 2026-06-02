import torch

# 디바이스 (GPU or CPU or MPS) 가져오는 함수
def get_device():
    # CUDA (Windows / Linux GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA GPU 사용: {torch.cuda.get_device_name(0)}")

    # Apple Silicon GPU (Mac MPS)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple Silicon GPU (MPS) 사용")

    # CPU fallback
    else:
        device = torch.device("cpu")
        print("CPU 사용")

    return device