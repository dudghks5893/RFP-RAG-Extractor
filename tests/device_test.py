import torch
from src.utils import get_device
'''
    테스트 방법:
    터미널에 python -m tests.device_test 입력
'''

# 디바이스 설정
device = get_device()

# 디바이스 확인
x = torch.ones(5, device=device)
print(x)
print(f"Device: {device}")