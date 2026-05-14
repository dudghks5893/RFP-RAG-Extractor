# test sooq
import torch
import langchain
import pymupdf
import ragas
import pandas as pd

print(f"PyTorch 버전: {torch.__version__}")
print(f"GPU/MPS 가속 가능 여부: {torch.cuda.is_available() or torch.backends.mps.is_available()}")
print("✅ 모든 핵심 라이브러리가 정상적으로 로드되었습니다!")