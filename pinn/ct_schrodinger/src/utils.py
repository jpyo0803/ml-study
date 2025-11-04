import torch
import numpy as np
import random
import os

def set_seed(seed):
    '''
        - 난수 재현성을 위해 python/random, numpy, torch 모두 고정
        - CUDA 멀티 GPU 사용 시 manual_seed_all 필요
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    '''
        - 출력 디렉토리(체크포인트/로그/결과) 미리 생성
    '''
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)