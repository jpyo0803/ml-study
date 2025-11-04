import scipy.io
import os
import torch

RAW_PATH = "data/raw/NLS.mat" # 원본 데이터 경로 
PROCESSED_DIR = "data/processed/" # 전처리 결과물 저장 폴더 

def process_nls_data():

    '''
        .mat 형식의 Raw data를 로딩
        로딩된 객체는 dict 데이터 구조

        keys = ['tt', 'uu', 'x'] 
    '''
    data = scipy.io.loadmat(RAW_PATH)

    t = torch.tensor(data['tt'], dtype=torch.float32).flatten().unsqueeze(1) # (201, 1)
    x = torch.tensor(data['x'], dtype=torch.float32).flatten().unsqueeze(1) # (256, 1)

    exact = torch.tensor(data['uu'], dtype=torch.complex128) # (256, 201)
    exact_u = exact.real.to(torch.float32) # (256, 201)
    exact_v = exact.imag.to(torch.float32) # (256, 201)
    exact_h = torch.sqrt(exact_u**2 + exact_v**2) # (256, 201)


    os.makedirs(PROCESSED_DIR, exist_ok=True) # 저장 위치 폴더 없으면 생성

    # 지정된 경로에 .pt 포멧으로 텐서 저장 
    torch.save(
        {
            'x': x,
            't': t,
            'u': exact_u,
            'v': exact_v,
            'h': exact_h,
        },
        os.path.join(PROCESSED_DIR, 'nls_processed.pt')
    ) 


if __name__ == "__main__":
    process_nls_data()