import scipy.io
import numpy as np
import os

RAW_PATH = "data/raw/NLS.mat" # 원본 데이터 경로 
PROCESSED_DIR = "data/processed/" # 전처리 결과물 저장 폴더 

def process_nls_data():
    os.makedirs(PROCESSED_DIR, exist_ok=True) # 폴더 없으면 생성 

    data = scipy.io.loadmat(RAW_PATH) # .mat 포멧 데이터를 dict 구조로 로드
    # data는 dict 형태로 key값은 'tt', 'uu', 'x' 가 존재 
    t = data['tt'].flatten()[:, np.newaxis] # 시간 데이터, t=[0, pi/2]
    x = data['x'].flatten()[:, np.newaxis] # 공간 데이터, x=[-5, 5]
    exact = data['uu']
    exact_u = np.real(exact) # real part
    exact_v = np.imag(exact) # imag part
    exact_h = np.sqrt(exact_u**2 + exact_v**2) # |h|

    # print("t shape: ", t.shape)
    # print("x shape: ", x.shape)
    # print("u shape: ", exact_u.shape)
    # print("v shape: ", exact_v.shape)

    '''
        t shape:  (201, 1)
        x shape:  (256, 1)
        u shape:  (256, 201)
        v shape:  (256, 201)
    '''

    # 여러개의 array를 비압축형태의 npz에 저장 
    np.savez(
        os.path.join(PROCESSED_DIR, 'nls_processed.npz'),
        x=x,
        t=t,
        u=exact_u,
        v=exact_v,
        h=exact_h,
    )

if __name__ == "__main__":
    process_nls_data()