"""
학습 스크립트
실행: python train.py
결과: 학습이 끝나면 가중치 파일 'mnist_cnn.pt' 가 저장됩니다.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import 숫자인식망

# ---------------------------------------------------------------
# 설정값
# ---------------------------------------------------------------
배치크기 = 64
에폭수 = 5
학습률 = 0.001
가중치_파일 = "mnist_cnn.pt"

# GPU(CUDA) 또는 애플 실리콘(MPS)이 있으면 사용하고, 없으면 CPU 사용
if torch.cuda.is_available():
    장치 = torch.device("cuda")
elif torch.backends.mps.is_available():
    장치 = torch.device("mps")
else:
    장치 = torch.device("cpu")


def 학습(모델, 로더, 최적화기, 에폭):
    모델.train()
    for 번호, (이미지, 정답) in enumerate(로더):
        이미지, 정답 = 이미지.to(장치), 정답.to(장치)
        최적화기.zero_grad()
        손실 = F.cross_entropy(모델(이미지), 정답)
        손실.backward()
        최적화기.step()
        if 번호 % 200 == 0:
            print(f"[에폭 {에폭}] {번호 * len(이미지)}/{len(로더.dataset)} 손실: {손실.item():.4f}")


def 평가(모델, 로더):
    모델.eval()
    맞은개수 = 0
    with torch.no_grad():
        for 이미지, 정답 in 로더:
            이미지, 정답 = 이미지.to(장치), 정답.to(장치)
            맞은개수 += (모델(이미지).argmax(dim=1) == 정답).sum().item()
    print(f"테스트 정확도: {100.0 * 맞은개수 / len(로더.dataset):.2f}% ({맞은개수}/{len(로더.dataset)})")


def main():
    print(f"사용 장치: {장치}")

    # MNIST 표준 정규화 값(평균 0.1307, 표준편차 0.3081)
    변환 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # 처음 실행하면 ./data 폴더에 MNIST 데이터가 자동으로 다운로드됩니다.
    학습셋 = datasets.MNIST("./data", train=True, download=True, transform=변환)
    테스트셋 = datasets.MNIST("./data", train=False, download=True, transform=변환)
    학습로더 = DataLoader(학습셋, batch_size=배치크기, shuffle=True)
    테스트로더 = DataLoader(테스트셋, batch_size=1000)

    모델 = 숫자인식망().to(장치)
    최적화기 = torch.optim.Adam(모델.parameters(), lr=학습률)

    for 에폭 in range(1, 에폭수 + 1):
        학습(모델, 학습로더, 최적화기, 에폭)
        평가(모델, 테스트로더)

    torch.save(모델.state_dict(), 가중치_파일)
    print(f"학습된 가중치를 '{가중치_파일}' 파일로 저장했습니다.")


if __name__ == "__main__":
    main()
