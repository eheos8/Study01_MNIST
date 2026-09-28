"""
모델 정의 파일
train.py(학습)와 app.py(그림판 앱)가 같은 모델 구조를 쓰도록 여기에 한 번만 정의합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class 숫자인식망(nn.Module):
    """손글씨 숫자(0~9)를 분류하는 간단한 CNN"""

    def __init__(self):
        super().__init__()
        # 합성곱 층: 이미지의 선, 곡선 같은 특징을 추출
        self.합성곱1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28 -> 28x28
        self.합성곱2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 14x14 -> 14x14
        # 드롭아웃: 과적합 방지
        self.드롭아웃 = nn.Dropout(0.25)
        # 완전연결 층: 추출된 특징으로 0~9 분류
        self.완전연결1 = nn.Linear(64 * 7 * 7, 128)
        self.완전연결2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.합성곱1(x)), 2)  # 28x28 -> 14x14
        x = F.max_pool2d(F.relu(self.합성곱2(x)), 2)  # 14x14 -> 7x7
        x = torch.flatten(x, 1)
        x = self.드롭아웃(F.relu(self.완전연결1(x)))
        return self.완전연결2(x)  # 각 숫자에 대한 점수(로짓)
