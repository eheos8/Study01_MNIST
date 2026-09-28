"""
손글씨 숫자 인식기 (MNIST CNN) - 그림판 앱

실행 방법 (mnist_cnn.pt, model.py 와 같은 폴더에서):
    윈도우 : py app.py
    맥     : python3 app.py
    (이미지 파일로 인식하려면 뒤에 파일 이름을 붙임: py app.py 내숫자.png)

주의: 창(tkinter)을 띄우므로 Colab이 아닌 내 컴퓨터에서 실행해야 합니다.
"""

import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps

from model import 숫자인식망

가중치_파일 = "mnist_cnn.pt"
장치 = torch.device("cpu")  # 인식(추론)은 CPU로도 충분히 빠름


# ---------------------------------------------------------------
# 1. 모델 불러오기
# ---------------------------------------------------------------
def 모델_불러오기():
    모델 = 숫자인식망()
    모델.load_state_dict(torch.load(가중치_파일, map_location=장치))
    모델.eval()  # 추론 모드 (드롭아웃 끔)
    print(f"모델을 불러왔습니다 ({가중치_파일}, 장치: {장치})")
    return 모델


# ---------------------------------------------------------------
# 2. 전처리: 어떤 이미지든 MNIST 형식(28x28, 검은 배경에 흰 글씨)으로 맞춤
# ---------------------------------------------------------------
def 전처리_이미지(그림: Image.Image) -> Image.Image:
    """글씨 영역만 잘라 긴 변을 20픽셀로 줄이고 28x28 가운데에 배치한 PIL 이미지를 반환"""
    그림 = 그림.convert("L")

    # 배경이 밝으면(흰 종이에 검은 글씨) 색을 반전
    if np.array(그림).mean() > 127:
        그림 = ImageOps.invert(그림)

    배열 = np.array(그림)
    글씨위치 = np.argwhere(배열 > 50)
    if len(글씨위치) == 0:
        raise ValueError("이미지에서 글씨를 찾지 못했습니다.")
    (위, 왼쪽), (아래, 오른쪽) = 글씨위치.min(axis=0), 글씨위치.max(axis=0) + 1
    그림 = Image.fromarray(배열[위:아래, 왼쪽:오른쪽])

    비율 = 20.0 / max(그림.size)
    새크기 = (max(1, round(그림.width * 비율)), max(1, round(그림.height * 비율)))
    그림 = 그림.resize(새크기, Image.LANCZOS)

    캔버스 = Image.new("L", (28, 28), 0)
    캔버스.paste(그림, ((28 - 새크기[0]) // 2, (28 - 새크기[1]) // 2))
    return 캔버스


def 텐서로_변환(작은그림: Image.Image) -> torch.Tensor:
    """28x28 이미지를 학습 때와 같은 방식으로 정규화해 (1, 1, 28, 28) 텐서로 변환"""
    텐서 = torch.from_numpy(np.array(작은그림, dtype=np.float32) / 255.0)
    return ((텐서 - 0.1307) / 0.3081).unsqueeze(0).unsqueeze(0)


def 예측(모델, 그림: Image.Image):
    """(예측한 숫자, 확신도(%), 숫자별 확률, 모델에 들어간 28x28 이미지) 반환"""
    작은그림 = 전처리_이미지(그림)
    with torch.no_grad():
        확률 = F.softmax(모델(텐서로_변환(작은그림)), dim=1)[0]
    숫자 = int(확률.argmax())
    return 숫자, float(확률[숫자]) * 100, 확률, 작은그림


# ---------------------------------------------------------------
# 3. 그림판 앱 화면
# ---------------------------------------------------------------
def 그림판_실행(모델):
    import tkinter as tk
    from PIL import ImageTk

    캔버스크기, 붓굵기 = 280, 18
    막대너비, 막대높이 = 136, 12
    글꼴 = "맑은 고딕"

    창 = tk.Tk()
    창.title("손글씨 숫자 인식기 (MNIST CNN)")
    창.resizable(False, False)

    # ---------- 왼쪽: 그리는 영역 ----------
    왼쪽 = tk.Frame(창)
    왼쪽.grid(row=0, column=0, padx=14, pady=12, sticky="n")
    tk.Label(왼쪽, text="여기에 숫자를 써 주세요 (0~9)", font=(글꼴, 11, "bold")).pack(anchor="w", pady=(0, 6))
    캔버스 = tk.Canvas(왼쪽, width=캔버스크기, height=캔버스크기, bg="black",
                    highlightthickness=2, highlightbackground="#9aa0a6", cursor="cross")
    캔버스.pack()

    # 화면에 그리는 것과 동시에 PIL 이미지에도 똑같이 그림 (모델 입력용)
    그림 = Image.new("L", (캔버스크기, 캔버스크기), 0)
    펜 = ImageDraw.Draw(그림)

    버튼줄 = tk.Frame(왼쪽)
    버튼줄.pack(fill="x", pady=(8, 0))
    tk.Label(왼쪽, text="마우스를 떼면 자동으로 인식합니다.", fg="#777777", font=(글꼴, 9)).pack(anchor="w", pady=(6, 0))

    # ---------- 오른쪽: 결과 영역 ----------
    오른쪽 = tk.Frame(창)
    오른쪽.grid(row=0, column=1, padx=(0, 14), pady=12, sticky="n")
    tk.Label(오른쪽, text="인식 결과", font=(글꼴, 11, "bold")).pack(anchor="w")
    결과숫자 = tk.StringVar(value="?")
    tk.Label(오른쪽, textvariable=결과숫자, font=(글꼴, 60, "bold"), width=3).pack()
    확신도글자 = tk.StringVar(value="확신도: -")
    tk.Label(오른쪽, textvariable=확신도글자, fg="#555555", font=(글꼴, 9)).pack()

    tk.Label(오른쪽, text="숫자별 확률", font=(글꼴, 9, "bold")).pack(anchor="w", pady=(10, 2))
    확률판 = tk.Frame(오른쪽, highlightthickness=1, highlightbackground="#cccccc")
    확률판.pack(fill="x")
    막대채움, 퍼센트글자 = [], []
    for i in range(10):
        줄 = tk.Frame(확률판)
        줄.pack(fill="x", padx=4, pady=1)
        tk.Label(줄, text=str(i), width=2, anchor="w", font=(글꼴, 8)).pack(side="left")
        막대 = tk.Canvas(줄, width=막대너비, height=막대높이, highlightthickness=0, bg="#eeeeee")
        막대.pack(side="left", padx=2)
        채움 = 막대.create_rectangle(0, 0, 0, 막대높이, fill="#2f6fed", width=0)
        막대채움.append((막대, 채움))
        퍼센트 = tk.StringVar(value="0.0%")
        tk.Label(줄, textvariable=퍼센트, width=6, anchor="e", font=(글꼴, 8)).pack(side="left")
        퍼센트글자.append(퍼센트)

    tk.Label(오른쪽, text="모델 입력 (28x28)", font=(글꼴, 9, "bold")).pack(anchor="w", pady=(10, 2))
    미리보기칸 = tk.Label(오른쪽, bg="black", width=20, height=10)  # 그림이 들어오면 크기가 바뀜
    미리보기칸.pack()
    미리보기참조 = {}  # 그림 객체가 사라지지 않게 붙잡아 두는 용도

    # ---------- 동작 ----------
    이전점 = {"x": None, "y": None}

    def 그리기(이벤트):
        if 이전점["x"] is not None:
            캔버스.create_line(이전점["x"], 이전점["y"], 이벤트.x, 이벤트.y,
                            fill="white", width=붓굵기, capstyle=tk.ROUND, smooth=True)
            펜.line([이전점["x"], 이전점["y"], 이벤트.x, 이벤트.y], fill=255, width=붓굵기)
        이전점["x"], 이전점["y"] = 이벤트.x, 이벤트.y

    def 결과_지우기():
        결과숫자.set("?")
        확신도글자.set("확신도: -")
        for (막대, 채움), 퍼센트 in zip(막대채움, 퍼센트글자):
            막대.coords(채움, 0, 0, 0, 막대높이)
            퍼센트.set("0.0%")
        미리보기칸.config(image="", width=20, height=10)
        미리보기참조.clear()

    def 인식():
        try:
            숫자, 확신도, 확률, 작은그림 = 예측(모델, 그림)
        except ValueError:
            확신도글자.set("먼저 숫자를 써 주세요.")
            return
        결과숫자.set(str(숫자))
        확신도글자.set(f"확신도: {확신도:.1f}%")
        for i, ((막대, 채움), 퍼센트) in enumerate(zip(막대채움, 퍼센트글자)):
            p = float(확률[i])
            막대.coords(채움, 0, 0, 막대너비 * p, 막대높이)
            퍼센트.set(f"{p * 100:.1f}%")
        # 모델이 실제로 본 28x28 이미지를 5배 확대해서 보여줌
        확대 = ImageTk.PhotoImage(작은그림.resize((140, 140), Image.NEAREST))
        미리보기참조["이미지"] = 확대
        미리보기칸.config(image=확대, width=140, height=140)

    def 손떼기(이벤트):
        이전점["x"], 이전점["y"] = None, None
        인식()  # 마우스를 떼면 자동으로 인식

    def 지우기():
        캔버스.delete("all")
        펜.rectangle([0, 0, 캔버스크기, 캔버스크기], fill=0)
        결과_지우기()

    캔버스.bind("<B1-Motion>", 그리기)
    캔버스.bind("<ButtonRelease-1>", 손떼기)
    tk.Button(버튼줄, text="인식하기", width=10, command=인식).pack(side="left")
    tk.Button(버튼줄, text="지우기", width=10, command=지우기).pack(side="right")

    창.mainloop()


# ---------------------------------------------------------------
# 4. 메인
# ---------------------------------------------------------------
def main():
    try:
        모델 = 모델_불러오기()
    except FileNotFoundError:
        print(f"'{가중치_파일}' 파일이 없습니다. 먼저 train.py 를 실행하거나 가중치 파일을 이 폴더에 두세요.")
        return

    if len(sys.argv) > 1:
        # 이미지 파일 인식 모드
        숫자, 확신도, _, _ = 예측(모델, Image.open(sys.argv[1]))
        print(f"예측한 숫자: {숫자}  (확신도 {확신도:.1f}%)")
    else:
        그림판_실행(모델)


if __name__ == "__main__":
    main()
