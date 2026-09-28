# CLAUDE.md (desktop_version: 데스크톱 앱)

## 개요
내 컴퓨터에서 실행하는 tkinter 그림판 앱입니다. 마우스로 숫자를 쓰고 마우스를 떼면 자동으로 인식합니다.

## 파일
| 파일 | 역할 |
|---|---|
| `model.py` | 모델 정의 (`숫자인식망`) |
| `train.py` | MNIST 학습 후 `mnist_cnn.pt` 저장 |
| `app.py` | 그림판 앱 (결과 숫자, 확신도, 숫자별 확률, 모델 입력 미리보기) |
| `mnist_cnn.pt` | 학습된 가중치 (`state_dict`) |

## 실행
```bash
py app.py         # 윈도우
python3 app.py    # 맥
```

## 주의사항
- 창(tkinter)을 띄우므로 Colab이나 서버에서는 실행되지 않습니다.
- `app.py`와 `model.py`, `mnist_cnn.pt`는 같은 폴더에 있어야 합니다.
- 가중치는 `state_dict`만 저장/불러오기 합니다. 모델 전체를 저장하지 않습니다.
- 전처리(글씨 영역 잘라내기 → 긴 변 20픽셀 → 28×28 가운데 배치)를 학습 때 값과 다르게 바꾸지 않습니다.
