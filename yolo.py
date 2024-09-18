import torch  # PyTorch 패키지
import cv2    # OpenCV 패키지

# YOLOv5 모델 로드
# yolov5s는 YOLOv5의 경량 버전으로, CPU에서도 비교적 빠르게 동작한다
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
# model.conf는 객체 탐지의 신뢰도 임계값을 설정한다. 이 값 이상일 때만 탐지 결과를 표시한다.
model.conf = 0.4

# 웹캠 초기화
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()  # 실시간으로 프레임을 읽어온다
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 이미지 전처리
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # frame을 RGB로 변환한다.

    # 객체 탐지
    results = model(img)  # RGB로 변환된 frame을 yolov5s 모델에 입력한다.

    # 결과를 원래 이미지에 그리기
    results.render()  # 객체를 탐지한다.
    annotated_frame = results.ims[0]  # 객체 탐지 결과를 frame에 그린다.

    # RGB를 BGR로 변환
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # frame 출력
    cv2.imshow('Real-Time Video Object Detection', annotated_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
