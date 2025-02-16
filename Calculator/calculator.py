import cv2

# 라즈베리파이 카메라 모듈 사용
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 실시간 영상 처리 (예: Grayscale 변환)
    # processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed_frame = frame

    # 화면에 결과 표시
    cv2.imshow("Camera Stream", processed_frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()