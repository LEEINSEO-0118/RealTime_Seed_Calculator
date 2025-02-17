import cv2
import numpy as np

#### Functions
def load_image(path, scale = 0.7):
    img = cv2.imread(path)
    img_resized = cv2.resize(img, (0,0), None, scale, scale)
    return img_resized

def preprocess_image(img, thresh_1=57, thresh_2=232):
    img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # 그레이 스케일 변환
    img_blur  = cv2.GaussianBlur(img_gray, (5,5), 1)       # 필터처리
    img_canny = cv2.Canny(img_blur, thresh_1, thresh_2)    # edge detector

    kernel = np.ones((3,3))
    img_dilated = cv2.dilate(img_canny, kernel, iterations=1)    # edge 라인을 굵게
    img_closed = cv2.morphologyEx(img_dilated, cv2.MORPH_CLOSE,
                                  kernel, iterations=6)          # edge 내부 small holes 제거, edge를 명확하게

    img_preprocessed = img_closed.copy()
    return img_preprocessed

def find_contours(img_preprocessed, img_original, epsilon_param=0.04):
    contours, hierarchy = cv2.findContours(image=img_preprocessed,
                                           mode=cv2.RETR_TREE, # for outermost contours
                                           method=cv2.CHAIN_APPROX_NONE)  # find contour
    if hierarchy is None or len(contours) == 0:
        return img_original  # Contour가 없으면 원본 이미지 반환

    # 컨투어 넓이 기준으로 내림차순 정렬, 상위 4개의 컨투어만 사용
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    filtered_contours = sorted_contours[:4]
    polygons = []
    epsilon = epsilon_param * cv2.arcLength(curve=filtered_contours[0],
                                            closed=True)  # epsilon 값이 작으면 많은 코너를 탐지한다.
                                                            # 사이즈 기준 객체의 정확한 코너를 찾기 위해서 epsilon 값을 설정하는 것에 주의해야 함
    polygon = cv2.approxPolyDP(curve=filtered_contours[0],
                                epsilon=epsilon, closed=True)  # contour를 통해 코너(Curves) 추정
    polygon = polygon.reshape(4, 2)  # original output of cv2.approxPolyDP() is in the format (number of corners, 1, 2), where the middle axis is irrelevant in our case. Hence, we can safely discard it.
    polygons.append(polygon)
    # Seed Box
    seed_contour = filtered_contours[2]
    seed_rect = cv2.minAreaRect(seed_contour) # Seed에 핏한 회전된 바운딩 박스
    seed_box = cv2.boxPoints(seed_rect).astype(int)
    return polygons, seed_box

def reorder_coords(polygon):
    rect_coords = np.zeros((4, 2)) # 각 모서리 구해서 넣기
    add = polygon.sum(axis=1) # 컬럼을 기준으로 합치기
    rect_coords[0] = polygon[np.argmin(add)]    # Top left # 최소값의 index 반환
    rect_coords[3] = polygon[np.argmax(add)]    # Bottom right
    subtract = np.diff(polygon, axis=1)
    rect_coords[1] = polygon[np.argmin(subtract)]    # Top right
    rect_coords[2] = polygon[np.argmax(subtract)]    # Bottom left
    return rect_coords

def calculate_sizes(rect_coords, seed_box):
    seed_left_top = seed_box[0]
    seed_left_bottom = seed_box[3]
    seed_right_bottom = seed_box[2]

    rect_left_top = rect_coords[0]
    rect_left_bottom = rect_coords[2]

    rect_pixel_distance = np.linalg.norm(rect_left_top - rect_left_bottom)
    actual_rect_length_mm = 30.0

    mm_per_pixel =  actual_rect_length_mm / rect_pixel_distance

    seed_left_lenght_mm = np.linalg.norm(seed_left_top - seed_left_bottom) * mm_per_pixel
    seeed_bootom_length_mm = np.linalg.norm(seed_left_bottom - seed_right_bottom) * mm_per_pixel
    result = [seed_left_lenght_mm, seeed_bootom_length_mm]

    return result



def write_size(seed_box, sizes, img_warped):
    img_result = img_warped.copy()
    cv2.drawContours(img_result, [seed_box], -1, (0, 0, 255), 2)
    cv2.putText(img_result, f'{np.float32(sizes[0]):.2f}mm',
                (seed_box[0][0]-250, seed_box[0][1]+100),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,255), 2)
    cv2.putText(img_result, f'{np.float32(sizes[1]):.2f}mm',
                (seed_box[3][0]+50, seed_box[3][1]+50),
                cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,255), 2)
    return img_result


def measure_size(image, paper_eps_param=0.04,canny_thresh_1=57, canny_thresh_2=232):

    # Loading and preprocessing original image.
    img_original = image.copy()
    img_preprocessed = preprocess_image(img_original,
                                                       thresh_1=canny_thresh_1,
                                                       thresh_2=canny_thresh_2)

    # Finding contours and Return Template Corners & Seed Box
    polygons, seed_box = find_contours(img_preprocessed,
                                           img_original,
                                           epsilon_param=paper_eps_param)

    # Reordering Template corners.
    rect_coords = np.float32(reorder_coords(polygons[0]))

    # Edge langth calculation.
    seed_size = calculate_sizes(rect_coords, seed_box)
    img_result = write_size(seed_box, seed_size, img_original)
    return img_result
#### Functions

# 라즈베리파이 카메라 모듈 사용
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 실시간 영상 처리 (예: Grayscale 변환)
    # processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        processed_frame = measure_size(frame)
    except:
        processed_frame = frame

    # 화면에 결과 표시
    cv2.imshow("Camera Stream", processed_frame)

    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()