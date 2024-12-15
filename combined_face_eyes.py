import numpy as np
import cv2
import dlib
import os
from imutils import face_utils
from numpy.linalg import norm

# dlib의 얼굴 감지기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    'shape_predictor_68_face_landmarks.dat')  # 모델 파일 경로

def eye_aspect_ratio(eye_points):
    # 수직 거리 (눈 위와 아래)
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # 수평 거리 (눈 양 옆)
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)  # EAR 계산
    return ear

def mouth_aspect_ratio(landmarks):
    # 입꼬리의 두 점 (48번, 54번)
    A = np.linalg.norm(np.array([landmarks.part(48).x, landmarks.part(48).y]) -
                       np.array([landmarks.part(54).x, landmarks.part(54).y]))
    return A

# 화질 너무 좋지 않는 이미지 감지 함수
def getBlurScore(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

# 얼굴에서 눈 감기 및 웃음 감지
def detect_smiling_and_eye(frame, faces, frame_number):
    for i, face in enumerate(faces):
        # 얼굴 영역 좌표
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
        # 얼굴 크기 계산
        face_width = x2 - x1
        face_height = y2 - y1

        # 얼굴 크기가 최소 크기보다 작은 경우 제외
        if face_width < 500 or face_height < 500:
            continue  # 크기가 작으면 해당 얼굴은 건너뜁니다.

        # 얼굴 영역에 여백 추가 (선택 사항)
        margin_x = int(face_width * 0.1)
        margin_y = int(face_height * 0.1)

        # 여백을 추가한 새로운 얼굴 영역 계산
        new_x1 = max(x1 - margin_x, 0)
        new_y1 = max(y1 - margin_y, 0)
        new_x2 = min(x2 + margin_x, frame_width)
        new_y2 = min(y2 + margin_y, frame_height)

        face_image_ = frame[new_y1:new_y2, new_x1:new_x2]
        if getBlurScore(face_image_) < 5:
            print(f"Frame {frame_number}: Blurry face, skipping.")
            continue  # 흐릿한 얼굴 건너뛰기

        # 얼굴을 회색조로 변환
        gray_face_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 랜드마크 예측
        # 원본 이미지에서 얼굴 영역을 기준으로 랜드마크 예측
        landmarks = predictor(gray_face_image, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # 왼쪽, 오른쪽 눈의 좌표
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        # EAR 계산
        left_ear = eye_aspect_ratio(np.array(left_eye))
        right_ear = eye_aspect_ratio(np.array(right_eye))
        ear = (left_ear + right_ear) / 2.0  # 두 눈의 EAR을 평균내기

        # 입 크기 감지 (웃음 여부 판단)
        lip_jaw_ratio = norm(
            landmarks[54] - landmarks[48]) / norm(landmarks[2] - landmarks[14])
        mouth_opening = norm(landmarks[57] - landmarks[51])
        mouth_nose = norm(landmarks[33] - landmarks[51])

        # 웃음 여부 판단: 입 크기, 입 벌어짐 등을 고려
        is_smiling = lip_jaw_ratio > 0.44 and mouth_opening / mouth_nose >= 1.05

        # EAR 출력
        print(f"Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Average EAR: {ear:.2f}")

        # EAR 값에 따른 눈 감김 여부 판단
        if ear < 0.2:  # 눈을 감고 있는 경우
            if is_smiling:
                print(f"웃고 있지만 눈을 감고 있음 + Frame {frame_number}")
                filename = os.path.join(output_folder, f"frame_{frame_number}_face_{i + 1}_smiling_and_eye_closed.jpg")
                cv2.imwrite(filename, frame[new_y1:new_y2, new_x1:new_x2])
            else:
                print(f"눈을 감고 있음 + Frame {frame_number}")
        else:
            print(f"눈을 뜨고 있음 + Frame {frame_number}")
            filename = os.path.join(output_folder, f"frame_{frame_number}_face_{i + 1}.jpg")
            cv2.imwrite(filename, frame[new_y1:new_y2, new_x1:new_x2])

# 비디오 파일 경로
video_path = './video_for_sim.mov'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# FPS 정보 가져오기
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS: {fps}")

# 비디오의 프레임 크기 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 얼굴 이미지를 저장할 폴더 생성
output_folder = 'faces'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 프레임 간격 설정
frame_interval = int(fps) - 1  # 1초마다 프레임 검사
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_number += 1

    # 프레임 간격마다 얼굴을 검사
    if frame_number % frame_interval == 0:
        # 얼굴 감지
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # 각 얼굴 이미지를 잘라서 저장 및 스마일/눈 감김 감지
        detect_smiling_and_eye(frame, faces, frame_number)

    # q 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 객체 해제 및 종료
cap.release()
cv2.destroyAllWindows()
