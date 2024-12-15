import numpy as np
import os
import cv2
import dlib
import face_recognition
from imutils import face_utils
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# dlib의 얼굴 감지기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 모델 파일 경로

# 얼굴 이미지가 저장될 폴더 생성
input_folder = 'faces'
output_folder = 'clustered_faces_final'
if not os.path.exists(input_folder):
    os.makedirs(input_folder)

# 얼굴에서 눈 감기 비율 계산
def eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)  # EAR 계산
    return ear

# 화질 너무 좋지 않는 이미지 감지 함수
def getBlurScore(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

# 얼굴에서 눈 감기 및 웃음 감지
def detect_smiling_and_eye(frame, faces, frame_number):
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = (face.left(), face.top(), face.right(), face.bottom())
        face_width = x2 - x1
        face_height = y2 - y1
        if face_width < 500 or face_height < 500:
            continue

        margin_x = int(face_width * 0.1)
        margin_y = int(face_height * 0.1)
        new_x1 = max(x1 - margin_x, 0)
        new_y1 = max(y1 - margin_y, 0)
        new_x2 = min(x2 + margin_x, frame_width)
        new_y2 = min(y2 + margin_y, frame_height)

        face_image_ = frame[new_y1:new_y2, new_x1:new_x2]
        if getBlurScore(face_image_) < 5:
            continue  # 흐릿한 얼굴 건너뛰기

        gray_face_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        landmarks = predictor(gray_face_image, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_ear = eye_aspect_ratio(np.array(left_eye))
        right_ear = eye_aspect_ratio(np.array(right_eye))
        ear = (left_ear + right_ear) / 2.0

        lip_jaw_ratio = norm(landmarks[54] - landmarks[48]) / norm(landmarks[2] - landmarks[14])
        mouth_opening = norm(landmarks[57] - landmarks[51])
        mouth_nose = norm(landmarks[33] - landmarks[51])

        is_smiling = lip_jaw_ratio > 0.44 and mouth_opening / mouth_nose >= 1.05

        if ear < 0.2:
            if is_smiling:
                filename = os.path.join(input_folder, f"frame_{frame_number}_face_{i + 1}_smiling_and_eye_closed.jpg")
                cv2.imwrite(filename, frame[new_y1:new_y2, new_x1:new_x2])
            else:
                continue
        else:
            filename = os.path.join(input_folder, f"frame_{frame_number}_face_{i + 1}.jpg")
            cv2.imwrite(filename, frame[new_y1:new_y2, new_x1:new_x2])

# 비디오 파일 경로
video_path = './video_for_sim.mov'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_interval = int(fps) - 1
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    if frame_number % frame_interval == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        detect_smiling_and_eye(frame, faces, frame_number)

cap.release()

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 이미지 파일 목록
image_files = os.listdir(input_folder)
image_paths = [os.path.join(input_folder, file) for file in image_files]

# 얼굴 특징 벡터 저장 리스트
features = []
filenames = []

# 얼굴 특징 추출
for image_path in image_paths:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(image_rgb)

    if len(encodings) > 0:
        features.append(encodings[0])
        filenames.append(image_path)
    else:
        print(f"얼굴을 찾을 수 없음: {image_path}")

if len(features) == 0:
    print("얼굴 인식된 이미지가 없습니다. 클러스터링을 수행할 수 없습니다.")
    exit()

# 코사인 유사도를 계산하여 군집화
cosine_distances = cosine_similarity(features)
clustering = AgglomerativeClustering(
    n_clusters=3, metric='cosine', linkage='average')
clustering.fit(features)

# 얼굴 각도 계산 함수
def calculate_face_angle(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_landmarks_list = face_recognition.face_landmarks(image_rgb)

    if len(face_landmarks_list) > 0:
        face_landmarks = face_landmarks_list[0]
        left_eye = face_landmarks['left_eye']
        right_eye = face_landmarks['right_eye']
        nose = face_landmarks['nose_bridge']

        eye_distance = np.linalg.norm(
            np.array(left_eye[0]) - np.array(right_eye[3]))
        nose_to_eyes_distance = np.linalg.norm(
            np.array(nose[0]) - np.array(left_eye[0]))
        angle = np.arctan(nose_to_eyes_distance / eye_distance)
        angle_deg = np.degrees(angle)
        return angle_deg
    else:
        return float('inf')


# 각 클러스터에 대해 대표 이미지 선정
for cluster_id in np.unique(clustering.labels_):
    print(f"Cluster {cluster_id}:")
    cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
    representative_image = None
    min_angle = float('inf')

    for idx in cluster_indices:
        angle = calculate_face_angle(filenames[idx])
        if angle < min_angle:
            min_angle = angle
            representative_image = filenames[idx]

    print(f"Representative Image for Cluster {cluster_id}: {representative_image}")

    if representative_image:
        output_image_path = os.path.join(output_folder, f"cluster_{cluster_id}_representative.jpg")
        img = cv2.imread(representative_image)
        cv2.imwrite(output_image_path, img)

print("클러스터링이 완료되었습니다.")