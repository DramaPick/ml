import cv2
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import face_recognition

# 이미지 폴더 경로 (얼굴 이미지가 저장된 폴더)
input_folder = 'faces'  # 얼굴 이미지가 저장된 폴더
output_folder = 'clustered_faces_dbscan'  # 클러스터링 결과를 저장할 폴더

# 폴더가 없다면 생성
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 얼굴 이미지 파일 목록
image_files = os.listdir(input_folder)
image_paths = [os.path.join(input_folder, file) for file in image_files]

# 얼굴 특징 벡터 저장 리스트
features = []
filenames = []
image_sizes = []
face_landmarks = []

# 얼굴 특징 추출
for image_path in image_paths:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 얼굴 특징 추출
    encodings = face_recognition.face_encodings(image_rgb)

    # 얼굴 랜드마크 추출
    landmarks = face_recognition.face_landmarks(image_rgb)

    if len(encodings) > 0 and len(landmarks) > 0:
        features.append(encodings[0])
        filenames.append(image_path)
        # 해상도: width * height
        image_sizes.append(image.shape[0] * image.shape[1])
        face_landmarks.append(landmarks[0])  # 첫 번째 얼굴 랜드마크
    else:
        print(f"얼굴을 찾을 수 없음: {image_path}")

# 특성 벡터가 없다면 종료
if len(features) == 0:
    print("얼굴 인식된 이미지가 없습니다. 클러스터링을 수행할 수 없습니다.")
    exit()

# cosine similarity를 사용하여 거리 계산 (DBSCAN은 거리 기반 알고리즘이므로)
cosine_distances = cosine_similarity(features)

# DBSCAN을 사용하여 클러스터링 수행
db = DBSCAN(eps=0.6, min_samples=2, metric='cosine')
db.fit(cosine_distances)

# 각 클러스터에 속한 이미지들 출력 및 대표 이미지 선정
for cluster_id in np.unique(db.labels_):
    print(f"Cluster {cluster_id}:")

    # DBSCAN에서 -1은 노이즈로 처리된 데이터를 의미 (클러스터에 속하지 않은 데이터)
    if cluster_id == -1:
        print("  Noisy Data")
        continue

    # 해당 클러스터에 속한 이미지들의 인덱스를 가져옴
    cluster_indices = np.where(db.labels_ == cluster_id)[0]

    # 클러스터 내 대표 이미지 선택
    representative_image = None
    best_score = float('inf')  # 가장 정면에 가까운 이미지 찾기 위한 초기화
    best_resolution = 0  # 가장 높은 해상도를 위한 초기화

    # 클러스터 내 이미지들 간의 평균 유사도를 계산하여 가장 유사한 이미지를 대표 이미지로 선택
    for idx in cluster_indices:
        # 얼굴 랜드마크로 정면에 가까운 이미지를 찾기
        landmarks = face_landmarks[idx]
        # 눈과 입의 기울기를 통해 얼굴 각도를 계산 (기울기가 작은 값이 정면에 가까움)
        eye_distance = np.linalg.norm(
            np.array(landmarks['left_eye'][0]) - np.array(landmarks['right_eye'][3]))
        mouth_distance = np.linalg.norm(
            np.array(landmarks['top_lip'][0]) - np.array(landmarks['bottom_lip'][3]))
        angle_score = eye_distance + mouth_distance  # 이 두 거리의 합이 작은 것이 정면

        # 이미지 해상도 비교 (크기가 큰 것이 해상도가 높은 이미지)
        resolution_score = image_sizes[idx]

        # 정면도와 해상도를 동시에 고려하여 대표 이미지를 선정
        if angle_score < best_score or (angle_score == best_score and resolution_score > best_resolution):
            best_score = angle_score
            best_resolution = resolution_score
            representative_image = filenames[idx]

    # 대표 이미지 출력
    print(
        f"Representative Image for Cluster {cluster_id}: {representative_image}")

    # 대표 이미지를 output 폴더에 저장 (복사)
    if representative_image:
        output_image_path = os.path.join(
            output_folder, f"cluster_{cluster_id}_representative.jpg")
        img = cv2.imread(representative_image)
        cv2.imwrite(output_image_path, img)

print("클러스터링이 완료되었습니다.")
