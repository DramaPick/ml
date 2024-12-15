import numpy as np
import cv2
import dlib
import os
from imutils import face_utils
from numpy.linalg import norm

# dlib의 얼굴 감지기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 해당 모델 파일 경로

# 눈의 수직 거리 계산 함수
def eye_aspect_ratio(eye_points):
    # 수직 거리 (눈 위와 아래)
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # 수평 거리 (눈 양 옆)
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)  # EAR 계산
    return ear

# 웃음 감지 함수 (입의 양 끝 거리 계산)
def mouth_aspect_ratio(landmarks):
    # 입꼬리의 두 점 (48번, 54번)
    A = np.linalg.norm(np.array([landmarks.part(48).x, landmarks.part(48).y]) -
                       np.array([landmarks.part(54).x, landmarks.part(54).y]))
    return A

output_folder = 'output_faces'
image_files = os.listdir(output_folder)
print(f"추출된 총 이미지 개수 : {len(image_files)}개")

total_img_list = []
if len(image_files) > 0:
    for i in range(len(image_files)):
        image_path = os.path.join(output_folder, image_files[i])
        image = cv2.imread(image_path)

        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                landmarks = face_utils.shape_to_np(landmarks)

                # 왼쪽, 오른쪽 눈의 좌표
                left_eye = landmarks[36:42]
                right_eye = landmarks[42:48]

                '''
                left_eye = []
                right_eye = []

                # 왼쪽 눈 (36~41)
                for i in range(36, 42):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    left_eye.append((x, y))

                # 오른쪽 눈 (42~47)
                for i in range(42, 48):
                    x = landmarks.part(i).x
                    y = landmarks.part(i).y
                    right_eye.append((x, y))'''

                # EAR 계산
                left_ear = eye_aspect_ratio(np.array(left_eye))
                right_ear = eye_aspect_ratio(np.array(right_eye))
                ear = (left_ear + right_ear) / 2.0  # 두 눈의 EAR을 평균내기

                # 입 크기 감지 (웃음 여부 판단)
                lip_jaw_ratio = norm(landmarks[54] - landmarks[48]) / norm(landmarks[2] - landmarks[14])
                mouth_opening = norm(landmarks[57] - landmarks[51])
                mouth_nose = norm(landmarks[33] - landmarks[51])

                # 웃음 여부 판단: 입 크기, 입 벌어짐 등을 고려
                is_smiling = lip_jaw_ratio > 0.44 and mouth_opening / mouth_nose >= 1.05

                # EAR 출력
                print(f"Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Average EAR: {ear:.2f}")

                # EAR 값에 따른 눈 감김 여부 판단
                if ear < 0.2:  # 이 값은 실험적으로 조정할 수 있습니다
                    if is_smiling:
                        print(f"웃고 있지만 눈을 감고 있음 + {image_path}")
                    else:
                        print(f"눈을 감고 있음 + {image_path}")
                else:
                    print("눈을 뜨고 있음")
                    total_img_list.append(image_path)

                # 랜드마크 점 그리기
                for (x, y) in left_eye + right_eye:
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # 초록색 점으로 표시

            # 결과 이미지 출력
            # cv2.imshow("Eye Aspect Ratio", image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

print(f"total image with opened eyes : {len(total_img_list)}")
