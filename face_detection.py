import cv2
import dlib
import os

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

# 얼굴 감지 모델 로드
detector = dlib.get_frontal_face_detector()
# 얼굴 랜드마크 예측 모델 로드
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # 모델 파일 필요

# 얼굴 이미지를 저장할 폴더 생성
output_folder = 'output_faces'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 화질 너무 좋지 않는 이미지 감지 함수 
def getBlurScore(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

# 얼굴 추출 및 저장 함수
def save_faces(frame, faces, frame_number):
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

        # 얼굴 이미지 잘라내기
        face_image = frame[new_y1:new_y2, new_x1:new_x2]

        if getBlurScore(face_image) < 5:
            print(f"Frame {frame_number}: Blurry face, skipping.")
            continue # 흐릿한 얼굴 건너뛰기
        
        filename = os.path.join(output_folder, f"frame_{frame_number}_face_{i + 1}.jpg")
        cv2.imwrite(filename, face_image)

# 프레임 간격 설정
frame_interval = int(fps)-1  # int(fps)-1 -> 1초마다 프레임 검사
frame_number = 0

print(f"frame interval: {frame_interval}")

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

        # 각 얼굴 이미지를 잘라서 저장
        save_faces(frame, faces, frame_number)

    # 화면에 출력 (선택적)
    # cv2.imshow('Frame', frame)

    # q 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 비디오 객체 해제 및 종료
cap.release()
cv2.destroyAllWindows()