import cv2
import dlib
import os

# dlib의 얼굴 감지기 및 랜드마크 예측기 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

output_folder = 'output_faces'
image_files = os.listdir(output_folder)

def getBlurScore(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

print(f"추출된 총 이미지 개수 : {len(image_files)}개")
# 이미지 파일이 있으면 첫 번째 이미지 파일을 가져옵니다.
image_list = []
if len(image_files) > 0:
    for i in range(1):
        image_path = os.path.join(output_folder, image_files[i])
        print(f"image_path : {image_path}")
        image = cv2.imread(image_path)

        if image is not None:
            pass

print(f"Total images with eyes open: {len(image_list)}")