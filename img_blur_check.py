import cv2
import argparse

def getBlurScore(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()

def main():
    # Argument parser 설정
    parser = argparse.ArgumentParser(
        description="Laplacian variance to measure image sharpness.")
    parser.add_argument("image_path", help="Path to the image file.")

    # 인자 파싱
    args = parser.parse_args()

    # 이미지 파일 읽기
    image = cv2.imread(args.image_path)

    if image is None:
        print("Error: The image file could not be loaded. Please check the file path.")
        return

    # 라플라시안 값 계산
    blur_score = getBlurScore(image)
    print(f"Laplacian Variance (Sharpness): {blur_score}")


if __name__ == "__main__":
    main()