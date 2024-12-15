import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def fft_image_enhancement(image, output_path, radius=30):
    """
    이미지에 대해 FFT를 적용하여 고화질 이미지를 생성하고 저장하는 함수.
    (고주파 필터를 조정하여 이미지 품질 향상)
    """
    # 1. 이미지를 주파수 도메인으로 변환 (FFT)
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # DC 성분을 중앙으로 이동

    # 2. 주파수 스펙트럼 시각화 (로그 스케일)
    magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum = np.log(magnitude_spectrum + 1)  # 0 값 방지

    # 3. 고주파 강조 (필터링)
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # 이미지 중앙

    # 고주파 필터링 (저주파 성분을 제거)
    mask = np.ones((rows, cols), np.uint8)
    x, y = np.fft.fftfreq(cols), np.fft.fftfreq(rows)

    # 원형 마스크로 저주파 필터링
    for i in range(cols):
        for j in range(rows):
            dist = np.sqrt((i - ccol)**2 + (j - crow)**2)
            if dist < radius:
                mask[j, i] = 0  # 저주파 성분은 제거

    # 주파수 도메인에서 필터링을 적용
    fshift_filtered = fshift * mask

    # 4. 역FFT (Inverse FFT) - 공간 도메인으로 변환
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # 5. 이미지 스케일링 (어두운 이미지를 밝게 조정)
    img_back = np.uint8(np.clip(img_back, 0, 255))  # 값 범위 조정

    # 6. 고화질 이미지 저장
    cv2.imwrite(output_path, img_back)

    return img_back


def process_images_in_folder(input_folder, output_folder):
    """
    주어진 폴더 내 모든 이미지에 대해 FFT 방식을 사용해 고화질 이미지를 생성하고 저장하는 함수.
    """
    # 폴더가 없다면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 폴더 내 모든 이미지 파일 처리
    image_files = os.listdir(input_folder)
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)

        # 이미지 파일이 아닌 경우 무시
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        # 이미지 읽기
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 그레이스케일로 읽기
        if image is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            continue

        print(f"처리 중: {image_file}")

        # 고화질 이미지 생성
        output_image_path = os.path.join(
            output_folder, f"enhanced_{image_file}")
        enhanced_image = fft_image_enhancement(image, output_image_path)

        # 결과 비교 (원본 vs 고화질)
        plt.subplot(121), plt.imshow(image, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(enhanced_image, cmap='gray')
        plt.title('Enhanced Image (FFT)'), plt.xticks([]), plt.yticks([])

        plt.show()

# 사용할 폴더 경로
input_folder = 'clustered_faces'  # 원본 이미지 폴더
output_folder = 'enhanced_faces'  # 고화질 이미지 저장 폴더

# 폴더 내 모든 이미지에 대해 처리
process_images_in_folder(input_folder, output_folder)