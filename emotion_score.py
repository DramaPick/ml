import cv2
from deepface import DeepFace
from multiprocessing import Pool, cpu_count

def process_frame(frame_info):
    frame_idx, frame, fps, emotion_threshold = frame_info

    frame_height, frame_width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # 가장 큰 얼굴 선택
    largest_face = max(faces, key=lambda box: box[2] * box[3])
    x, y, w, h = largest_face
    face_area = w * h
    frame_area = frame_width * frame_height

    if face_area / frame_area < 0.05:
        return None

    # DeepFace로 감정 분석
    face = frame[y:y+h, x:x+w]
    try:
        analysis = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
        if isinstance(analysis, list):
            analysis = max(analysis, key=lambda result: max(result.get("emotion", {}).values()))

        emotion_scores = analysis.get("emotion", {})
        max_emotion = max(emotion_scores, key=emotion_scores.get)
        max_score = emotion_scores[max_emotion]

        if max_score > emotion_threshold:
            timestamp = frame_idx / fps
            return {"timestamp": timestamp, "emotion": max_emotion, "score": max_score}

    except Exception as e:
        print(f"Error analyzing frame {frame_idx}: {e}")
        return None

# --- 메인 실행 코드 ---

def extract_emotion_highlights(video_path, fps=60, emotion_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    sampling_rate = fps  # 초당 1프레임 분석
    frames_to_process = []

    # 프레임 수집
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sampling_rate == 0:
            frames_to_process.append((frame_idx, frame, fps, emotion_threshold))

        frame_idx += 1

    cap.release()

    # 병렬 처리 실행
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_frame, frames_to_process)

    # None 값 제거
    highlights = [result for result in results if result is not None]

    return highlights

# --- 실행 예시 ---

if __name__ == "__main__":
    video_path = "눈물의여왕.mp4"  # 비디오 파일 경로
    fps = 60
    emotion_threshold = 0.5

    highlights = extract_emotion_highlights(video_path, fps, emotion_threshold)

    # 결과 출력
    for highlight in highlights:
        print(f"Timestamp: {highlight['timestamp']}s, Emotion: {highlight['emotion']}, Score: {highlight['score']}")
