import tensorflow as tf
from typing import List
import cv2
import os
import mediapipe as mp

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def crop_mouth(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mouth = []
            for i in range(48, 68):  # Mouth landmarks
                x = int(face_landmarks.landmark[i].x * frame.shape[1])
                y = int(face_landmarks.landmark[i].y * frame.shape[0])
                mouth.append((x, y))

            mouth = sorted(mouth, key=lambda x: x[1])
            top_left = mouth[0]
            bottom_right = mouth[-1]

            cropped_frame = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            resized_frame = cv2.resize(cropped_frame, (140, 46))  # Resize to (width, height)
            return resized_frame

    return frame

def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    cropped_frames = []
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret or i >= 75:
            break
        cropped_frame = crop_mouth(frame)
        frame = tf.image.rgb_to_grayscale(cropped_frame)
        frames.append(frame)
        cropped_frames.append(cropped_frame)  # Keep cropped frame for debugging/visualization
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std, cropped_frames

def load_data(path: tf.Tensor) -> List[float]:
    path = bytes.decode(path.numpy())
    file_name = os.path.basename(path).split('.')[0]
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
    video_path = os.path.join(script_dir, 'data', 's1', f'{file_name}.mpg')
    
    frames, cropped_frames = load_video(video_path)
    return frames, cropped_frames
