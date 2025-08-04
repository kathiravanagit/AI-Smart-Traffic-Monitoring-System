from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
#import simpleaudio as sa
from ultralytics import YOLO
import time

app = Flask(__name__)

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

DROWSINESS_THRESHOLD = 0.3
SLEEP_THRESHOLD = 0.2
DROWSINESS_FRAMES = 20
SLEEP_FRAMES = 40


drowsiness_counter = 0
sleep_counter = 0


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load beep sound
#wave_obj = sa.WaveObject.from_wave_file('C:\\Users\\DHARSHAN\\Downloads\\beep.wav')


model = YOLO("models/yolov8n.pt")


def smooth_speed(new_speed, prev_speed, alpha=0.2):
    return alpha * new_speed + (1 - alpha) * prev_speed


def overlay_text(frame, text, position, font_scale, color, thickness, bg_color=(0, 0, 0), alpha=0.6):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x, text_y = position

    overlay = frame.copy()
    cv2.rectangle(overlay, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), bg_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    
    roi_y1 = frame_height // 3
    roi_y2 = (frame_height // 3) * 2
    known_distance_meters = 10

    
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        
        frame_count += 1

        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = rgb_frame.astype('uint8')

        
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                
                left_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in [362, 385, 387, 263, 373, 380]])
                right_eye = np.array([(landmarks[i].x, landmarks[i].y) for i in [33, 160, 158, 133, 153, 144]])

                
                h, w, _ = frame.shape
                left_eye = (left_eye * np.array([w, h])).astype(int)
                right_eye = (right_eye * np.array([w, h])).astype(int)

                
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                global drowsiness_counter, sleep_counter
                if avg_ear < SLEEP_THRESHOLD:
                    sleep_counter += 1
                    if sleep_counter >= SLEEP_FRAMES:
                        cv2.putText(frame, "SLEEP ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(frame, (left_eye[0][0], left_eye[0][1]), (right_eye[3][0], right_eye[3][1]), (0, 0, 255), 2)
                        #wave_obj.play()
                elif avg_ear < DROWSINESS_THRESHOLD:
                    drowsiness_counter += 1
                    if drowsiness_counter >= DROWSINESS_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        cv2.rectangle(frame, (left_eye[0][0], left_eye[0][1]), (right_eye[3][0], right_eye[3][1]), (0, 255, 255), 2)
                        #wave_obj.play()
                    sleep_counter = 0
                else:
                    drowsiness_counter = 0
                    sleep_counter = 0
                    cv2.rectangle(frame, (left_eye[0][0], left_eye[0][1]), (right_eye[3][0], right_eye[3][1]), (0, 255, 0), 2)

                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                          mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1))

        
        results = model(frame)

        total_vehicles = 0
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])

                if cls in [2, 3, 5, 7]:
                    total_vehicles += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        
        elapsed_time = time.time() - start_time
        fps_current = frame_count / elapsed_time

        overlay_text(frame, f"FPS: {fps_current:.2f}", (10, 30), 0.8, (255, 255, 255), 2, bg_color=(0, 0, 0))
        overlay_text(frame, f"Total Vehicles: {total_vehicles}", (10, 70), 0.8, (255, 255, 255), 2, bg_color=(0, 0, 0))
        overlay_text(frame, f"Timestamp: {time.strftime('%H:%M:%S')}", (10, 110), 0.8, (255, 255, 255), 2, bg_color=(0, 0, 0))

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
