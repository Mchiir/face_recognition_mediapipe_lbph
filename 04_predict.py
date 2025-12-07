#!/usr/bin/env python3
"""
04_predict.py
Run FaceMesh overlay + LBPH face recognition using the trained model.
"""

import cv2
import mediapipe as mp
import json
import math

MODEL_PATH = "models/lbph_model.xml"
LABEL_MAP_PATH = "models/label_map.json"
CAM_INDEX = 1

# Drawing specs
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
DRAWING_SPEC = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(255, 0, 255))
MESH_SPEC = mp_draw.DrawingSpec(thickness=1, color=(255, 0, 255))
EYE_SPEC = mp_draw.DrawingSpec(thickness=2, circle_radius=1, color=(0, 255, 0))
LIPS_SPEC = mp_draw.DrawingSpec(thickness=2, circle_radius=1, color=(0, 0, 255))
NOSE_SPEC = mp_draw.DrawingSpec(thickness=2, circle_radius=1, color=(255, 255, 0))

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

def draw_iris_circle(img, face_landmarks, indices, color=(0, 255, 255)):
    h, w, _ = img.shape
    cx = int(face_landmarks.landmark[indices[0]].x * w)
    cy = int(face_landmarks.landmark[indices[0]].y * h)
    dists = []
    for idx in indices[1:]:
        px = int(face_landmarks.landmark[idx].x * w)
        py = int(face_landmarks.landmark[idx].y * h)
        dists.append(math.dist([cx, cy], [px, py]))
    if dists:
        radius = int(sum(dists) / len(dists))
        cv2.circle(img, (cx, cy), radius, color, 2)

def draw_nostril_circle(img, face_landmarks, outer_idx, inner_idx, color=(0, 165, 255)):
    h, w, _ = img.shape
    ox = face_landmarks.landmark[outer_idx].x * w
    oy = face_landmarks.landmark[outer_idx].y * h
    ix = face_landmarks.landmark[inner_idx].x * w
    iy = face_landmarks.landmark[inner_idx].y * h
    cx = int((ox + ix) / 2.0)
    cy = int((oy + iy) / 2.0)
    radius = int(math.dist([ox, oy], [ix, iy]) / 2.0)
    if radius > 0:
        cv2.circle(img, (cx, cy), radius, color, 2)

def load_model_and_map(model_path: str, label_map_path: str):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
    return recognizer, label_map

def main():
    recognizer, label_map = load_model_and_map(MODEL_PATH, LABEL_MAP_PATH)
    cap = cv2.VideoCapture(CAM_INDEX)

    with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as fm:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                           DRAWING_SPEC, MESH_SPEC)
                    if hasattr(mp_face_mesh, "FACEMESH_LEFT_EYE"):
                        mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_LEFT_EYE, None, EYE_SPEC)
                    if hasattr(mp_face_mesh, "FACEMESH_RIGHT_EYE"):
                        mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_RIGHT_EYE, None, EYE_SPEC)
                    if hasattr(mp_face_mesh, "FACEMESH_LIPS"):
                        mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_LIPS, None, LIPS_SPEC)
                    if hasattr(mp_face_mesh, "FACEMESH_NOSE"):
                        mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_NOSE, None, NOSE_SPEC)

                    draw_nostril_circle(frame, face_landmarks, 98, 97)
                    draw_nostril_circle(frame, face_landmarks, 327, 326)
                    draw_iris_circle(frame, face_landmarks, LEFT_IRIS)
                    draw_iris_circle(frame, face_landmarks, RIGHT_IRIS)

                    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                    ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    face_crop = frame[y_min:y_max, x_min:x_max]
                    if face_crop.size > 0:
                        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                        try:
                            label_id, confidence = recognizer.predict(gray)
                            name = label_map.get(str(label_id), "Unknown")
                            text = f"{name} ({int(confidence)})"
                        except Exception:
                            text = "Unknown"
                        cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("FaceMesh + Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()