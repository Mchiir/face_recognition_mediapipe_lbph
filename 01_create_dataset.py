#!/usr/bin/env python3
"""
01_create_dataset.py
Create a face dataset using MediaPipe face detection.
"""

import cv2
import os
import time
import sys
import mediapipe as mp

DATASET_DIR = "dataset"
CAM_INDEX = 1
IMAGES_TO_CAPTURE = 50
SAVE_INTERVAL_MS = 300
PAUSE_AFTER_DONE = 2.0
PADDING = 0.25  # relative padding around detected bbox

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def relbox_to_bbox(xmin, ymin, w, h, iw, ih, pad=PADDING):
    # convert relative bbox to pixel coords and apply padding
    x = xmin * iw
    y = ymin * ih
    ww = w * iw
    hh = h * ih
    pad_w = ww * pad
    pad_h = hh * pad
    x1 = int(clamp(x - pad_w, 0, iw - 1))
    y1 = int(clamp(y - pad_h, 0, ih - 1))
    x2 = int(clamp(x + ww + pad_w, 0, iw - 1))
    y2 = int(clamp(y + hh + pad_h, 0, ih - 1))
    if x2 <= x1: x2 = clamp(x1 + 1, 0, iw - 1)
    if y2 <= y1: y2 = clamp(y1 + 1, 0, ih - 1)
    return x1, y1, x2, y2

def main():
    print(f"Capture {IMAGES_TO_CAPTURE} images. Press 'q' to quit.")
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam."); sys.exit(1)

    name = None
    save_path = None
    count = 0
    last_ts = 0

    mp_face = mp.solutions.face_detection

    with mp_face.FaceDetection(min_detection_confidence=0.5) as face_detector:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] No frame."); break

                frame = cv2.flip(frame, 1)
                ih, iw = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = face_detector.process(rgb)
                detections = results.detections if results and results.detections else []
                n = len(detections)

                # prompt when exactly one face and name not set
                if n == 1 and name is None:
                    bbox = detections[0].location_data.relative_bounding_box
                    x1, y1, x2, y2 = relbox_to_bbox(bbox.xmin, bbox.ymin, bbox.width, bbox.height, iw, ih)
                    display = frame.copy()
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.imshow("Capture", display); cv2.waitKey(1)
                    entered = input("One face detected. Enter character name: ").strip()
                    if not entered:
                        print("Invalid name."); continue
                    name = entered
                    save_path = os.path.join(DATASET_DIR, name)
                    ensure_dir(save_path)
                    print(f"Saving to: {save_path}")
                    time.sleep(0.3)

                # if not exactly one face, show hint and continue
                if n != 1:
                    if n > 1:
                        cv2.putText(frame, "Multiple faces - show only one", (10,60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    cv2.putText(frame, f"Faces: {n}", (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.imshow("Capture", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                # one face detected
                bbox = detections[0].location_data.relative_bounding_box
                x1, y1, x2, y2 = relbox_to_bbox(bbox.xmin, bbox.ymin, bbox.width, bbox.height, iw, ih)

                # save crop from original frame (no overlay)
                if name and count < IMAGES_TO_CAPTURE:
                    now_ms = time.time() * 1000
                    if (now_ms - last_ts) >= SAVE_INTERVAL_MS:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            fname = os.path.join(save_path, f"{name}_{count+1:03d}.jpg")
                            cv2.imwrite(fname, gray_crop)
                            count += 1
                            last_ts = now_ms
                            print(f"[{name}] saved {count}/{IMAGES_TO_CAPTURE}")

                # display rectangle and status
                disp = frame.copy()
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(disp, f"Saved: {count}/{IMAGES_TO_CAPTURE}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Capture", disp)

                if name and count >= IMAGES_TO_CAPTURE:
                    print(f"\nFinished capturing {IMAGES_TO_CAPTURE} images for '{name}'\nSaved to: {save_path}")
                    time.sleep(PAUSE_AFTER_DONE)
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nInterrupted.")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Exiting.")

if __name__ == "__main__":
    main()