#!/usr/bin/env python3
"""
02_review_dataset.py
Preview and curate images in the dataset (permanent delete).
"""

import cv2
import time
from pathlib import Path
import numpy as np
from shutil import rmtree

DATASET_DIR = Path("dataset")
WINDOW_NAME = "Preview"
FRAME_W, FRAME_H = 1280, 720
TOP_BAR_H, BOTTOM_BAR_H = 55, 55
VIEW_W, VIEW_H = FRAME_W, FRAME_H - TOP_BAR_H - BOTTOM_BAR_H
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 180, 0)
WHITE = (255, 255, 255)
PAD_X, PAD_Y = 20, 15
FLASH_MS = 400

def _cleanup_old_trash(root: Path):
    for trash_dir in root.rglob(".trash"):
        try:
            rmtree(trash_dir, ignore_errors=True)
            print(f"[cleanup] Removed old trash folder: {trash_dir}")
        except Exception as e:
            print(f"[warn] Could not remove {trash_dir}: {e}")

def _truncate_to_width(text, max_w, font, scale, thick):
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    if tw <= max_w:
        return text
    if len(text) <= 4:
        return text[:1] + "…"
    left, right = 0, len(text)
    base = text
    while left + 2 < right - 2:
        candidate = base[:2 + left] + "…" + base[-(2 + left):]
        (cw, _), _ = cv2.getTextSize(candidate, font, scale, thick)
        if cw > max_w:
            left += 1
        else:
            right -= 1
    return base[:2 + left] + "…" + base[-(2 + left):]

def preview(folder: str = "dataset", pattern: str = "*.jpg", delay_ms: int = 250):
    folder = Path(folder)
    _cleanup_old_trash(folder)

    files = sorted(
        f for f in folder.rglob(pattern)
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        and not f.name.startswith(".")
    )
    if not files:
        print(f"No images found in '{folder}'")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    flash_ts = 0.0
    flash_text = ""

    def draw_flash(frame):
        if (time.time() - flash_ts) * 1000 <= FLASH_MS and flash_text:
            (tw, th), _ = cv2.getTextSize(flash_text, FONT, 0.65, 2)
            x1, y1 = FRAME_W - PAD_X - tw - 24, 8
            x2, y2 = FRAME_W - PAD_X, 8 + th + 18
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 160, 0), -1)
            cv2.putText(frame, flash_text, (x1 + 12, y2 - 8), FONT, 0.65, WHITE, 2, cv2.LINE_AA)

    def delete_current(img_path: Path):
        nonlocal flash_ts, flash_text
        try:
            img_path.unlink(missing_ok=True)
            flash_text = "Deleted permanently"
            flash_ts = time.time()
            print(f"[deleted] {img_path}")
            return True
        except Exception as e:
            flash_text = "Delete failed"
            flash_ts = time.time()
            print(f"[error] Could not delete {img_path}: {e}")
            return False

    def draw_frame(img_path: Path, autoplay: bool, delay: int, idx: int):
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        img = cv2.imread(str(img_path))
        if img is not None and img.size > 0:
            ih, iw = img.shape[:2]
            scale = min(VIEW_W / iw, VIEW_H / ih)
            new_w, new_h = max(1, int(iw * scale)), max(1, int(ih * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            off_x = (VIEW_W - new_w) // 2
            off_y = TOP_BAR_H + (VIEW_H - new_h) // 2
            frame[off_y:off_y+new_h, off_x:off_x+new_w] = resized

        cv2.rectangle(frame, (0, 0), (FRAME_W, TOP_BAR_H), GREEN, -1)
        left_text = f"[{idx+1}/{len(files)}] "
        state_text = f"  |  {'PLAY' if autoplay else 'PAUSE'}  |  delay={delay}ms"
        max_name_w = FRAME_W - 2*PAD_X - cv2.getTextSize(left_text + state_text, FONT, 0.8, 2)[0][0]
        name = _truncate_to_width(img_path.name, max_name_w, FONT, 0.8, 2)
        top_text = f"{left_text}{name}{state_text}"
        cv2.putText(frame, top_text, (PAD_X, TOP_BAR_H - PAD_Y), FONT, 0.8, WHITE, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (0, FRAME_H - BOTTOM_BAR_H), (FRAME_W, FRAME_H), GREEN, -1)
        help_text = "p: prev   n: next   space/s: pause/resume   +/- speed   d: DELETE (PERMANENT)   q/ESC: quit"
        cv2.putText(frame, help_text, (PAD_X, FRAME_H - PAD_Y), FONT, 0.65, WHITE, 2, cv2.LINE_AA)

        draw_flash(frame)
        return frame

    idx = 0
    autoplay = False
    last = time.time()

    while True:
        if not files:
            blank = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            cv2.rectangle(blank, (0, 0), (FRAME_W, TOP_BAR_H), GREEN, -1)
            cv2.putText(blank, "No images left. Press q to exit.",
                        (PAD_X, TOP_BAR_H - PAD_Y), FONT, 0.8, WHITE, 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, blank)
        else:
            out = draw_frame(files[idx], autoplay, delay_ms, idx)
            cv2.imshow(WINDOW_NAME, out)

        if files and autoplay and (time.time() - last) * 1000 >= delay_ms:
            idx = (idx + 1) % len(files)
            last = time.time()

        key = cv2.waitKeyEx(30)
        KEY_LEFT, KEY_RIGHT = 2424832, 2555904
        if key == -1:
            continue
        if key in (ord('q'), ord('Q'), 27):
            break
        if files and (key in (KEY_RIGHT, ord('n'), ord('N'))):
            idx = (idx + 1) % len(files); last = time.time()
        elif files and (key in (KEY_LEFT, ord('p'), ord('P'))):
            idx = (idx - 1) % len(files); last = time.time()
        elif key in (ord(' '), ord('s'), ord('S')):
            autoplay = not autoplay; last = time.time()
        elif key in (ord('+'), ord('=')):
            delay_ms = max(50, int(delay_ms * 0.8))
        elif key in (ord('-'), ord('_')):
            delay_ms = min(5000, int(delay_ms * 1.25))
        elif files and key in (ord('d'), ord('D')):
            to_delete = files[idx]
            if delete_current(to_delete):
                try:
                    files.pop(idx)
                except Exception:
                    pass
                if files:
                    idx %= len(files)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    preview(folder=str(DATASET_DIR), pattern="*.jpg", delay_ms=250)