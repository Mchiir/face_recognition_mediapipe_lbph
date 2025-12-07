# FaceMesh (AI WITHOUT ML)

## Project Overview

This project demonstrates real-time **facial landmark detection** using **MediaPipe Face Mesh**.

The project allows you to:

- Detect and visualize facial landmarks in real-time.
- Highlight eyes, lips, nose, nostrils, and iris with different colors.
- Use Python and OpenCV for webcam capture and image preprocessing.

> Recommended Python version: **3.12.7** (MediaPipe has compatibility issues with Python 3.13+).

---

## Project Structure

project/
│── 01_create_dataset.py
│── 02_review_dataset.py
│── 03_train_model.py
│── 04_predict.py
│── 05_reset_projet.py
│── dataset/ → your captured faces
│── models/
│ ├── lbph_model.xml
│ └── label_map.json
│── README.md
│── requirements.txt

## Technologies Used

- **Python 3.12.7**
- **OpenCV** – Webcam capture and image preprocessing.
- **MediaPipe** – Real-time facial landmark detection.
- **NumPy** – Array manipulation for image processing.
- **protobuf, attrs, matplotlib** – Required dependencies for MediaPipe.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Mchiir/face_recognition_mediapipe_lbph.git
cd face_recognition_mediapipe_lbph
```

2. Create and activate a virtual environment with Python 3.12.7:

```bash
# Windows
"C:\Program Files\Python312\python.exe" -m venv .venv
.venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install --upgrade pip
pip install --upgrade --no-deps --force-reinstall -r requirements.txt
```

## Project flow logic

## 1. Capture Face Images

Run:

```bash
python 01_create_dataset.py
```

> You will be asked to enter a name for current person/character.
> Images are saved to: dataset/<your_name>/

---

## 2. Review dataset

Run:

```bash
python 02_review_dataset.py
```

> Follow onscreen commands to clean your data for model training.

---

## 3. Train the LBPH Model

Run:

```bash
python 03_train_model.py
```

> This will generate: models/lbph_model.xml and models/label_map.json

---

## 4. Run Face Recognition

Run:

```bash
python 04_predict.py
```

> The camera window will show you your face landmark and recognized name.

---

## 5. Reset project (Optional)

Run:

```bash
python 05_reset_project.py
```

---

## Notes

- Always run the script inside the **Python 3.12.7 virtual environment** to avoid dependency issues.
- Adjust `cap = cv2.VideoCapture(1)` if your primary webcam is at a different index (0 for default).

## License

This project is licensed under the **MIT License**.
