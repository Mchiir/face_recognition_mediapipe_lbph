# FaceMesh (AI WITHOUT ML)

## Project Overview

This project demonstrates real-time **facial landmark detection** using **MediaPipe Face Mesh**.

The project allows you to:

- Detect and visualize facial landmarks in real-time.
- Highlight eyes, lips, nose, nostrils, and iris with different colors.
- Use Python and OpenCV for webcam capture and image preprocessing.

> Recommended Python version: **3.12.7** (MediaPipe has compatibility issues with Python 3.13+).

---

## Project structure

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
git clone https://github.com/Mchiir/Webcam-OCR-FaceMesh.git
cd Webcam-OCR-FaceMesh/FaceMesh-MediaPipe-LBPH
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

4. Make sure Tesseract OCR is installed on your system and its path is properly set.

## Usage

1. Run the main script:

```bash
(.venv) cd Webcam-OCR-FaceMesh
(.venv) python main.py
```

## Notes

- Make sure **Tesseract OCR** is installed on your system and its path is set in your environment variables if using OCR.
- Always run the script inside the **Python 3.12.7 virtual environment** to avoid dependency issues.
- Adjust `cap = cv2.VideoCapture(1)` if your primary webcam is at a different index (0 for default).

## License

This project is licensed under the **MIT License**.
