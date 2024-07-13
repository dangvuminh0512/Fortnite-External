import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import torch
from ultralytics import YOLO
import easyocr
import Excel  # Import the update_excel function from Excel.py

# Function to preprocess image for OCR
def preprocess_image_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_img = cv2.resize(processed_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return processed_img

# Function to rotate image if necessary
def rotate_image_if_needed(cropped_img):
    h, w = cropped_img.shape[:2]
    if h > w:
        rotated_img = cv2.rotate(cropped_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return rotated_img
    return cropped_img

def detect_and_recognize_text(model, reader, frame):
    results = model(frame)[0]

    detected_texts = {
        'scored': [],
        'id': [],
        'subject': []
    }

    for box in results.boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
        class_label = int(box.cls[0])

        label_map = {0: 'id', 1: 'scored', 2: 'subject'}
        label = label_map.get(class_label, 'unknown')

        cropped_image = frame[y_min:y_max, x_min:x_max]

        preprocessed_image = preprocess_image_for_ocr(cropped_image)
        preprocessed_image = rotate_image_if_needed(preprocessed_image)

        ocr_results = reader.readtext(preprocessed_image)

        for ocr_result in ocr_results:
            detected_texts[label].append(ocr_result[1])

    return detected_texts

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv8 Text Recognition")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.model_path = 'runs/detect/train/weights/best.pt'
        self.model = YOLO(self.model_path)
        self.reader = easyocr.Reader(['en'])

        self.cap = cv2.VideoCapture(0)
        self.frame_count = 0
        self.running = False

        self.setup_ui()

    def setup_ui(self):
        self.title_label_1 = tk.Label(self.root, text="GRADUATION THESIS", fg="black", font=("Arial", 30))
        self.title_label_1.grid(column=0, row=0, pady=10)

        self.title_label_2 = tk.Label(self.root, text="Automated Test Scoring System", fg="black", font=("Arial", 20))
        self.title_label_2.grid(column=0, row=1, pady=10)

        self.start_button = tk.Button(self.root, text="Start", command=self.start)
        self.start_button.grid(column=0, row=2, pady=5)

        self.label = tk.Label(self.root)
        self.label.grid(column=0, row=3, pady=10)

    def start(self):
        if not self.running:
            self.running = True
            self.start_button.grid_remove()
            self.title_label_1.grid_remove()
            self.title_label_2.grid_remove()
            self.update_frame()

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            if self.frame_count % 5 == 0:
                recognized_texts = detect_and_recognize_text(self.model, self.reader, frame)
                if recognized_texts['id'] and recognized_texts['scored'] and recognized_texts['subject']:
                    Excel.update_excel('Class_list.xlsx', recognized_texts['subject'][0], recognized_texts['id'][0], recognized_texts['scored'][0])
                else:
                    print("Insufficient data to update the Excel sheet.")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        if self.running:
            self.root.after(10, self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
