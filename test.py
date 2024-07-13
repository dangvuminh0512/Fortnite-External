import cv2
import easyocr
import time
from jiwer import wer, cer
import Excel  # Import the update_excel function from Excel.py
from ultralytics import YOLO

def preprocess_image(image):
    """Preprocess the image for better OCR results."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def detect_and_recognize_text(image_path, model_path):
    # Load YOLOv8 model
    model = YOLO(model_path)

    # Load image
    image = cv2.imread(image_path)

    # Perform object detection
    results = model(image)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    detected_texts = {
        'scored': [],
        'id': [],
        'subject': []
    }

    # Iterate over detected bounding boxes
    for result in results:
        boxes = result.boxes  # Extract bounding boxes
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  # Get coordinates
            class_label = int(box.cls[0])  # Get the class label of the detected object

            # Map class label to text category
            label_map = {0: 'id', 1: 'scored', 2: 'subject'}
            label = label_map.get(class_label, 'unknown')

            # Crop the detected text region from the image
            cropped_image = image[y_min:y_max, x_min:x_max]

            # Preprocess the cropped image
            preprocessed_image = preprocess_image(cropped_image)

            # Measure time before EasyOCR processing
            start_time = time.time()

            # Use EasyOCR to recognize text
            ocr_results = reader.readtext(preprocessed_image)

            # Measure time after EasyOCR processing
            end_time = time.time()

            # Calculate processing time
            processing_time = end_time - start_time
            print(f"EasyOCR took {processing_time:.2f} seconds for {len(ocr_results)} text regions.")

            # Store recognized text based on the label
            for ocr_result in ocr_results:
                detected_texts[label].append(ocr_result[1])

    return detected_texts

def calculate_error_rates(recognized_texts, ground_truths):
    error_rates = {'wer': {}, 'cer': {}}
    for label, recognized in recognized_texts.items():
        if recognized:
            recognized_text = ' '.join(recognized)
            ground_truth = ground_truths.get(label, '')  # Fetch ground truth for the label
            error_rates['wer'][label] = wer(ground_truth, recognized_text)
            error_rates['cer'][label] = cer(ground_truth, recognized_text)
    return error_rates

# Usage example
if __name__ == "__main__":
    image_path = 'dataset/test/images/1000001909_jpg.rf.81eedc71e9a8520be01440a48f0aeeda.jpg'  # Replace with your image path
    model_path = 'runs/detect/train3/weights/best.pt'  # Replace with your YOLOv8 model path

    recognized_texts = detect_and_recognize_text(image_path, model_path)



    # Ground truth text for comparison (replace with actual ground truths)
    ground_truths = {
        'scored': '12,5',  # Replace with the actual ground truth for scored text
        'id': 'BI12-264',  # Replace with the actual ground truth for ID text
        'subject': 'ICT.O06'  # Replace with the actual ground truth for subject text
    }

    # Calculate WER and CER
    error_rates = calculate_error_rates(recognized_texts, ground_truths)

    print("WER:", error_rates['wer'])
    print("CER:", error_rates['cer'])

    workbook_path = 'Class_list.xlsx'  # Replace with your Excel file path

    if recognized_texts['id'] and recognized_texts['scored'] and recognized_texts['subject']:
        Excel.update_excel(workbook_path, recognized_texts['subject'][0], recognized_texts['id'][0], recognized_texts['scored'][0])
    else:
        print("Insufficient data to update the Excel sheet.")
