import math

import cv2
import easyocr
import editdistance
from ultralytics import YOLO


def calculate_cer(predicted_text, ground_truth):
    # Character Error Rate 계산
    if not ground_truth:
        return 1.0 if predicted_text else 0.0
    distance = editdistance.eval(predicted_text, ground_truth)
    cer = distance / len(ground_truth)
    return round(cer, 4)


def compute_y(x):
    # CER 기반 지수 함수 점수화
    numerator = 0.5 * (math.exp(5 * x) - 1)
    denominator = math.exp(5) - 1
    return numerator / denominator + 0.5


def resolve_image(image_path=None, image_ndarray=None):
    if image_ndarray is not None:
        return image_ndarray
    elif image_path:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")
        return img
    else:
        raise ValueError("image_path 또는 image_ndarray 중 하나는 제공해야 합니다.")


class PPEAnalyzer:
    def __init__(self):
        self.models = {
            'glasses': YOLO('yolo11n_glasses only_64b_200e_0414.pt'),
            'hat': YOLO('yolo11n_cap only_64b_200e_0414.pt'),
            'mask': YOLO('yolo11n_Mask(New) only_64b_200e_0428.pt'),
            'helmet': YOLO('yolo11n_Helmet only_64b_200e_0527.pt')
        }
        self.ocr_reader = easyocr.Reader(['en'])

    def detect_wearing_items(self, image_path=None, image_ndarray=None, threshold=0.8):
        img = resolve_image(image_path, image_ndarray)
        detected = {'glasses': 0, 'hat': 0, 'mask': 0}
        for item in ['glasses', 'hat', 'mask']:
            results = self.models[item](img)[0]
            for box in results.boxes:
                if float(box.conf[0]) >= threshold:
                    detected[item] = 1
                    break
        return detected

    def detect_helmet(self, image_path=None, image_ndarray=None):
        img = resolve_image(image_path, image_ndarray)
        results = self.models['helmet'](img)[0]
        for box in results.boxes:
            if int(box.cls) == 0:
                return 1
        return 0

    def extract_text(self, image_path=None, image_ndarray=None):
        img = resolve_image(image_path, image_ndarray)
        results = self.ocr_reader.readtext(img)
        return [
            {"text": text, "confidence": round(confidence, 2)}
            for _, text, confidence in results
        ]


if __name__ == "__main__":
    ppe_analyzer = PPEAnalyzer()
    print(ppe_analyzer.detect_wearing_items('../MyFace Dataset Lite/django_unchained/probe/164.png'))

    temp = cv2.imread('../MyFace Dataset Lite/django_unchained/probe/164.png')
    print(ppe_analyzer.detect_wearing_items(image_ndarray=temp))
