from ultralytics import YOLO
import cv2
import os
import easyocr
import math
import editdistance


# 안경, 모자, 모자 착용 여부 체크 함수
# 딕셔너리 형태로 각 사물 존재여부 반환
def get_Wearing_Metadata():
    # 모델 경로 및 색상 설정
    models = {
        'glasses': {
            'model': YOLO('yolo11n_glasses only_64b_200e_0414.pt')
        },
        'hat': {
            'model': YOLO('yolo11n_cap only_64b_200e_0414.pt')
        },
        'mask': {
            'model': YOLO('yolo11n_Mask(New) only_64b_200e_0428.pt')
        }
    }

    # 이미지가 저장된 폴더 경로
    image_folder = "test_\\finaltest"
    results_metadata = {}

    # 폴더 내 모든 이미지 파일에 대해 예측 수행
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        # 탐지 여부 초기화
        detected = {
            'glasses': False,
            'hat': False,
            'mask': False
        }

        for label, data in models.items():
            model = data['model']

            # 예측 수행
            results = model(image)[0]

            # 바운딩 박스 존재 여부 확인
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf >= 0.8:
                    detected[label] = True
                    # break  # 하나만 감지되면 True로 충분

        # 결과 이미지 저장 코드
        # results_metadata[image_name] = [detected['glasses'], detected['hat'], detected['mask']]
        # print(f"{image_name}: {results_metadata[image_name]}")

    return results_metadata


def check_hat_in_image(image_path):
    model = YOLO("yolo11n_cap only_64b_200e_0414.pt")
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 감지 실행
    results = model(img)[0]

    for box in results.boxes:
        class_id = int(box.cls)
        if class_id == 0:
            return True

    return False


# 이미지에서 헬멧이 있는지 여부 확인 함수 (True / False)
def check_helmet_in_image(image_path):
    model = YOLO("yolo11n_Helmet only_64b_200e_0527.pt")
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

    # 감지 실행
    results = model(img)[0]

    # 클래스 이름을 확인해서 'helmet' 또는 class_id가 0이라면 감지된 것으로 간주
    for box in results.boxes:
        class_id = int(box.cls)
        if class_id == 0:  # 헬멧 클래스가 0이라고 가정 (필요시 조정)
            return True

    return False


# 이미지에서 OCR 실행 후 결과값 반환 함수
def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en'])  # 필요한 언어 설정
    results = reader.readtext(image_path)

    return [
        {"text": text, "confidence": round(confidence, 2)}
        for _, text, confidence in results
    ]


# OCR 결과값에 대한 CER 값 계산 함수
def calculate_cer(predicted_text, ground_truth):
    """
    CER (Character Error Rate) = edit_distance / length of ground truth
    """
    if not ground_truth:
        return 1.0 if predicted_text else 0.0  # 둘 다 비어있으면 0.0, 정답만 비어있으면 1.0

    # CER 계산
    distance = editdistance.eval(predicted_text, ground_truth)
    cer = distance / len(ground_truth)
    return round(cer, 4)


# CER 값의 범위를 수정하는 지수 함수
def compute_y(x):
    numerator = 0.5 * (math.exp(5 * x) - 1)
    denominator = math.exp(5) - 1
    y = numerator / denominator + 0.5
    return y


# 사용 코드
if __name__ == "__main__":
    # 착용 장비 메타데이터 검출
    # print(get_Wearing_Metadata())

    # 헬멧 착용 여부 확인
    cc = check_helmet_in_image('test_\\top\\5.jpg')
    print(cc)

    # 헬멧을 착용하고 있을 경우, OCR 진행
    if (cc == True):
        result = extract_text_from_image('test_\\top\\5.jpg')
        for item in result:
            print(item["text"])  # 추출된 텍스트
            print(item["confidence"])  # 해당 텍스트의 신뢰도
            CER = calculate_cer(item['text'], 'PAYBACK')
            print(CER)
            print(compute_y(CER))
