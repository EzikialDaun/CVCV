import cv2
import pandas as pd
from deepface import DeepFace

MOVIE_NAME = 'django_unchained'
TARGET_DIR = 'crop'
LABEL_PATH = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\label.csv'
IMAGE_DIR = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\probe'

label_df = pd.read_csv(LABEL_PATH, header=0).values

for label in label_df:
    file_name = f"{IMAGE_DIR}\\{label[0]}"
    print(file_name)
    analyzed = DeepFace.analyze(img_path=file_name, detector_backend='retinaface', enforce_detection=False,
                                actions=('age',))
    img = cv2.imread(file_name)
    img_h, img_w = img.shape[:2]
    region = analyzed[0]['region']
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    origin_crop = img[y:y + h, x:x + w]
    # 확장값 계산
    dx = w // 2
    dy = h // 2
    # 새 좌표 계산 (경계 고려)
    x1 = max(x - dx, 0)
    y1 = max(y - dy, 0)
    x2 = min(x + w + dx, img_w)
    y2 = min(y + h + dy, img_h)
    # 크롭
    face_crop = img[y1:y2, x1:x2]
    cv2.imwrite('origin\\' + file_name.split('\\')[-1], origin_crop)
    cv2.imwrite('extended\\' + file_name.split('\\')[-1], face_crop)
