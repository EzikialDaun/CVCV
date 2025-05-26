import os
import re

import pandas as pd
from deepface import DeepFace
from glob import glob


def get_df_property(image_path):
    result = {'frame': image_path.split('\\')[1]}
    objs = DeepFace.analyze(
        img_path=image_path, actions=['gender', 'race'], detector_backend='retinaface', enforce_detection=False
    )
    filtered_results = [item for item in objs if item.get('face_confidence', 0) > 0.5]
    if len(filtered_results) <= 0:
        print("얼굴을 찾지 못했습니다.")
        return None
    else:
        result['Man'] = round(filtered_results[0]['gender']['Man'] / 100, 2)
        result['Asian'] = round(filtered_results[0]['race']['asian'] / 100, 2)
        result['Indian'] = round(filtered_results[0]['race']['indian'] / 100, 2)
        result['Black'] = round(filtered_results[0]['race']['black'] / 100, 2)
        result['White'] = round(filtered_results[0]['race']['white'] / 100, 2)
        result['Middle_Eastern'] = round(filtered_results[0]['race']['middle eastern'] / 100, 2)
        result['Latino_Hispanic'] = round(filtered_results[0]['race']['latino hispanic'] / 100, 2)
        return result


def natural_key(filename):
    # 숫자와 문자를 나눠서 정렬 키 생성: '10.png' → ['10', '.png']
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]


if __name__ == '__main__':
    MODEL_FOLDER = 'model'  # .keras 파일들이 들어있는 폴더
    IMAGE_FOLDER = 'test'  # 예측할 .png 이미지들이 들어있는 폴더
    RESULT_FILE = 'df_prediction_results.csv'

    attribute_names = [
        'asian'
        'white'
        'black'
        'age'
        'male'
    ]

    # -------------------------
    # 모든 이미지 파일 불러오기
    # -------------------------
    image_paths = glob(os.path.join(IMAGE_FOLDER, '*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = sorted(image_paths, key=lambda x: natural_key(os.path.basename(x)))

    if not image_paths:
        raise FileNotFoundError(f"❌ 이미지 폴더({IMAGE_FOLDER})에 이미지 파일이 없습니다.")

    # -------------------------
    # 예측 수행
    # -------------------------
    results = []

    for img_path in image_paths:
        analyzed = get_df_property(img_path)
        print(analyzed)
        results.append(analyzed)
    # -------------------------
    # 결과 저장 및 출력
    # -------------------------
    df_results = pd.DataFrame(results)
    print("\n✅ 예측 결과:")
    print(df_results)

    # 저장할 경우:
    df_results.to_csv(RESULT_FILE, index=False)
    print(f"\n📁 '{RESULT_FILE}' 파일로 결과 저장 완료!")
