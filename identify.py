import os
import re
from glob import glob

import pandas as pd

from df_analyze import get_df_property
from predict_property import predict_property


def create_dict(path, model_dir):
    deepface_dict = get_df_property(path)
    facial_dict = predict_property(path, model_dir)
    facial_dict.update(deepface_dict)
    return facial_dict


def natural_key(filename):
    # 숫자와 문자를 나눠서 정렬 키 생성: '10.png' → ['10', '.png']
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', filename)]


if __name__ == '__main__':
    # 모델 파일들이 들어있는 폴더
    MODEL_FOLDER = 'h5_models'
    # 예측할 이미지들이 들어있는 폴더
    IMAGE_FOLDER = 'test'
    RESULT_FILE = 'prediction_result.csv'

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
        dt = create_dict(img_path, MODEL_FOLDER)
        print(dt)
        results.append(dt)
    # -------------------------
    # 결과 저장 및 출력
    # -------------------------
    df_results = pd.DataFrame(results)
    print("\n✅ 예측 결과:")
    print(df_results)

    # 저장할 경우:
    df_results.to_csv(RESULT_FILE, index=False)
    print(f"\n📁 '{RESULT_FILE}' 파일로 결과 저장 완료!")
