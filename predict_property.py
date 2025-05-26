import os
from glob import glob

import keras
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

from utility import natural_key


def predict_property(image_path, model_dir, silent=False):
    model_paths = glob(os.path.join(model_dir, '*.h5'))
    model_paths.sort()
    if not model_paths:
        raise FileNotFoundError(f"폴더({MODEL_FOLDER})에 모델 파일이 없습니다.")

    attribute_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]

    # 이미지 로드 및 전처리
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    result = {'frame': image_path.split('\\')[1]}

    for model_path, attr in zip(model_paths, attribute_names):
        model = keras.models.load_model(model_path)

        prediction = model.predict(img_array, verbose=0)
        score = float(prediction[0][0])
        # predicted_class = int(score > 0.5)

        result[f'{attr}'] = round(score, 2)
        if not silent:
            print(f'{attr} - {round(score, 2)}')

    return result


if __name__ == '__main__':
    IMAGE_FOLDER = 'test'
    MODEL_FOLDER = 'h5_models'
    RESULT_FILE = 'prediction_results.csv'

    image_paths = glob(os.path.join(IMAGE_FOLDER, '*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = sorted(image_paths, key=lambda x: natural_key(os.path.basename(x)))

    if not image_paths:
        raise FileNotFoundError(f"❌ 이미지 폴더({IMAGE_FOLDER})에 이미지 파일이 없습니다.")

    results = []

    for img_path in image_paths:
        row_result = predict_property(img_path, MODEL_FOLDER)
        # print(row_result)
        results.append(row_result)

    df_results = pd.DataFrame(results)
    print("\n✅ 예측 결과:")
    print(df_results)

    # 저장할 경우:
    df_results.to_csv(RESULT_FILE, index=False)
    print(f"\n📁 '{RESULT_FILE}' 파일로 결과 저장 완료!")
