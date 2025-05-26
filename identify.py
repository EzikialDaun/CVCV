# 인물 식별

import os
from glob import glob

import pandas as pd
from deepface import DeepFace

from extract_attributes import create_dict
from utility import natural_key

IMAGE_FOLDER = '..\\MyFace Dataset Lite\\django_unchained\\probe'
MODEL_FOLDER = 'h5_models'
PROFILE_DIR = '..\\MyFace Dataset Lite\\django_unchained\\profile'
PROFILE_PATH = '..\\MyFace Dataset Lite\\django_unchained\\profile.csv'

profile_df = pd.read_csv(PROFILE_PATH)
print(profile_df)

image_paths = glob(os.path.join(IMAGE_FOLDER, '*'))
image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_paths = sorted(image_paths, key=lambda x: natural_key(os.path.basename(x)))

if not image_paths:
    raise FileNotFoundError(f"이미지 폴더({IMAGE_FOLDER})에 이미지 파일이 없습니다.")

for img_path in image_paths:
    unused_attr = ['frame', 'name', 'Wearing_Hat', 'Wearing_Necklace', 'Wearing_Necktie', 'Eyeglasses']
    dt = create_dict(img_path, MODEL_FOLDER)
    for attr in unused_attr:
        dt.pop(attr, None)
    print(dt)

    dfs = DeepFace.find(img_path=img_path, db_path=PROFILE_DIR, threshold=1.04,
                        detector_backend='retinaface', model_name='Facenet512')
    pair_list = []
    print(dfs)
    for item in dfs[0].values:
        character = item[0].split('\\')[-2]
        facial_distance = item[11]
        # 후보 이름과 그 얼굴 유사도의 쌍 추출
        pair = {'name': character, 'facial_distance': facial_distance}
        pair_list.append(pair)
        print(pair)
        # 후보 이름과 일치하는 프로필 행 가져오기
        profile_dict = profile_df.loc[profile_df['name'] == character].to_dict('records')
        profile_dict.pop('name', None)

        print(profile_dict)
