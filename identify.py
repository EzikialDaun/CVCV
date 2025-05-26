# 인물 식별

import os
from glob import glob

import pandas as pd
from deepface import DeepFace

from cal_distance import dict_similarity
from extract_attributes import create_dict
from utility import natural_key


def optimize_dict(target_dict, unused_list):
    for attr in unused_list:
        target_dict.pop(attr, None)
    return target_dict


def identify(target_path, model_dir, profile_dir, profile_path, vanilla=False, silent=False):
    result_list = []
    if vanilla:
        dfs = DeepFace.find(img_path=target_path, db_path=profile_dir, threshold=1.04,
                            detector_backend='retinaface', model_name='Facenet512', silent=silent)
        for item in dfs[0].values:
            character = item[0].split('\\')[-2]
            # 유사도
            facial_distance = item[11]
            result = {'name': character, 'facial_distance': facial_distance}
            result_list.append(result)
        expected_character = min(result_list, key=lambda x: x['facial_distance'])
        return expected_character
    else:
        profile_df = pd.read_csv(profile_path)
        unused_list = ['frame', 'name']
        attr_dict = create_dict(target_path, model_dir)
        dfs = DeepFace.find(img_path=target_path, db_path=profile_dir, threshold=1.04,
                            detector_backend='retinaface', model_name='Facenet512', silent=silent)
        for item in dfs[0].values:
            # 사전에 정의된 등장인물 이름
            character = item[0].split('\\')[-2]
            # 유사도
            facial_distance = item[11]
            # 후보 이름과 그 얼굴 유사도의 쌍 추출
            result = {'name': character, 'facial_distance': facial_distance}
            # 후보 이름과 일치하는 프로필 행 가져오기
            profile_dict = profile_df.loc[profile_df['name'] == character].to_dict('records')[0]
            attr_distance = dict_similarity(optimize_dict(attr_dict, unused_list),
                                            optimize_dict(profile_dict, unused_list))
            result['attr_distance'] = attr_distance
            result['distance'] = (1 - attr_distance) * result['facial_distance']
            result_list.append(result)
        expected_character = min(result_list, key=lambda x: x['distance'])
        return expected_character


if __name__ == '__main__':
    IMAGE_DIR = '..\\MyFace Dataset Lite\\django_unchained\\probe'
    MODEL_DIR = 'h5_models'
    PROFILE_DIR = '..\\MyFace Dataset Lite\\django_unchained\\profile'
    PROFILE_PATH = '..\\MyFace Dataset Lite\\django_unchained\\profile.csv'

    image_paths = glob(os.path.join(IMAGE_DIR, '*'))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = sorted(image_paths, key=lambda x: natural_key(os.path.basename(x)))

    if not image_paths:
        raise FileNotFoundError(f"이미지 폴더({IMAGE_DIR})에 이미지 파일이 없습니다.")

    for img_path in image_paths:
        print(
            f"{img_path.split('\\')[-1]} - {identify(img_path, MODEL_DIR, PROFILE_DIR, PROFILE_PATH, False, True)}")
