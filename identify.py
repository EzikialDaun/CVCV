# 인물 식별

import pandas as pd
from deepface import DeepFace

from cal_distance import dict_similarity
from extract_attributes import create_dict


def optimize_dict(target_dict, unused_list):
    for attr in unused_list:
        target_dict.pop(attr, None)
    return target_dict


def identify(target_path, model_dir, profile_dir, profile_path, vanilla_mode=False, bias=0.0, silent=False):
    result_list = []
    if vanilla_mode:
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
            result['distance'] = (1 - attr_distance) * (result['facial_distance'] + bias)
            result_list.append(result)
        expected_character = min(result_list, key=lambda x: x['distance'])
        return expected_character


if __name__ == '__main__':
    IMAGE_DIR = 'test'
    MODEL_DIR = 'h5_models'
    PROFILE_DIR = '..\\MyFace Dataset Lite\\django_unchained\\profile'
    PROFILE_PATH = '..\\MyFace Dataset Lite\\django_unchained\\profile.csv'
    LABEL_PATH = '..\\MyFace Dataset Lite\\django_unchained\\label.csv'
    RESULT_PATH = 'identify_result.csv'
    cnt = 0
    proposed_cnt = 0
    vanilla_cnt = 0
    label_df = pd.read_csv(LABEL_PATH, header=0).values
    dict_list = []
    for label in label_df:
        file_name = f"{IMAGE_DIR}\\{label[0]}"
        proposed = identify(file_name, MODEL_DIR, PROFILE_DIR, PROFILE_PATH, False, 0.0, True)
        vanilla = identify(file_name, MODEL_DIR, PROFILE_DIR, PROFILE_PATH, True, 0.0, True)
        if proposed['name'] == label[1]:
            proposed_cnt += 1
        if vanilla['name'] == label[1]:
            vanilla_cnt += 1
        cnt += 1
        print(f"{label[0]}({label[1]})")
        print(proposed)
        print(f"proposed - {proposed_cnt} / {cnt} = {round(proposed_cnt / cnt, 2)}")
        print(vanilla)
        print(f"vanilla - {vanilla_cnt} / {cnt} = {round(vanilla_cnt / cnt, 2)}")
        result_dict = {'frame': label[0], 'label': label[1], 'proposed_answer': proposed['name'],
                       'vanilla_answer': vanilla['name']}
        dict_list.append(result_dict)
    df_result = pd.DataFrame(dict_list)
    df_result.to_csv(RESULT_PATH, index=False)
