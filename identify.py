# 인물 식별
import cv2
import pandas as pd
from deepface import DeepFace

from cal_distance import dict_similarity
from resnet_celeba import CelebAAttributePredictor

# 19가지 속성 기본 방법
"""
target_attributes = [
    'Arched_Eyebrows',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Eyeglasses',
    'Goatee',
    'High_Cheekbones',
    'Male',
    'Narrow_Eyes',
    'No_Beard',
    'Receding_Hairline',
    'Sideburns',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie'
]
"""

# 1번 방법
target_attributes = [
    'Arched_Eyebrows',
    'Bags_Under_Eyes',
    'High_Cheekbones',
    'Narrow_Eyes'
]


# facial distance가 가장 작은 1개만 남기고 중복 삭제
def deduplicate_by_min_distance(data):
    result = {}
    for item in data:
        profile = item['profile']
        dist = item['facial_distance']
        if profile not in result or dist < result[profile]['facial_distance']:
            result[profile] = item
    return list(result.values())


# 인물 속성 추출 모델 선언
predictor = CelebAAttributePredictor('resnet_celeba_all_attrs_2000each.pth', target_attrs=target_attributes)


# 입력 -> 추론 대상 이미지 경로, 프로필 디렉토리, 프로필 csv 경로, only facenet512 모드, 콘솔 텍스트 출력 금지 모드, 얼굴 유사도 임계
# 출력 -> (추론 결과 등장인물 이름, [등장인물 후보와 얼굴 유사도, 속성 일치 여부 딕셔너리])
def identify(target_path, profile_dir, profile_path, vanilla_mode=False, silent=False,
             threshold=1.04):
    result_list = []
    test_list = []
    if vanilla_mode:
        dfs = DeepFace.find(img_path=target_path, db_path=profile_dir, threshold=threshold,
                            detector_backend='retinaface', model_name='Facenet512', silent=silent)
        for item in dfs[0].values:
            character = item[0].split('\\')[-2]
            # 유사도
            facial_distance = item[11]
            result = {'name': character, 'facial_distance': facial_distance}
            result_list.append(result)
        expected_character = min(result_list, key=lambda x: x['facial_distance'])
        return expected_character, test_list
    else:
        profile_df = pd.read_csv(profile_path)

        analyzed = DeepFace.analyze(img_path=target_path, detector_backend='retinaface', enforce_detection=False,
                                    actions=('race', 'gender'))
        img = cv2.imread(target_path)
        region = analyzed[0]['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        face_crop = img[y:y + h, x:x + w]
        print(analyzed)
        dfs = DeepFace.find(img_path=target_path, db_path=profile_dir, threshold=threshold,
                            detector_backend='retinaface', model_name='Facenet512', silent=silent,
                            enforce_detection=False)
        attr_dict = predictor.predict_image(image_ndarray=face_crop)
        print(attr_dict)

        for item in dfs[0].values:
            # 사전에 정의된 등장인물 이름
            character = item[0].split('\\')[-2]
            # 유사도
            facial_distance = item[11]
            # 후보 이름과 그 얼굴 유사도의 쌍 추출
            result = {'name': character}
            # 후보 이름과 일치하는 프로필 행 가져오기
            profile_dict = profile_df.loc[profile_df['name'] == character].to_dict('records')[0]
            target_dict = {}
            for i in target_attributes:
                if i in profile_dict:
                    target_dict[i] = profile_dict[i]
            print(target_dict)
            match_result = {key: int(attr_dict[key] == target_dict[key]) for key in attr_dict}
            match_result['frame'] = target_path.split('\\')[-1]
            match_result['profile'] = character
            match_result['facial_distance'] = facial_distance
            test_list.append(match_result)
            test_list = deduplicate_by_min_distance(test_list)
            facial_similarity = 1 - (facial_distance / threshold)
            attr_similarity = dict_similarity(attr_dict, target_dict)
            result['face_similarity'] = facial_similarity
            result['attr_similarity'] = attr_similarity
            result['distance'] = (attr_similarity + facial_similarity) / 2.0
            print(result)
            result_list.append(result)
        expected_character = max(result_list, key=lambda x: x['distance'])
        return expected_character, test_list


if __name__ == '__main__':
    MOVIE_NAME = 'django_unchained'
    IMAGE_DIR = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\probe'
    PROFILE_DIR = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\profile'
    PROFILE_PATH = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\profile.csv'
    LABEL_PATH = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\label.csv'
    RESULT_PATH = 'identify_result.csv'
    MATCH_RESULT_PATH = 'match_result.csv'

    cnt = 0
    proposed_cnt = 0
    vanilla_cnt = 0
    dict_list = []
    match_list = []

    label_df = pd.read_csv(LABEL_PATH, header=0).values

    for label in label_df:
        file_name = f"{IMAGE_DIR}\\{label[0]}"
        try:
            proposed = identify(file_name, PROFILE_DIR, PROFILE_PATH, vanilla_mode=False, silent=False,
                                threshold=0.8)
            vanilla = identify(file_name, PROFILE_DIR, PROFILE_PATH, vanilla_mode=True, silent=False,
                               threshold=0.8)
            match_list += proposed[1]
            if proposed[0]['name'] == label[1]:
                proposed_cnt += 1
            if vanilla[0]['name'] == label[1]:
                vanilla_cnt += 1
            cnt += 1
            print(f"{label[0]}({label[1]})")
            print(proposed)
            print(f"proposed - {proposed_cnt} / {cnt} = {round(proposed_cnt / cnt, 2)}")
            print(vanilla)
            print(f"vanilla - {vanilla_cnt} / {cnt} = {round(vanilla_cnt / cnt, 2)}")
            result_dict = {'frame': label[0], 'label': label[1], 'proposed_answer': proposed[0]['name'],
                           'vanilla_answer': vanilla[0]['name']}
            dict_list.append(result_dict)
        except ValueError as e:
            print(e)

    match_csv = pd.DataFrame(match_list)
    match_csv.to_csv(MATCH_RESULT_PATH, index=False)
    df_result = pd.DataFrame(dict_list)
    df_result.to_csv(RESULT_PATH, index=False)
