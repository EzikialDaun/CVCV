# 인물 식별 및 F1-score 계산
import cv2
import pandas as pd
from deepface import DeepFace

from cal_distance import dict_similarity
from ppe_analyzer import PPEAnalyzer
from resnet_celeba import CelebAAttributePredictor

# 사용할 얼굴 속성
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

"""
target_attributes = [
    'Bald', 'Bangs', 'Eyeglasses', 'Male', 'No_Beard',
    'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie'
]
"""

celeba_attributes = [
    'No_Beard',
    'Male'
]

transient_attributes = [
    'Eyeglasses',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'glasses',
    'hat',
    'mask'
]

deepface_attributes = [
    'white',
    'black',
    'asian',
    'indian',
    'middle eastern',
    'latino hispanic',
    'male'
]

equipments = [
    'glasses',
    'hat',
]

entire_attributes = celeba_attributes + deepface_attributes + equipments

# 얼굴 속성 예측기
predictor = CelebAAttributePredictor('resnet_celeba_all_attrs_2000each.pth', target_attrs=celeba_attributes)
ppe_analyzer = PPEAnalyzer()


def split_dict_by_keys(data, key_list):
    a = {k: v for k, v in data.items() if k in key_list}
    b = {k: v for k, v in data.items() if k not in key_list}
    return a, b


def identify(target_path, profile_dataframe, profile_dir, vanilla_mode=False, silent=False, threshold=1.0):
    result_list = []

    if vanilla_mode:
        find_results = DeepFace.find(img_path=target_path, db_path=profile_dir, threshold=threshold,
                                     detector_backend='retinaface', model_name='Facenet512', silent=silent)

        for item in find_results[0].values:
            character = item[0].split('\\')[-2]
            facial_distance = item[11]
            result_list.append({'name': character, 'facial_distance': facial_distance})

        best_match = min(result_list, key=lambda z: z['facial_distance'])
        return best_match, None

    # 속성 기반 비교
    analysis = DeepFace.analyze(img_path=target_path, detector_backend='retinaface',
                                enforce_detection=False, actions=('race', 'gender'))

    img = cv2.imread(target_path)
    img_h, img_w = img.shape[:2]
    region = analysis[0]['region']
    x, y, w, h = region['x'], region['y'], region['w'], region['h']
    dx, dy = w // 2, h // 2
    x1, y1 = max(x - dx, 0), max(y - dy, 0)
    x2, y2 = min(x + w + dx, img_w), min(y + h + dy, img_h)
    face_crop = img[y1:y2, x1:x2]

    search_results = DeepFace.find(img_path=target_path, db_path=profile_dir, threshold=threshold,
                                   detector_backend='retinaface', model_name='Facenet512', silent=silent,
                                   enforce_detection=False)

    attr_dict = predictor.predict_image(image_ndarray=face_crop)
    equipment_dict = ppe_analyzer.detect_wearing_items(image_ndarray=face_crop)
    # 마스크 배제
    equipment_dict.pop('mask', None)
    attr_dict.update(equipment_dict)
    race_dict = {'white': 0, 'latino hispanic': 0, 'asian': 0, 'black': 0, 'indian': 0, 'middle eastern': 0,
                 analysis[0]['dominant_race']: 1}
    if analysis[0]['dominant_gender'] == 'Man':
        attr_dict.update({'male': 1})
    else:
        attr_dict.update({'male': 0})
    attr_dict.update(race_dict)
    attr_a_dict, attr_b_dict = split_dict_by_keys(attr_dict, transient_attributes)

    best_similarity = -1
    best_result = None
    best_target_dict = None

    for item in search_results[0].values:
        character = item[0].split('\\')[-2]
        facial_distance = item[11]

        profile_data = profile_dataframe.loc[profile_dataframe['name'] == character].to_dict('records')[0]
        target_dict = {attr: profile_data[attr] for attr in entire_attributes if
                       attr in profile_data}
        target_a_dict, target_b_dict = split_dict_by_keys(target_dict, transient_attributes)
        facial_similarity = 1 - (facial_distance / threshold)
        attr_similarity = dict_similarity(attr_dict, target_dict)
        w_f = 0.5
        w_a = 0.5
        similarity = w_a * attr_similarity + w_f * facial_similarity

        for k, v in target_a_dict.items():
            if target_a_dict[k] == 1 and attr_a_dict[k] == 1:
                print(f'k : {k} equals, previous sim: {attr_similarity}')
                attr_similarity = min([1, attr_similarity * 1.1])
                print(f'k : {k} equals, new sim : {attr_similarity}')

        if similarity > best_similarity:
            best_similarity = similarity
            best_result = {
                'name': character,
                'face_similarity': facial_similarity,
                'attr_similarity': attr_similarity,
                'similarity': similarity
            }
            best_target_dict = target_dict

    return best_result, (attr_dict, best_target_dict)


if __name__ == '__main__':
    MOVIE_NAME = 'django_unchained'
    IMAGE_DIR = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\probe'
    PROFILE_DIR = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\profile'
    PROFILE_PATH = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\profile.csv'
    LABEL_PATH = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\label.csv'
    RESULT_PATH = 'identify_result.csv'

    label_df = pd.read_csv(LABEL_PATH).values
    profile_df = pd.read_csv(PROFILE_PATH)

    proposed_cnt = 0
    vanilla_cnt = 0
    total_cnt = 0
    results = []

    # F1-score 관련 카운터
    attr_score = {attr: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for attr in entire_attributes}

    for label in label_df:
        file_name = f"{IMAGE_DIR}\\{label[0]}"
        ground_truth = label[1]

        try:
            proposed, (predicted_attr, true_attr) = identify(file_name, profile_df, PROFILE_DIR, vanilla_mode=False,
                                                             threshold=0.8)
            vanilla, _ = identify(file_name, profile_df, PROFILE_DIR, vanilla_mode=True, threshold=0.8)

            total_cnt += 1
            if proposed['name'] == ground_truth:
                proposed_cnt += 1
            if vanilla['name'] == ground_truth:
                vanilla_cnt += 1

            print(f"{label[0]}({ground_truth})")
            print("Proposed:", proposed)
            print(f"Proposed Accuracy: {proposed_cnt} / {total_cnt} = {round(proposed_cnt / total_cnt, 2)}")
            print("Vanilla:", vanilla)
            print(f"Vanilla Accuracy: {vanilla_cnt} / {total_cnt} = {round(vanilla_cnt / total_cnt, 2)}")

            results.append({
                'frame': label[0],
                'label': ground_truth,
                'proposed_answer': proposed['name'],
                'vanilla_answer': vanilla['name']
            })

            # F1 계산을 위한 TP/FP/FN/TN 누적
            for attr in entire_attributes:
                prediction = predicted_attr[attr]
                truth = true_attr[attr]

                if prediction == 1 and truth == 1:
                    attr_score[attr]['TP'] += 1
                elif prediction == 1 and truth == 0:
                    attr_score[attr]['FP'] += 1
                elif prediction == 0 and truth == 1:
                    attr_score[attr]['FN'] += 1
                elif prediction == 0 and truth == 0:
                    attr_score[attr]['TN'] += 1

        except Exception as e:
            print(f"Error with {label[0]}: {e}")

    # F1-score 출력 및 저장
    print("\n=== Attribute-wise F1-Score ===")
    f1_rows = []

    for attr in entire_attributes:
        TP = attr_score[attr]['TP']
        FP = attr_score[attr]['FP']
        FN = attr_score[attr]['FN']
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        print(f"{attr:<20} Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        f1_rows.append({
            'attribute': attr,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': attr_score[attr]['TN']
        })

    # CSV 저장
    f1_df = pd.DataFrame(f1_rows)
    f1_df.to_csv("attr_f1_score.csv", index=False)

    # 결과 저장
    pd.DataFrame(results).to_csv(RESULT_PATH, index=False)
