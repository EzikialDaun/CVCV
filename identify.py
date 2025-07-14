import cv2
import pandas as pd
from deepface import DeepFace

from cal_distance import dict_similarity
from ppe_analyzer import PPEAnalyzer
from resnet_celeba import CelebAAttributePredictor


# data 딕셔너리의 키 중에서 key_list에 속하는지 여부로 두 딕셔너리로 분리
# 인물 속성 중 장비 속성을 분리하기 위해 사용
def split_dict_by_keys(data, key_list):
    a = {k: v for k, v in data.items() if k in key_list}
    b = {k: v for k, v in data.items() if k not in key_list}
    return a, b


class FaceIdentifier:
    def __init__(self, model_path, profile_csv_path):
        # celebA 데이터셋으로 훈련시킨 모델이 예측하는 속성들
        self.celeba_attributes = ['No_Beard', 'Male']
        # 일관성이 낮아 따로 유사도를 평가하는 속성들.
        self.transient_attributes = [
            'Eyeglasses', 'Wearing_Earrings', 'Wearing_Hat',
            'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
            'glasses', 'hat', 'mask'
        ]
        # deepface retinaface 모델에서 예측하는 속성들
        self.deepface_attributes = [
            'white', 'black', 'asian', 'indian', 'middle eastern',
            'latino hispanic', 'male'
        ]
        # YOLO 모델로 예측하는 속성들
        self.equipments = ['glasses', 'hat']
        # 속성 통합
        self.entire_attributes = self.celeba_attributes + self.deepface_attributes + self.equipments
        # CelebA 모델
        self.predictor = CelebAAttributePredictor(model_path, target_attrs=self.celeba_attributes)
        # YOLO 모델
        self.ppe_analyzer = PPEAnalyzer()
        # 프로필 데이터프레임
        self.profile_df = pd.read_csv(profile_csv_path)

    """
        target_path: 이미지 경로
        profile_dir: 프로필 디렉토리
        vanilla_mode: True이면 속성 유사도 비교 없이 facenet512 얼굴 유사도로만 인물 식별 시도하는 모드
        silent: True이면 일부 콘솔 print문 비활성화
        threshold: facenet512 얼굴 유사도 비교 시 후보군을 한정하는 임계값.
    """
    """
        바닐라 모드 True: 인물의 이름, None
        바닐라 모드 False: 인물의 이름, (target_path의 속성 추출 결과 딕셔너리, 가장 유사한 프로필의 속성 딕셔너리)
    """

    def identify_person(self, target_path, profile_dir, vanilla_mode=False, silent=False, threshold=1.0):
        if vanilla_mode:
            result_list = []
            search_results = DeepFace.find(img_path=target_path, db_path=profile_dir, threshold=threshold,
                                           detector_backend='retinaface', model_name='Facenet512', silent=silent)
            for item in search_results[0].values:
                character = item[0].split('\\')[-2]
                facial_distance = item[11]
                result_list.append({'name': character, 'facial_distance': facial_distance})
            return min(result_list, key=lambda z: z['facial_distance']), None

        # 속성 기반 추론
        analysis = DeepFace.analyze(img_path=target_path, detector_backend='retinaface',
                                    enforce_detection=False, actions=('race', 'gender'))

        # 크롭 후 가로 세로 2배 확장
        img = cv2.imread(target_path)
        region = analysis[0]['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        dx, dy = w // 2, h // 2
        x1, y1 = max(x - dx, 0), max(y - dy, 0)
        x2, y2 = min(x + w + dx, img.shape[1]), min(y + h + dy, img.shape[0])
        face_crop = img[y1:y2, x1:x2]

        # 얼굴 유사도 비교 후 후보군 한정
        search_results = DeepFace.find(img_path=target_path, db_path=profile_dir, threshold=threshold,
                                       detector_backend='retinaface', model_name='Facenet512',
                                       silent=silent, enforce_detection=False)

        # target_path 이미지의 속성 추출 과정
        attr_dict = self.predictor.predict_image(image_ndarray=face_crop)
        equipment_dict = self.ppe_analyzer.detect_wearing_items(image_ndarray=face_crop)
        equipment_dict.pop('mask', None)
        attr_dict.update(equipment_dict)
        if analysis[0]['dominant_gender'] == 'Man':
            attr_dict.update({'male': 1})
        else:
            attr_dict.update({'male': 0})
        attr_dict.update({'white': 0, 'latino hispanic': 0, 'asian': 0, 'black': 0, 'indian': 0, 'middle eastern': 0,
                          analysis[0]['dominant_race']: 1})

        # 일관성이 높은 속성과 낮은 속성으로 분리
        attr_a, attr_b = split_dict_by_keys(attr_dict, self.transient_attributes)

        best_similarity = -1
        best_result = None
        best_target_dict = None
        # 후보군 순회
        for item in search_results[0].values:
            character = item[0].split('\\')[-2]
            facial_distance = item[11]

            profile_data = self.profile_df.loc[self.profile_df['name'] == character].to_dict('records')[0]
            target_dict = {k: profile_data[k] for k in self.entire_attributes if k in profile_data}
            target_a, target_b = split_dict_by_keys(target_dict, self.transient_attributes)

            # 얼굴 거리 -> 얼굴 유사도로 정규화(필요 시 수정)
            facial_similarity = 1 - (facial_distance / threshold)
            attr_similarity = dict_similarity(attr_dict, target_dict)

            # 장비 속성 일치 시 유사도 보너스(필요 시 수정)
            for k in target_a:
                if target_a[k] == 1 and attr_a.get(k, 0) == 1:
                    attr_similarity = min(1.0, attr_similarity * 1.0)

            # 유사도 공식(필요 시 수정)
            similarity = 0.5 * attr_similarity + 0.5 * facial_similarity

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
    # 경로 명시
    MOVIE_NAME = 'top_gun_maverick'
    IMAGE_DIR = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\probe'
    PROFILE_DIR = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\profile'
    PROFILE_PATH = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\profile.csv'
    LABEL_PATH = f'..\\MyFace Dataset Lite\\{MOVIE_NAME}\\label.csv'
    RESULT_PATH = 'identify_result.csv'

    label_df = pd.read_csv(LABEL_PATH).values
    identifier = FaceIdentifier(
        model_path='resnet_celeba_all_attrs_2000each.pth',
        profile_csv_path=PROFILE_PATH
    )

    attr_score = {attr: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for attr in identifier.entire_attributes}
    proposed_cnt = 0
    vanilla_cnt = 0
    total_cnt = 0
    results = []

    # 레이블 순회
    for label in label_df:
        file_path = f"{IMAGE_DIR}\\{label[0]}"
        ground_truth = label[1]
        try:
            # 논문에서 제안하는 방법
            proposed, (predicted_attr, profile_truth_attr) = identifier.identify_person(
                target_path=file_path,
                profile_dir=PROFILE_DIR,
                vanilla_mode=False,
                threshold=1.0
            )
            # 바닐라(기존) 방법(only facenet512)
            vanilla, _ = identifier.identify_person(
                target_path=file_path,
                profile_dir=PROFILE_DIR,
                vanilla_mode=True,
                threshold=1.0
            )

            total_cnt += 1
            if proposed['name'] == ground_truth:
                proposed_cnt += 1
            if vanilla['name'] == ground_truth:
                vanilla_cnt += 1

            # F1 score 누적
            for attr in identifier.entire_attributes:
                p = predicted_attr[attr]
                t = profile_truth_attr[attr]
                if p == 1 and t == 1:
                    attr_score[attr]['TP'] += 1
                elif p == 1 and t == 0:
                    attr_score[attr]['FP'] += 1
                elif p == 0 and t == 1:
                    attr_score[attr]['FN'] += 1
                elif p == 0 and t == 0:
                    attr_score[attr]['TN'] += 1

            result_dict = {
                'frame': label[0],
                'label': ground_truth,
                'proposed_answer': proposed['name'],
                'vanilla_answer': vanilla['name']
            }
            results.append(result_dict)

            print(result_dict)
            # 정확도 출력
            print(f"\nProposed Accuracy: {proposed_cnt} / {total_cnt} = {round(proposed_cnt / total_cnt, 3)}")
            print(f"Vanilla Accuracy: {vanilla_cnt} / {total_cnt} = {round(vanilla_cnt / total_cnt, 3)}")

        except Exception as e:
            print(f"Error with {label[0]}: {e}")

    # F1-score 출력 및 저장
    f1_rows = []
    print("\n=== Attribute-wise F1-Score ===")
    for attr in identifier.entire_attributes:
        TP = attr_score[attr]['TP']
        FP = attr_score[attr]['FP']
        FN = attr_score[attr]['FN']
        TN = attr_score[attr]['TN']

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        print(f"{attr:<20} Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        f1_rows.append({
            'attribute': attr,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN
        })

    # 실험 결과 저장. 실험 종료 후 별도 폴더로 옮기길 권장
    pd.DataFrame(f1_rows).to_csv("attr_f1_score.csv", index=False)
    pd.DataFrame(results).to_csv("identify_result.csv", index=False)
