# 딕셔너리 유사도 비교

def dict_similarity(dict1, dict2):
    """
    0~1 범위의 값들을 가진 두 딕셔너리 간의 유사도를 계산.
    유사도는 각 key의 (1 - 절댓값 차이)를 평균낸 값.
    """
    if dict1.keys() != dict2.keys():
        raise ValueError("두 딕셔너리는 동일한 키를 가져야 합니다.")

    similarities = []
    for key in dict1:
        v1 = dict1[key]
        v2 = dict2[key]
        diff = abs(v1 - v2)
        # 유사도는 1에서 차이를 뺀 값
        sim = 1 - diff
        similarities.append(sim)

    return sum(similarities) / len(similarities)


if __name__ == '__main__':
    unused_list = ['frame', 'Wearing_Hat', 'Wearing_Necklace', 'Wearing_Necktie']
    a = {'5_o_Clock_Shadow': 0.66, 'Arched_Eyebrows': 0.46, 'Attractive': 0.27,
         'Bags_Under_Eyes': 0.86,
         'Bald': 1.0, 'Bangs': 0.43, 'Big_Lips': 0.86, 'Big_Nose': 0.74, 'Black_Hair': 0.57, 'Blond_Hair': 0.0,
         'Brown_Hair': 0.01, 'Bushy_Eyebrows': 0.49, 'Chubby': 0.67, 'Eyeglasses': 0.01, 'Goatee': 0.93,
         'Gray_Hair': 0.0,
         'Heavy_Makeup': 0.0, 'High_Cheekbones': 0.01, 'Male': 1.0, 'Mustache': 0.98, 'Narrow_Eyes': 0.71,
         'No_Beard': 0.01,
         'Pointy_Nose': 0.25, 'Straight_Hair': 0.08, 'Wavy_Hair': 0.06, 'Wearing_Hat': 0.75, 'Wearing_Necklace': 0.01,
         'Wearing_Necktie': 0.26, 'Young': 0.53, 'Man': 0.81, 'Asian': 0.4, 'Indian': 0.16, 'Black': 0.25,
         'White': 0.03,
         'Middle_Eastern': 0.03, 'Latino_Hispanic': 0.12}
    b = {'5_o_Clock_Shadow': 0.29, 'Arched_Eyebrows': 0.1, 'Attractive': 0.12,
         'Bags_Under_Eyes': 0.48,
         'Bald': 0.28, 'Bangs': 0.3, 'Big_Lips': 0.556, 'Big_Nose': 0.67, 'Black_Hair': 0.23, 'Blond_Hair': 0.0,
         'Brown_Hair': 0.03, 'Bushy_Eyebrows': 0.3, 'Chubby': 0.67, 'Eyeglasses': 0.0, 'Goatee': 0.93,
         'Gray_Hair': 0.0,
         'Heavy_Makeup': 0.0, 'High_Cheekbones': 0.01, 'Male': 1.0, 'Mustache': 0.98, 'Narrow_Eyes': 0.71,
         'No_Beard': 0.01,
         'Pointy_Nose': 0.25, 'Straight_Hair': 0.08, 'Wavy_Hair': 0.06, 'Wearing_Hat': 0.75, 'Wearing_Necklace': 0.01,
         'Wearing_Necktie': 0.26, 'Young': 0.53, 'Man': 0.81, 'Asian': 0.4, 'Indian': 0.16, 'Black': 0.25,
         'White': 0.03,
         'Middle_Eastern': 0.03, 'Latino_Hispanic': 0.12}
    print(dict_similarity(a, b))
