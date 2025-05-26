import pandas as pd

# 파일 불러오기
labels_df = pd.read_csv('../MyFace Dataset Lite/top_gun_maverick/label.csv')
attributes_df = pd.read_csv('result/top_gun_maverick.csv')
deepface_df = pd.read_csv('result/top_gun_maverick_df.csv')

# label.csv와 예측된 속성 데이터를 병합
merged_df = pd.merge(labels_df, attributes_df, on='frame')
merged_df = pd.merge(merged_df, deepface_df, on='frame')

# 속성 컬럼만 추출 (name, filename 제외)
attribute_columns = [col for col in merged_df.columns if col not in ['frame', 'name']]

# 인물별 속성 평균 계산 및 반올림 후 정수화
profile_df = merged_df.groupby('name')[attribute_columns].mean().reset_index()

# 결과 확인
print(profile_df)

# 필요하다면 저장
profile_df.to_csv('profile.csv', index=False)
