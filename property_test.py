import pandas as pd

# -------------------------------
# CSV 불러오기
# -------------------------------
label_df = pd.read_csv('../MyFace Dataset Lite/top_gun_maverick/label.csv')  # frame, name
profile_df = pd.read_csv('../MyFace Dataset Lite/top_gun_maverick/profile.csv')  # name, 속성들
result_df = pd.read_csv('result_extraction.csv')  # frame, 속성들

# -------------------------------
# 병합하여 비교용 테이블 구성
# -------------------------------
merged_df = result_df.merge(label_df, on='frame')  # frame 기준으로 label 연결
merged_df = merged_df.merge(profile_df, on='name', suffixes=('_pred', '_true'))  # name 기준으로 진리값 연결

# -------------------------------
# 속성 리스트 추출
# -------------------------------
attribute_cols = [col for col in result_df.columns if col != 'frame']

# -------------------------------
# 속성별 정확도 계산
# -------------------------------
accuracy_results = {}

for attr in attribute_cols:
    pred = merged_df[f"{attr}_pred"]
    true = merged_df[f"{attr}_true"]
    accuracy = (pred == true).sum() / len(merged_df)
    accuracy_results[attr] = round(accuracy, 3)

# -------------------------------
# 결과 출력
# -------------------------------
print("\n속성별 정확도:")
for attr, acc in accuracy_results.items():
    print(f"{attr:25s}: {acc:.3f}")

# -------------------------------
# 저장 (선택)
# -------------------------------
pd.DataFrame({'attribute': list(accuracy_results.keys()), 'accuracy': list(accuracy_results.values())}).to_csv(
    'attribute_accuracy.csv', index=False)
print("\n속성별 정확도 CSV 저장 완료 → 'attribute_accuracy.csv'")
