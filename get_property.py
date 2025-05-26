import os

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------
# 설정
# -------------------------
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 5
ATTRIBUTE = 'Wearing_Hat'  # 예측할 속성 이름
TOTAL_SAMPLES = 5000  # 사용할 전체 표본 수 지정

# 데이터 경로
main_folder = '../archive'
DATA_DIR = main_folder + '/img_align_celeba/img_align_celeba'  # CelebA 이미지 폴더
LABELS_CSV = main_folder + '/list_attr_celeba.csv'  # CelebA 속성 CSV

# -------------------------
# 데이터 준비
# -------------------------
df = pd.read_csv(LABELS_CSV)

# 속성 하나만 선택하고 라벨을 0/1로 변환
df[ATTRIBUTE] = (df[ATTRIBUTE] == 1).astype(str)

# 이미지 경로를 완성
df['image_path'] = df['image_id'].apply(lambda x: os.path.join(DATA_DIR, x))
# ★ 과소 샘플링 적용
# 1. 클래스별로 분리
class_0 = df[df[ATTRIBUTE] == "False"]
class_1 = df[df[ATTRIBUTE] == "True"]

# 2. 소수 클래스 기준으로 수를 맞춤
minority_count = min(len(class_0), len(class_1))

class_0_under = class_0.sample(minority_count, random_state=42)
class_1_under = class_1.sample(minority_count, random_state=42)

# 3. 다시 합치기
df_balanced = pd.concat([class_0_under, class_1_under]).sample(frac=1, random_state=42).reset_index(drop=True)

# 4. 필요하면 표본 수 제한
if TOTAL_SAMPLES is not None:
    df_balanced = df_balanced.sample(n=TOTAL_SAMPLES, random_state=42)

# 사용 데이터프레임 교체
df = df_balanced

# 먼저 80%(train+val) vs 20%(test)로 나누기
train_val_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df[ATTRIBUTE],  # 클래스 비율 유지
    random_state=42
)

# train_val_df를 다시 75%:25%로 나누어 (train:val) → 결과적으로 전체에서 60%:20%
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.25,
    stratify=train_val_df[ATTRIBUTE],
    random_state=42
)

print(f"Train samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")
print(f"Test samples: {len(test_df)}")

print("Train distribution:")
print(train_df[ATTRIBUTE].value_counts(normalize=True))

print("Validation distribution:")
print(val_df[ATTRIBUTE].value_counts(normalize=True))

print("Test distribution:")
print(test_df[ATTRIBUTE].value_counts(normalize=True))

# -------------------------
# 데이터 제너레이터
# -------------------------
train_gen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
val_gen = ImageDataGenerator(rescale=1. / 255)
test_gen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_gen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col=ATTRIBUTE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),  # ★ 224x224로 리사이즈
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_generator = val_gen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col=ATTRIBUTE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

test_generator = test_gen.flow_from_dataframe(
    dataframe=test_df,
    x_col='image_path',
    y_col=ATTRIBUTE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# -------------------------
# 모델 구성
# -------------------------
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base_model.trainable = False  # feature extractor로 사용

model = Sequential([
    base_model,
    Flatten(),
    BatchNormalization(),  # ★ 추가: 학습 안정화
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # ★ L2 정규화 추가
    Dropout(0.5),  # ★ Dropout 추가
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# -------------------------
# 모델 학습
# -------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# -------------------------
# 테스트 평가
# -------------------------
loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# 모델 저장
model.save(ATTRIBUTE + '.h5')
