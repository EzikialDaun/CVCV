# .keras -> .h5

import os
from glob import glob

import tensorflow as tf

from utility import natural_key


def k2h5(path, new_dir='.'):
    # .keras 모델 로드
    model = tf.keras.models.load_model(path)
    # .h5 파일로 저장
    file_name = path.split('\\')[1]
    file_name = natural_key(file_name)[0]
    model.save(f'{new_dir}\\{file_name}.h5')


if __name__ == '__main__':
    DIR = 'model'
    NEW_DIR = 'h5_models'

    # 모든 이미지 파일 불러오기
    model_paths = glob(os.path.join(DIR, '*'))
    model_paths = [p for p in model_paths if p.lower().endswith('.keras')]
    model_paths = sorted(model_paths, key=lambda x: natural_key(os.path.basename(x)))

    if not model_paths:
        raise FileNotFoundError(f"이미지 폴더({DIR})에 모델 파일이 없습니다.")

    for model_path in model_paths:
        k2h5(model_path, NEW_DIR)
