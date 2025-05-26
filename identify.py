import os

# from df_analyze import get_df_property
from predict_property import predict_property

if __name__ == '__main__':
    TARGET_PATH = 'test\\0.png'
    MODEL_DIR = 'model'

    # a = get_df_property(TARGET_PATH)
    b = predict_property(TARGET_PATH, MODEL_DIR)
    # print(a)
    print(b)
