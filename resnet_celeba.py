import os
import csv
from glob import glob

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from utility import natural_key


class CelebAAttributePredictor:
    def __init__(self, model_path, target_attrs=None, device=None):
        self.ATTRIBUTES = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
            'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
            'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
            'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]

        self.target_attrs = target_attrs or self.ATTRIBUTES
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

    def _load_model(self, path):
        model = models.resnet18()
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, len(self.ATTRIBUTES)),
            nn.Sigmoid()
        )
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict_image(self, image_path=None, image_ndarray=None, threshold=0.5):
        if image_ndarray is not None:
            # OpenCV 형식(BGR)을 PIL 형식(RGB)으로 변환
            if isinstance(image_ndarray, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image_ndarray, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("image_ndarray는 numpy.ndarray 타입이어야 합니다.")
        elif image_path is not None:
            image = Image.open(image_path).convert('RGB')
        else:
            raise ValueError("image_path 또는 image_ndarray 중 하나는 반드시 제공해야 합니다.")

        image = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            output = output.cpu().numpy().flatten()

        result = {
            attr: int(prob > threshold)
            for attr, prob in zip(self.ATTRIBUTES, output)
            if attr in self.target_attrs
        }
        return result

    def predict_directory(self, folder_path, threshold=0.5):
        result_dict = {attr: [] for attr in self.target_attrs}
        supported_exts = ('.png', '.jpg', '.jpeg')

        for filename in os.listdir(folder_path):
            if filename.lower().endswith(supported_exts):
                file_path = os.path.join(folder_path, filename)
                preds = self.predict_image(file_path, threshold)
                for attr in self.target_attrs:
                    result_dict[attr].append(preds.get(attr, 0))

        return result_dict

    def predict_to_csv(self, folder_path, output_csv, threshold=0.5):
        image_paths = glob(os.path.join(folder_path, '*'))
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths = sorted(image_paths, key=lambda x: natural_key(os.path.basename(x)))

        if not image_paths:
            raise FileNotFoundError(f"{folder_path} 폴더에 이미지가 없습니다.")

        with open(output_csv, 'w', newline='') as csvfile:
            header = ['frame'] + self.target_attrs
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()

            for img_path in image_paths:
                frame_name = os.path.basename(img_path)
                print(frame_name)
                row = {'frame': frame_name}
                row.update(self.predict_image(img_path, threshold))
                writer.writerow(row)

        print(f"CSV 저장 완료: {output_csv}")


if __name__ == "__main__":
    model_path = 'resnet_celeba_all_attrs_2000each.pth'
    target_attributes = [
        'Arched_Eyebrows', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Eyeglasses',
        'Goatee', 'High_Cheekbones', 'Male', 'Narrow_Eyes', 'No_Beard',
        'Receding_Hairline', 'Sideburns', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie'
    ]

    predictor = CelebAAttributePredictor(model_path, target_attrs=target_attributes)

    # 단일 이미지 예측
    print(predictor.predict_image('test/2.png'))

    # 디렉토리 예측
    # results = predictor.predict_directory('test')

    # CSV로 저장
    # predictor.predict_to_csv('test', 'output.csv')
