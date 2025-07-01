# test.py
import torch
from torchvision import transforms, models
from dataset import CelebADataset
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('Agg')  # GUI 없이 그래프를 그릴 수 있는 백엔드 사용
import matplotlib.pyplot as plt
import numpy as np

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
main_dir = '../archive'
csv_path = main_dir + '/list_attr_celeba.csv'
image_dir = main_dir + '/img_align_celeba/img_align_celeba'
model_path = 'resnet_celeba_all_attrs_2000each.pth'
batch_size = 64

# 이미지 목록
img_list = [f"{i:06d}.jpg" for i in range(200001, 202001)]

# 다중 속성 리스트
# attrs = ['Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Straight_Hair', 'Wavy_Hair']
attrs = [
    '5_o_Clock_Shadow',
    'Arched_Eyebrows',
    'Attractive',
    'Bags_Under_Eyes',
    'Bald',
    'Bangs',
    'Big_Lips',
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Blurry',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Double_Chin',
    'Eyeglasses',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mouth_Slightly_Open',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Receding_Hairline',
    'Rosy_Cheeks',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Wearing_Earrings',
    'Wearing_Hat',
    'Wearing_Lipstick',
    'Wearing_Necklace',
    'Wearing_Necktie',
    'Young'
]

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# 데이터셋 & 로더
test_dataset = CelebADataset(csv_path, image_dir, transform, img_list, attributes=attrs)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 모델 로드
model = models.resnet18()
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 40),
    torch.nn.Sigmoid()
)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 평가
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()

print(f"Test Accuracy: {correct / total:.4f}")
print("-----------------------------------")

# 평가: 속성별 정확도
num_attrs = len(attrs)
correct_per_attr = [0] * num_attrs
total_per_attr = [0] * num_attrs

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).float()

        for i in range(num_attrs):
            correct_per_attr[i] += (preds[:, i] == labels[:, i]).sum().item()
            total_per_attr[i] += labels.size(0)  # 배치 크기만큼 증가

# # 결과 출력
# for i, attr in enumerate(attrs):
#     acc = correct_per_attr[i] / total_per_attr[i]
#     print(f"{attr} Accuracy: {acc:.4f}")


# 결과 출력 (정확도 높은 순 정렬)
attr_accuracies = [(attr, correct_per_attr[i] / total_per_attr[i]) for i, attr in enumerate(attrs)]
attr_accuracies.sort(key=lambda x: x[1], reverse=True)  # 정확도 기준 내림차순 정렬

for attr, acc in attr_accuracies:
    print(f"{attr} Accuracy: {acc:.4f}")

# 상위 5개, 하위 5개 추출
top5 = attr_accuracies[:5]
bottom5 = attr_accuracies[-5:]


# 시각화
def plot_attr_accuracies(data, title, color, filename):
    attrs, accs = zip(*data)
    plt.figure(figsize=(10, 4))
    # plt.bar(attrs, accs, color=color, width=0.15)  # 막대 폭 줄이기
    plt.bar_label(plt.bar(attrs, accs, color=color, width=0.15))
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.05, 0.2))  # y축 눈금 0.05 간격
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()  # 메모리 해제


plot_attr_accuracies(top5, 'Top 5 Attributes by Accuracy', color='green', filename='graph\\top5_accuracy.png')
plot_attr_accuracies(bottom5, 'Bottom 5 Attributes by Accuracy', color='red', filename='graph\\bottom5_accuracy.png')


# 전체 속성 정확도 시각화
def plot_all_attr_accuracies(data, title, color, filename):
    attrs, accs = zip(*data)
    plt.figure(figsize=(16, 6))  # 너비 늘려서 가독성 향상
    bars = plt.bar(attrs, accs, color=color, width=0.4)  # 막대 너비 줄임
    plt.bar_label(bars, fmt='%.2f', fontsize=8, label_type='edge', padding=2)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1.05, 0.2))
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(rotation=60, ha='right', fontsize=8)  # 글자 겹침 방지
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# 사용 예시
plot_all_attr_accuracies(
    attr_accuracies,
    title='Accuracy of All 40 Attributes',
    color='skyblue',
    filename='graph\\all_attr_accuracy.png'
)


# 시각화 함수
def plot_top15_attr_accuracies(data, title, color, filename):
    attrs, accs = zip(*data)
    plt.figure(figsize=(14, 5))  # 충분한 가로 공간 확보
    bars = plt.bar(attrs, accs, color=color, width=0.5)  # 막대 적당한 굵기
    plt.bar_label(bars, fmt='%.2f', fontsize=9, label_type='edge', padding=3)
    min_acc = min(accs)
    max_acc = max(accs)

    # y축 범위 및 간격 설정
    y_min = max(0, min_acc - 0.05)
    y_max = min(1, max_acc + 0.05)
    plt.ylim(y_min, y_max)
    plt.yticks(np.linspace(round(y_min, 2), round(y_max, 2), 6))  # 적절한 간격 설정

    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


top15 = attr_accuracies[:15]
# 사용 예시
plot_top15_attr_accuracies(
    top15,
    title='Top 15 Attributes by Accuracy',
    color='seagreen',
    filename='graph\\top15_accuracy.png'
)
