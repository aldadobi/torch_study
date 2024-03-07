import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
      root="/home/yeom/Desktop/youngeon/code/data",
      train=True,
      download=True,
      transform=ToTensor()
)

test_data = datasets.FashionMNIST(
      root="/home/yeom/Desktop/youngeon/code/data",
      train=False,
      download=True,
      transform=ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
   # print(sample_idx)
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
'''
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
      def __init__ (self, annotations_file, img_dir, transform=None, target_transform=None):
           
           self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
           self.img_dir = img_dir
           self.transform = transform
           self.target_transform = target_transform


      def __len__(self):
           return len(self.img_labels)      
      
      def __getitem__(self,idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                image = self.target_transform(label)
            sample = {'image' : image , 'label' : label}
            return sample
'''
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

###########################
# DataLoader를 통해 순회하기(iterate)
# ------------------------------------------------------------------------------------------
#
# ``DataLoader`` 에 데이터셋을 불러온 뒤에는 필요에 따라 데이터셋을 순회(iterate)할 수 있습니다.
# 아래의 각 순회(iteration)는 (각각 ``batch_size=64`` 의 특징(feature)과 정답(label)을 포함하는) ``train_features`` 와
# ``train_labels`` 의 묶음(batch)을 반환합니다. ``shuffle=True`` 로 지정했으므로, 모든 배치를 순회한 뒤 데이터가 섞입니다.
# (데이터 불러오기 순서를 보다 세밀하게(finer-grained) 제어하려면 `Samplers <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`_
# 를 살펴보세요.)

# 이미지와 정답(label)을 표시합니다.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")