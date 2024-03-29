from __future__ import  division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()

landmark_frame = pd.read_csv('/home/yeom/Desktop/youngeon/code/data/faces/face_landmarks.csv')
n = 65
img_name = landmark_frame.iloc[n, 0]
#print(img_name)
landmarks = landmark_frame.iloc[n, 1:]
landmarks = np.asarray(landmarks)

landmarks = landmarks.astype('float').reshape(-1,2)
#img_name2 = os.path.join('/home/yeom/Desktop/youngeon/code/data/faces/', landmark_frame.iloc[n, 0])
#image = io.imread(img_name2)
#print(image)
#print('Image name: {}'.format(img_name))
#print('Landmarks shape: {}'.format(landmarks.shape))
#print('First 4 Landmarks: {}'.format(landmarks[:4]))


def show_landmarks(image, landmarks):
      plt.imshow(image)
      plt.scatter(landmarks[:, 0], landmarks[:, 1], s= 10, marker='.', c='r')
      plt.pause(0.001)

plt.figure()
#show_landmarks(io.imread(os.path.join('/home/yeom/Desktop/youngeon/code/data/faces/', img_name)), landmarks)
plt.show()

#print(len(landmark_frame))
class FaceLandmarksDataset(Dataset):


      def __init__(self, csv_file, root_dir, transform = None):


            self.landmarks_frame = pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform
      

      def __len__(self):
            return len(self.landmarks_frame)

      def __getitem__(self, idx):
            if torch.is_tensor(idx):
                  idx = idx.tolist()

            img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx , 0])

            image = io.imread(img_name)
            landmarks = self.landmarks_frame.iloc[idx , 1:]
            landmarks = np.array([landmarks])
            landmarks = landmarks.astype('float').reshape(-1, 2)
            sample = {'image' : image , 'landmarks' : landmarks}
            
            if self.transform:
                  sample = self.transform(sample)

            return sample
      

face_dataset = FaceLandmarksDataset('/home/yeom/Desktop/youngeon/code/data/faces/face_landmarks.csv', '/home/yeom/Desktop/youngeon/code/data/faces/')

fig = plt.figure()

for i in range(len(face_dataset)):
      sample = face_dataset[i]

      print(i, sample['image'].shape, sample['landmarks'].shape)

      ax = plt.subplot(1, 4, i + 1)
      plt.tight_layout()
      ax.set_title('Sample #{}'.format(i))
      ax.axis('off')
      show_landmarks(**sample)

      if i == 3:
            plt.show()
            break

######################################################
#     tsfm = Transform(params)
#     transformed_sample = tsfm(sample)
######################################################
      
class Rescale(object):

      def __init__(self,output_size):
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size

      def __call__(self,sample):
            image, landmarks = sample['image'], sample['landmarks']

            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                  if h > w:
                        new_h, new_w = self.output_size * h / w, self.output_size
                  else:
                        new_h, new_w = self.output_size, self.output_size * w / h
            else:
                  new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            img = transform.resize(image, (new_h,new_w))

            landmarks = landmarks * [new_h/h , new_w/w] # element-wise multiplication

            return {'image' : img, 'landmarks' : landmarks}
      
class RandomCrop(object):

      def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                  self.output_size = (output_size, output_size)
            else:
                  assert len(output_size) == 2
                  self.output_size = output_size

      def __call__(self, sample):
            image, landmarks = sample['image'], sample['landmarks']

            h, w = image.shape[:2]
            new_h , new_w = self.output_size

            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            image = image[top : top + new_h, left : left + new_w]
            landmarks = landmarks - [left, top]

            return {'image' : image, 'landmarks' : landmarks}
      
class ToTensor(object):
      def __call__(self,sample):
            image, landmarks = sample['image'],sample['landmarks']
            image = image.transpose((2,0,1))
            return {'image' : torch.from_numpy(image), 'landmarks' : torch.from_numpy(landmarks)}



