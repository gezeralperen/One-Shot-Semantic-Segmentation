import PIL.Image as Image
from torch.utils.data import Dataset
import os
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from torchvision import transforms


class UAVData(Dataset):
    def __init__(self, support_index, cls):

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        self.label_path = 'UAV_DATA/dataset/semantic_drone_dataset/label_images_semantic/'
        self.img_path = 'UAV_DATA/dataset/semantic_drone_dataset/original_images/'
        self.img_files = os.listdir(self.img_path[:-1])
        self.mask_files = os.listdir(self.label_path[:-1])
        
        self.support_index = support_index
        self.cls = cls

        
        self.transform = transforms.Compose([transforms.Resize(size=(512, 512)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.img_mean, self.img_std)])


        self.support_mask = torch.tensor(np.array(Image.open(self.label_path + self.mask_files[support_index]).convert('L')))
        self.support_mask[self.support_mask == cls] = 255
        self.support_mask[self.support_mask != 255] = 0
        self.support_image = Image.open(self.img_path + self.img_files[support_index]).convert('RGB')


    def __len__(self) -> int:
        return len(self.img_files)-1
    
    def __getitem__(self, index: int):
        if index >= self.support_index:
            i = index + 1
        else:
            i = index
        query_mask = torch.tensor(np.array(Image.open(self.label_path + self.mask_files[i]).convert('L')))
        
        query_mask[query_mask == self.cls] = 255
        query_mask[query_mask != 255] = 0

        query_image = Image.open(self.img_path + self.img_files[i]).convert('RGB')

        query_img = self.transform(query_image)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(self.support_image)])

        support_masks_tmp = []
        smask = F.interpolate(self.support_mask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img.unsqueeze(0),
                 'query_mask': query_mask.unsqueeze(0),
                 'query_name': '',

                 'support_imgs': support_imgs.unsqueeze(0),
                 'support_masks': support_masks.unsqueeze(0),
                 'support_names': '',

                 'class_id': torch.tensor(self.cls)}

        return batch


class UAVDataMulti(Dataset):
    def __init__(self, support_index):

        self.img_mean = [0.485, 0.456, 0.406]
        self.img_std = [0.229, 0.224, 0.225]

        self.label_path = 'UAV_DATA/dataset/semantic_drone_dataset/label_images_semantic/'
        self.img_path = 'UAV_DATA/dataset/semantic_drone_dataset/original_images/'
        self.img_files = os.listdir(self.img_path[:-1])
        self.mask_files = os.listdir(self.label_path[:-1])
        
        self.support_index = support_index
        
        self.transform = transforms.Compose([transforms.Resize(size=(512, 512)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.img_mean, self.img_std)])


        self.support_masks = [torch.tensor(np.array(Image.open(self.label_path + self.mask_files[si]).convert('L'))) for si in support_index]
        self.support_images = [Image.open(self.img_path + self.img_files[si]).convert('RGB') for si in support_index]


    def __len__(self) -> int:
        return len(self.img_files)-len(self.support_index)
    
    def __getitem__(self, index: int):
        i = index

        query_mask = torch.tensor(np.array(Image.open(self.label_path + self.mask_files[i]).convert('L')))
        

        query_image = Image.open(self.img_path + self.img_files[i]).convert('RGB')

        query_img = self.transform(query_image)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in self.support_images])

        support_masks_tmp = []
        for smask in self.support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img.unsqueeze(0),
                 'query_mask': query_mask.unsqueeze(0),
                 'query_name': '',

                 'support_imgs': support_imgs.unsqueeze(0),
                 'support_masks': support_masks.unsqueeze(0),
                 'support_names': '',

                 'class_id': 0}

        return batch

if __name__ == '__main__':
    ds = UAVData(31, 15)
    query_mask, query_image = ds[31]

    im1 = np.array(ds.support_mask)
    im2 = np.array(ds.support_image)
    im3 = np.array(query_mask)
    im4 = np.array(query_image)

    im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
    im4 = cv2.cvtColor(im4, cv2.COLOR_RGB2BGR)

    im1 = cv2.resize(im1, (im1.shape[1]//6, im1.shape[0]//6))
    im2 = cv2.resize(im2, (im2.shape[1]//6, im2.shape[0]//6))
    im3 = cv2.resize(im3, (im3.shape[1]//6, im3.shape[0]//6))
    im4 = cv2.resize(im4, (im4.shape[1]//6, im4.shape[0]//6))

    cv2.imshow('Support Mask', im1)
    cv2.imshow('Support Image', im2)
    cv2.imshow('Query Mask', im3)
    cv2.imshow('Query Image', im4)

    cv2.waitKey()
