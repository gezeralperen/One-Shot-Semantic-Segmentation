from data_loader import UAVData
from hsnet.model.hsnet import HypercorrSqueezeNetwork
import torch
import torch.nn as nn
from hsnet.common.utils import to_cuda
from torch.utils.data import DataLoader
import cv2
import numpy as np

import time


classes = [
    'paved-area',
    'dirt',
    'grass',
    'gravel',
    'water',
    'rocks',
    'pool',
    'vegetation',
    'roof',
    'wall',
    'window',
    'door',
    'fence',
    'fence-pole',
    'person',
    'dog',
    'car',
    'bicycle',
    'tree',
    'bald-tree',
    'ar-marker',
    'obstacle',
]

first = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

if __name__ == '__main__':

    # Find first appearances
    # K=1
    # while K < len(first)+1:
    #     data_scan = UAVData(-1,K)
    #     for i, batch in enumerate(data_scan):
    #         q_mask = batch['query_mask'].squeeze(0).detach().cpu().numpy().astype(np.uint8)
    #         if np.max(q_mask) == 255:
    #             first[K-1]=i
    #             K += 1
    #             break
    first = [0, 0, 0, 1, 5, 3, 10, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 2, 0, 0, 0, 0]

    # Model initialization
    model = HypercorrSqueezeNetwork('resnet101', False)
    model.eval()

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model)
    model.to(device)

    # Load trained model
    model.load_state_dict(torch.load('hsnet.pt'))

    for C in range(len(classes)):
        data = UAVData(first[C], C+1)

        ious = []

        print(f'Testing for {classes[C]}')

        for i, batch in enumerate(data):
            batch = to_cuda(batch)

            start = time.time()

            pred_mask = model.module.predict_mask_nshot(batch,1).squeeze(0).detach().cpu().numpy().astype(np.uint8)*255

            end = time.time()

            print(f"{1/(end-start):.1f} Hz")

            q_mask = batch['query_mask'].squeeze(0).detach().cpu().numpy().astype(np.uint8)

            intersection = np.sum((q_mask/255)*(pred_mask/255))
            union = np.sum(1-(1-q_mask/255)*(1-pred_mask/255))

            if union == 0:
                continue

            iou = intersection/union
            ious.append(iou)

            print(f'{100*i/len(data):.1f}%')

            continue

            q = batch['query_img'].detach().cpu().numpy().reshape((3,512,512))
            q = ((np.transpose(q, (1,2,0))+2.7)*40).astype(np.uint8)

            opacity = 0.25

            ground_truth = q.copy()
            ground_truth[q_mask==255] = (np.full((512,512,3), [0,0,255], dtype=np.uint8)*np.repeat(q_mask[:, :, np.newaxis]/255, 3, axis=2)*opacity + q*(1-opacity))[q_mask==255]

            prediction = q.copy()
            prediction[pred_mask==255] = (np.full((512,512,3), [255,0,0], dtype=np.uint8)*np.repeat(pred_mask[:, :, np.newaxis]/255, 3, axis=2)*opacity + q*(1-opacity))[pred_mask==255]


            q = cv2.cvtColor(q, cv2.COLOR_RGB2BGR)
            prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
            ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2BGR)

            results = np.concatenate((q, ground_truth, prediction), axis=1)

            cv2.imshow('Result', results)
            cv2.waitKey()
            print(i)

        print(f'100.0%')
        print(f'Mean IoU for {classes[C]}: {np.nanmean(ious)*100:.2f}%')