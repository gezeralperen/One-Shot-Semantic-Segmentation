from data_loader import UAVData, UAVDataMulti
from hsnet.model.hsnet import HypercorrSqueezeNetwork
import torch
import torch.nn as nn
from hsnet.common.utils import to_cuda
from torch.utils.data import DataLoader
import cv2
import numpy as np


def batch_to_class(batch, class_index):
        support_masks = batch['support_masks'].clone()
        support_masks_tmp = []
        for support_mask in support_masks:
            support_mask[support_mask == class_index] = 255
            support_mask[support_mask != 255] = 0
            support_masks_tmp.append(support_mask)
        support_masks_tmp = torch.stack(support_masks_tmp)
        query_mask = batch['query_mask'].clone()
        query_mask[query_mask == class_index] = 255
        query_mask[query_mask != 255] = 0

        return {'query_img': batch['query_img'],
                 'query_mask': query_mask,
                 'query_name': batch['query_name'],

                 'support_imgs': batch['support_imgs'],
                 'support_masks': support_masks_tmp,
                 'support_names': batch['support_names'],

                 'class_id': batch['class_id']}

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


    data = UAVDataMulti([0,1,2,3,5,6,10])

    ious = []

    print(f'Testing for Multiclass')

    for i, batch in enumerate(data):
        batch = to_cuda(batch)

        multi_pred = torch.empty((0,512,512)).to(device)

        for c in range(len(classes)):
            class_batch = batch_to_class(batch, c+1)

            pred_mask = model.module.predict_mask_nshot(class_batch,7, prob=True)
            multi_pred = torch.cat((multi_pred,pred_mask), dim=0)



        p_mask = torch.argmax(multi_pred, dim=0).detach().cpu().numpy().astype(np.uint8)+1
        q_mask = batch['query_mask'].squeeze(0).detach().cpu().numpy().astype(np.uint8)

        class_ious = []

        for c in range(len(classes)):
            q = np.zeros((512,512), dtype=np.uint8)
            p = np.zeros((512,512), dtype=np.uint8)
            q[q_mask==c+1] = 1
            p[p_mask==c+1] = 1
            intersection = np.sum(q*p)
            union = np.sum(1-(1-q)*(1-p))
            if union == 0:
                iou = np.nan
            else:
                iou = intersection/union
            class_ious.append(iou)

        ious.append(class_ious)


        print(f'{100*i/len(data):.1f}% Mean IoU: {np.nanmean(ious)*100:.2f}%', end='\r')


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
    print(f'Mean IoU: {np.nanmean(ious)*100:.2f}%')

    ious = np.array(ious)

    print(f"Class Based Mean IoU : {np.nanmean(ious, axis=0)}")
    print(classes)