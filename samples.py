import enum
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

colors = [ #BGR
    [0, 0, 0],
    [128, 64, 128],
    [0, 76, 130],
    [0, 102, 0],
    [87, 130, 112],
    [168, 42, 28],
    [30, 41, 48],
    [89, 50, 0],
    [35, 142, 107],
    [70, 70, 70],
    [156, 102, 102],
    [12, 228, 254],
    [12, 148, 254],
    [153, 153, 190],
    [153, 153, 153],
    [96, 22, 255],
    [0, 51, 102],
    [150, 143, 9],
    [32, 11, 119],
    [0, 51, 51],
    [190, 250, 190],
    [146, 150, 112],
    [115, 135, 2],
]
if __name__ == '__main__':


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


        q = batch['query_img'].detach().cpu().numpy().reshape((3,512,512))
        q = ((np.transpose(q, (1,2,0))+2.7)*40).astype(np.uint8)


        prediction = np.zeros((512,512,3), dtype=np.uint8)
        ground_truth = np.zeros((512,512,3), dtype=np.uint8)

        for j, color in enumerate(colors):
            prediction[p_mask==j] = color
            ground_truth[q_mask==j] = color

        q = cv2.cvtColor(q, cv2.COLOR_RGB2BGR)

        results = np.concatenate((q, ground_truth, prediction), axis=1)

        cv2.imwrite(f'sample_results/{i:03d}.png', results)


        # cv2.imshow('Result', results)
        # cv2.waitKey()
        # print(i)


    print(f'100.0%')
    print(f'Mean IoU: {np.nanmean(ious)*100:.2f}%')

    ious = np.array(ious)

    print(f"Class Based Mean IoU : {np.nanmean(ious, axis=0)}")
    print(classes)