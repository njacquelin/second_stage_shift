from torchvision import transforms
from torch.autograd import Variable
import torch

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import cv2

from dataloader import get_dataloaders


def torch2np(img, inv_trans=True, float_to_uint8=True) :
    invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                   transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
                                  ])
    if inv_trans : img = invTrans(img)
    img = img.permute(1, 2, 0)
    img = img.numpy()
    if float_to_uint8 :
        img *= 255
        img = img.astype(np.uint8)
    return img


def generate_expected_out(reverse_mask, expected_output, out) :
    return out*reverse_mask + expected_output


if __name__=='__main__' :
    data_path = '/home/nicolas/unsupervised-detection/dataset/general/'
    size = (256, 172)

    train_dataloader = get_dataloaders(data_path, size, batch_size=1, train_test_ratio=1)
    train_dataloader.dataset.set_augment(False)

    for batch in train_dataloader:
        image = batch['img'][0]
        img = torch2np(image)

        out, x, y = batch["label"][0]
        x, y = float(x)*size[0], float(y)*size[1]
        x, y = int(x), int(y)
        title = 'YES' if out > 0.5 else 'NO'
        title += "  " + str(x) + "  " + str(y)
        if out > 0.5 :
            start_point = (size[0]//2, size[1]//2)
            end_point = (size[0]//2 + x, size[1]//2 + y)
            img = np.ascontiguousarray(img)
            img = cv2.arrowedLine(img, start_point, end_point, (0, 0, 0), 1)

        plt.title(title)
        plt.imshow(img)
        plt.show()