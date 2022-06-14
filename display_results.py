import torch
from torch import load, unsqueeze, stack, no_grad
from torch.cuda import empty_cache
from torchvision import transforms

import os
from cv2 import addWeighted, resize
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import gc
from random import shuffle

from model import Shift_model


def tensor_to_heatmap(out, thresholod=0.5) :
    out = stack((out, out, out), dim=2).cpu().numpy()
    out = out.astype(np.float64)
    if thresholod is not None :
        out = np.where(out > thresholod, 1., 0.)
    return out

def get_transform(x) :
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])
    tensor = img_transform(x).float()
    # tensor = unsqueeze(tensor, 0)
    return tensor


if __name__=='__main__':

    path = 'shift.pth'
    models_path = './models'

    full_images_path = '/home/nicolas/unsupervised-detection/dataset/general/no'
    # full_images_path = '/home/nicolas/unsupervised-detection/dataset/general/most_difficult_examples/false_positives'
    # full_images_path = '/home/nicolas/unsupervised-detection/dataset/general/most_difficult_examples/false_negatives'
    size = (256, 172)
    # full_images_path = '/home/nicolas/swimmers_tracking/extractions/unlabelled_images'
    # size = (1024, 1024)

    step_x = size[0] // 5
    step_y = size[1] // 5

    dense = True
    threshold = 0.8

    model = Shift_model()
    model_path = os.path.join(models_path, path)
    model.load_state_dict(load(model_path))
    # model = model.cuda()
    model.eval()

    for root, dirs, files in os.walk(full_images_path) :
        shuffle(files)
        # files.sort()
        # files.sort(key=len, reverse=False)
        batch = []
        imgs = []

        with torch.no_grad():
            for i, file in enumerate(files) :
                ###### IMAGE PREPROCESSING
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = resize(img, size)
                tensor_img = get_transform(img)#.cuda()
                tensor_img = unsqueeze(tensor_img, 0)

                ###### MODEL FORWARD
                out_x_y = model(tensor_img, training=False)[0]
                out = out_x_y[0].numpy()

                ### ARROWS
                x = out_x_y[1].numpy()
                y = out_x_y[2].numpy()
                # x, y = x * size[0], y * size[1]
                x, y = x * size[0], y * size[1]
                if dense:
                    out = cv2.resize(out, size, interpolation=cv2.INTER_LINEAR)
                    x = cv2.resize(x, size)
                    y = cv2.resize(y, size)
                    x, y = x.astype(int), y.astype(int)
                    for xx in range(0, x.shape[0], step_x):
                        for yy in range(0, y.shape[1], step_y):
                            reduction_factor = 1
                            if out[xx, yy] > threshold:
                                start_point = (yy * reduction_factor, xx * reduction_factor)
                                end_point = (start_point[0] + x[xx, yy],
                                             start_point[1] + y[xx, yy])
                                img = cv2.arrowedLine(img, start_point, end_point, (255, 0, 0), 1)
                else:
                    out = np.mean(out, axis=(0, 1))
                    x = np.mean(x, axis=(0, 1))
                    y = np.mean(y, axis=(0, 1))
                    if out > threshold:
                        start_point = (size[0] // 2, size[1] // 2)
                        end_point = (start_point[0] + int(x),
                                     start_point[1] + int(y))
                        img = cv2.arrowedLine(img, start_point, end_point, (255, 0, 0), 1)

                ###

                ###### DISPLAY
                plt.imshow(img)
                plt.show()