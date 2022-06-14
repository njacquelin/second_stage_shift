import pickle
import sys

import pandas as pd
import os
import cv2
from random import randint, random


def get_positif(data, presence_min_threshold, ratio_wh, size):
    index = randint(0, len(data)-1)
    bbox = data.iloc[index]
    center_x = bbox['x']
    center_y = bbox['y']
    w = bbox['w']
    h = bbox['h']
    if w/h > ratio_wh :
        w = max(size[0], w)
        h = w//2
    else:
        h = max(size[1], h)
        w = 2*h

    delta_x = w * (1 - presence_min_threshold**0.5) * random() - size[0] * (1 - presence_min_threshold) / 2
    delta_y = h * (1 - presence_min_threshold**0.5) * random() - size[1] * (1 - presence_min_threshold) / 2

    new_center_x = center_x + int(delta_x)
    new_center_x = 1600-w//2 if new_center_x > 1600-w//2 else w//2 if new_center_x < w//2 else new_center_x
    new_center_y = center_y + int(delta_y)
    new_center_y = 900-h//2 if new_center_y > 900 - h//2 else h//2 if new_center_y < h//2 else new_center_y

    xmin = new_center_x - w // 2
    ymin = new_center_y - h // 2
    xmax = new_center_x + w // 2
    ymax = new_center_y + h // 2

    final_shift_x = center_x - new_center_x
    final_shift_y = center_y - new_center_y

    return (xmin, xmax, ymin, ymax), (final_shift_x, final_shift_y)


def get_negatif(data, size):
    condition = False
    while not condition :
        y1 = randint(0, 1000)
        y2 = randint(0, 1000)
        y2 = y2 if abs(y1 - y2) >= size[1] \
            else y1 + size[1] if y1 < 500 else y1 - size[1]
        ymin, ymax = min(y1, y2), max(y1, y2)

        potential_squares1 = data['ymin'] < ymax
        potential_squares2 = data['ymax'] > ymin
        potential_squares = data[potential_squares1 & potential_squares2].groupby(level=0).first() # groupby stuff => unique
        condition = True if len(potential_squares) != 0 else condition

    xmin = min(potential_squares['xmin'])
    xmax = max(potential_squares['xmax'])

    intervals = []
    if xmin >= size[0] : intervals.append([0, xmin])
    if 2000-xmax >= size[0] : intervals.append([xmax, 2000])

    if len(intervals) == 0 :
        print('error intervals')
        sys.exit()

    xmin, xmax = intervals[randint(0, len(intervals)-1)]
    side = min(xmax-xmin, ymax-ymin)

    y_anchor = randint(ymin, ymax - side)
    x_anchor = randint(xmin, xmax - side)

    return x_anchor, x_anchor+side, y_anchor, y_anchor+side


def get_and_adapt_data(path):
    data = pickle.load(open(path, 'rb'))
    data.sort_values(['frame'], inplace=True)
    # data['xmin'] *= 2000 // 1600
    # data['xmax'] *= 2000 // 1600
    # data['ymin'] *= 1000 // 900
    # data['ymax'] *= 1000 // 900
    # data['x'] *= 2000 // 1600
    # data['w'] *= 2000 // 1600
    # data['y'] *= 1000 // 900
    # data['h'] *= 1000 // 900
    data['xmin'] = data['x'] - data['w']//2
    data['xmax'] = data['x'] + data['w']//2
    data['ymin'] = data['y'] - data['h']//2
    data['ymax'] = data['y'] + data['h']//2
    return data


def normalize(delta_x, delta_y, size):
    delta_x /= size[0]
    delta_y /= size[1]
    return delta_x, delta_y



if __name__ == '__main__':
    # path = '../../PhD_HPE/data/annotations/runs_2styles.pkl'  # for 640x320 images
    path = "/home/nicolas/swimmers_tracking/extractions/labels_pickle/dataframe_bboxes.pkl"  # for 1600x900 images

    pair_of_img_per_frames = 9
    presence_min_threshold = 0.3
    size = (256, 172)
    ratio_wh = 2

    data = get_and_adapt_data(path)
    # data = pickle.load(open(path, 'rb'))
    # data.sort_values(['frame'], inplace=True)

    for i, f in enumerate(data['frame'].unique()) :
        if i%50==0 : print(f)
        frame = data[data['frame'] == f]

        positifs = []
        for i in range(pair_of_img_per_frames):
            pos, final_shift = get_positif(frame, presence_min_threshold, ratio_wh, size)
            positifs.append((pos, final_shift))

        # negatifs = []
        # for i in range(pair_of_img_per_frames):
        #     neg = get_negatif(frame, size)
        #     negatifs.append(neg)

        img_path = os.path.join("/home/nicolas/swimmers_tracking/extractions/labelled_images/both", f)
        img = cv2.imread(img_path)

        positifs_path = '/home/nicolas/unsupervised-detection/dataset/general/shifts'
        if not os.path.isdir(positifs_path) : os.mkdir(positifs_path)
        for i, ((xmin, xmax, ymin, ymax), (delta_x, delta_y)) in enumerate(positifs):
            delta_x, delta_y = normalize(delta_x, delta_y, size)
            crop_name = f[:-4] + '_' + str(delta_x) + '_' + str(delta_y) + '_.jpg'
            crop_path = os.path.join(positifs_path, crop_name)
            crop = img[ymin:ymax, xmin:xmax]
            crop = cv2.resize(crop, size)
            cv2.imwrite(crop_path, crop)


        # negatifs_path = '../dataset/breaststroke/no'
        # if not os.path.isdir(negatifs_path): os.mkdir(negatifs_path)
        # for i, (xmin, xmax, ymin, ymax) in enumerate(negatifs):
        #     crop_path = os.path.join(negatifs_path, str(i) + '_' + f)
        #     crop = img[ymin:ymax, xmin:xmax]
        #     crop = cv2.resize(crop, size)
        #     if random() > 0.5 : crop = cv2.flip(crop, 1)
        #     cv2.imwrite(crop_path, crop)
