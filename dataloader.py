import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import os
import cv2
from cv2 import resize, GaussianBlur
import numpy as np
from random import random


class F_M_Dataloader(Dataset) :
    def __init__(self, path, size) :
        self.transform = self.get_transform()
        self.size = size
        self.augment_data = True

        F_path = os.path.join(path, 'shifts')
        F_files = os.listdir(F_path)
        F_files = [os.path.join(F_path, f) for f in F_files]
        x_shifts = [float(f.split("_")[-3]) for f in F_files]
        y_shifts = [float(f.split("_")[-2]) for f in F_files]
        F_labels = [torch.tensor([1., x, y]) for f, x, y in zip(F_files, x_shifts, y_shifts)]

        M_path = os.path.join(path, 'no')
        M_files = os.listdir(M_path)
        M_files = [os.path.join(M_path, f) for f in M_files]
        M_labels = [torch.tensor([0., 0., 0.]) for f in M_files]

        self.files = M_files + F_files
        self.labels = M_labels + F_labels

    def __len__(self) :
        return len(self.files)

    def __getitem__(self, idx) :
        label = self.labels[idx]
        img_name = self.files[idx]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.augment_data:

            if random() < 0.9:
                img = self.color_shift(img)

            if random() < 0.9:  # gaussian blurr
                if random() > 0.5:
                    kernel = 3
                    img = GaussianBlur(img, (kernel, kernel), 0)
                else:
                    img = self.salt_pepper(img)

            if random() < 0.9:
                img = self.random_patch(img)

            if random() > 0.5:
                img = self.lr_flip(img)
                label[1] *= -1

        img = resize(img, self.size)
        img = self.transform(img)

        return {'img': img, 'label': label}

    def salt_pepper(self, img):
        s_vs_p = 0.5
        amount = 0.04

        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        img[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        img[tuple(coords)] = 0
        return img

    def zoom_out(self, img):
        reducing_factor = 1 + random() * 0.5  # [1, 2[
        h, w, _ = img.shape
        new_size = int(w / reducing_factor), int(h / reducing_factor)
        x_margin = int(random() * (h - new_size[1]))
        y_margin = int(random() * (w - new_size[0]))

        grey = np.zeros_like(img)
        img = cv2.resize(img, new_size)
        grey[x_margin: x_margin + new_size[1],
        y_margin: y_margin + new_size[0]] = img

        return grey

    def get_transform(self):
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
        return img_transform

    def random_crop(self, img):
        min = 0.7  # side of the cropped image in respect to the original one
        prop = random() * (1 - min) + min

        h, w, _ = img.shape
        xmin = int(random() * w * (1 - prop))
        xmax = int(xmin + prop * w)
        ymin = int(random() * h * (1 - prop))
        ymax = int(ymin + prop * h)

        img = img[ymin:ymax, xmin:xmax]

        return img

    def color_shift(self, img):
        brightness = int((random() * 128 - 64))
        contrast = int((random() * 128 - 64))
        hue = random() * 40 - 20

        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)

        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img)
        hnew = np.mod(h + hue, 180).astype(np.uint8)
        hsv = cv2.merge([hnew, s, v])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img

    def random_patch(self, img):
        max_patch_nb = 5
        patch_nb = int(random() * max_patch_nb)
        size_max = 0.2
        size_min = 0.05
        for p in range(patch_nb):

            prop_x = random() * size_max + size_min
            prop_y = random() * (size_max - prop_x) ** 2 + size_min
            if random() > 0.5: prop_x, prop_y = prop_y, prop_x

            h, w, _ = img.shape
            xmin = int(random() * w * (1 - prop_x))
            ymin = int(random() * h * (1 - prop_y))
            xmax = int(xmin + prop_x * w)
            ymax = int(ymin + prop_y * h)

            r = random()
            # r = 0.9
            if r < 0.3:  # noise
                img[ymin:ymax, xmin:xmax] = np.random.randint(0, 255, (ymax - ymin, xmax - xmin, 3))
            elif r < 0.6:  # grey shade
                img[ymin:ymax, xmin:xmax] = np.ones((ymax - ymin, xmax - xmin, 3)) * int(random() * 255)
            else:
                hsv_patch = np.ones((ymax - ymin, xmax - xmin, 3)) * [random() * 180, random() * 256, random() * 256]
                bgr_patch = cv2.cvtColor(hsv_patch.astype(np.uint8), cv2.COLOR_HSV2BGR)
                img[ymin:ymax, xmin:xmax] = bgr_patch

        return img

    def lr_flip(self, img):
        img = np.flip(img, axis=1)
        return img

    def to_tensor(self, x):
        # x = transforms.ToTensor()(x).float()
        x = torch.tensor(x)
        x = x.permute(2, 0, 1).long()
        return x

    def set_augment(self, val):
        self.augment_data = val


def get_dataloaders(data_path, size, batch_size=32, train_test_ratio=0.8):
    dataset = F_M_Dataloader(data_path, size)
    if train_test_ratio != 1 :
        train_size = int(train_test_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataset.data_augmentation = False
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader
    else :
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader