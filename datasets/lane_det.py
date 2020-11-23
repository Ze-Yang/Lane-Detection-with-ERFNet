import os
import numpy as np
import cv2
from paddle.fluid.io import Dataset


class LaneDataSet(Dataset):
    def __init__(self, dataset_path, data_list='train', transform=None, is_val=False):
        self.img = os.listdir(os.path.join(dataset_path, data_list))
        self.is_val = is_val
        self.is_testing = 'test' in data_list
        if not self.is_testing:
            self.sky = ['10011130', '10010014', '10024306', '10010116', '10008480', '10016709', '10016688', '10012704',
                        '10016634', '10010679', '10024403', '10013078', '10010443', '10016355', '10014527', '10020544']
        print("{} images loaded.".format(len(self.img)))
        self.img_list = [os.path.join(dataset_path, data_list, x) for x in self.img]
        if 'train' in data_list:
            self.label_list = [x.replace(data_list, 'train_label').replace('jpg', 'png') for x in self.img_list]

        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_list[idx])
        # im_copy = np.copy(image)
        size = image.shape
        if not self.is_testing:
            label = cv2.imread(self.label_list[idx], cv2.IMREAD_UNCHANGED)
            if not self.is_val:
                crop_height = int(size[0] * 1 / 3)
                if self.img[idx][:8] in self.sky:
                    # h = np.random.randint(crop_height + 1)
                    # image = image[h:h + size[0] - crop_height]
                    # label = label[h:h + size[0] - crop_height]
                    image = image[:(size[0] - crop_height)]
                    label = label[:(size[0] - crop_height)]
                else:
                    image = image[crop_height:]
                    label = label[crop_height:]

        if self.transform:
            if self.is_testing:
                for transform in self.transform:
                    image = transform(image)
                # import matplotlib.pyplot as plt
                # image += np.array([103.939, 116.779, 123.68])
                # image = image[:, :, ::-1].astype(np.uint8)
                # plt.imshow(image)
                # plt.show()
                return np.transpose(image, (2, 0, 1)).astype('float32'), self.img[idx], size
            else:
                for transform in self.transform:
                    image, label = transform((image, label))
            # if (label == 17).any() or (label == 16).any() or (label == 9).any() or (label == 10).any():
            # import matplotlib.pyplot as plt
            # image += np.array([103.939, 116.779, 123.68])
            # image = image[:, :, ::-1].astype(np.uint8)
            # plt.imshow(im_copy[:, :, ::-1].astype(np.uint8))
            # plt.show()
            # plt.imshow(image)
            # plt.show()
            # plt.imshow((label * 10).astype(np.uint8))
            # plt.show()

        return np.transpose(image, (2, 0, 1)).astype('float32'), label.astype('int64')


def collate_fn(batch):
    img = [x[0] for x in batch]
    name = np.array([int(x[1].replace('.jpg', '')) for x in batch])
    size = np.array([x[2] for x in batch])
    img = np.stack(img, axis=0)

    return [img, name, size]
