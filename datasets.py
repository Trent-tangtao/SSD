import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import os
from PIL import Image,ImageDraw,ImageFont
import random
from utils import transform,resize
from utils import *
import torchvision.transforms.functional as FT


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False, mosaic=True):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()

        assert self.split in {'TRAIN', 'TEST'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        # Read data files
        with open(os.path.join(data_folder, self.split + '_images.json'), 'r') as j:
            self.images = json.load(j)
        with open(os.path.join(data_folder, self.split + '_objects.json'), 'r') as j:
            self.objects = json.load(j)

        self.mosaic = mosaic
        self.num_samples = len(self.images)
        assert len(self.images) == len(self.objects)

    def show(self, original_image, det_boxes, det_labels):
        print(det_labels)
        det_labels = [rev_label_map[l] for l in det_labels.tolist()]

        # Annotate
        annotated_image = original_image
        draw = ImageDraw.Draw(annotated_image)
        font = ImageFont.truetype("./calibril.ttf", 15)

        # Suppress specific classes, if needed
        for i in range(det_boxes.size(0)):
            # Boxes
            box_location = det_boxes[i].tolist()
            draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
            draw.rectangle(xy=[l + 1. for l in box_location],outline=label_color_map[det_labels[i]])
            # Text
            text_size = font.getsize(det_labels[i].upper())
            text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
            textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                                box_location[1]]
            draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
            draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                      font=font)
        del draw
        annotated_image.show()
        temp = input()
        print(temp)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        if self.mosaic:
            image, boxes, labels, difficulties = self.get_data_with_Mosaic(i, image, boxes, labels, difficulties)
            # self.show(image,boxes,labels)



        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def get_data_with_Mosaic(self, index, image, boxes, labels, difficulties):
        """loads images in a mosaic
        """
        img_size = 300
        image_s4 = Image.new('RGB', (img_size*2, img_size*2), (255, 255, 255))
        boxes_s4 = torch.FloatTensor([])
        labels_s4 = labels
        difficulties_s4 = difficulties

        indices = [index] + [random.randint(0, self.num_samples - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            image = Image.open(self.images[index], mode='r')
            image = image.convert('RGB')
            objects = self.objects[index]
            boxes = torch.FloatTensor(objects['boxes'])
            labels = torch.LongTensor(objects['labels'])
            difficulties = torch.ByteTensor(objects['difficulties'])
            new_image, new_boxes = resize(image, boxes, dims=(img_size, img_size), return_percent_coords=False)

            # place img in img4
            if i == 0:  # top left
                image_s4.paste(new_image, (0, img_size))
                box = new_boxes+torch.FloatTensor([0, img_size, 0, img_size]).unsqueeze(0)
            elif i == 1:  # top right
                labels_s4 = torch.cat([labels_s4,labels])
                difficulties_s4 = torch.cat([difficulties_s4,difficulties])
                image_s4.paste(new_image, (img_size, img_size))
                box = new_boxes + torch.FloatTensor([img_size, img_size, img_size, img_size]).unsqueeze(0)
            elif i == 2:  # bottom left
                labels_s4 = torch.cat([labels_s4, labels])
                difficulties_s4 = torch.cat([difficulties_s4, difficulties])
                image_s4.paste(new_image, (0, 0))
                box = new_boxes
            elif i == 3:  # bottom right
                labels_s4 = torch.cat([labels_s4, labels])
                difficulties_s4 = torch.cat([difficulties_s4, difficulties])
                image_s4.paste(new_image, (img_size, 0))
                box = new_boxes + torch.FloatTensor([img_size, 0, img_size, 0]).unsqueeze(0)

            boxes_s4 = torch.cat([boxes_s4, box], dim=0)
        return image_s4, boxes_s4, labels_s4, difficulties_s4

    def __len__(self):
        return len(self.images)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)

        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
