import os
import numpy as np
import torch
from PIL import Image
import sys
import shutil
import json
import skimage
import cv2
import matplotlib.pyplot as plt
import transforms as T

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class UNIMIBDataset(object):
    def __init__(self, root, transforms, train=True):
        self.root = root
        self.transforms = transforms
        self.train = train
        if train:
            self.imgs = list((os.listdir(os.path.join(root, "train"))))
        else:
            self.imgs = list(os.listdir(os.path.join(root, "test")))
        

    def __getitem__(self, idx):
        if self.train:
            img_path = os.path.join(self.root, "train", self.imgs[idx])
        else:
            img_path = os.path.join(self.root, "test", self.imgs[idx])
        
        img = Image.open(img_path)
        width, height = img.size
        img = img.convert("RGB")
        tar = self.get_label(self.imgs[idx][:-4], width, height)
        masks = tar["mask"]
        
        # get bounding box coordinates for each mask
        num_objs = len(masks)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = tar["labels"]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
    
    def __len__(self):
        return len(self.imgs)

    def get_label(self, img_name, width, height):
        classes = json.load(open(os.path.join(self.root, "classes.json")))
        if self.train:
            annotations = json.load(
                open(os.path.join(self.root, "train_region_data.json")))
        else:
            annotations = json.load(
                open(os.path.join(self.root, "test_region_data.json")))

        annotations = annotations[img_name]    
        
        # Add images

        target = {}

        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. There are stores in the
        # shape_attributes (see json format above)
        polygons = [r['shape_attributes'] for r in annotations['regions'].values()]
        
        categories = [r['region_attributes']['category']
                        for r in annotations['regions'].values()]
        
        # get the class_ids
        class_ids = []
        for c in categories:
            class_ids.append(classes[c])

        target["labels"] = torch.as_tensor(class_ids, dtype=torch.int64)

        masks = gen_mask(polygons, width, height)
        target["mask"] = masks
        
        # img = cv2.imread(image_path)
        # img = cv2.flip(img, 1)

        # result = color_splash(img, masks.transpose(1, 2, 0))
        # plt.imshow(result)
        # plt.show()
        # break
    
        return target


def get_unimib(root, train=True):
    classes = json.load(open(os.path.join(root, "classes.json")))
    if train:
        annotations = json.load(
            open(os.path.join(root, "train_region_data.json")))
    else:
        annotations = json.load(
            open(os.path.join(root, "test_region_data.json")))
    annotations = list(annotations.values())  # don't need the dict keys

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    for a in annotations:
        target = {}

        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. There are stores in the
        # shape_attributes (see json format above)
        polygons = [r['shape_attributes'] for r in a['regions'].values()]
        # print(polygons)
        x = [p['all_points_x'] for p in polygons]
        y = [p['all_points_x'] for p in polygons]
        # get the category list of regions
        categories = [r['region_attributes']['category']
                        for r in a['regions'].values()]
        
        # get the class_ids
        class_ids = []
        for c in categories:
            class_ids.append(classes[c])

        target["labels"] = torch.as_tensor(class_ids, dtype=torch.int64)
        if train:
            image_path = os.path.join(root+'/train', a['filename']+'.jpg')
        else:
            image_path = os.path.join(root+'/test', a['filename']+'.jpg')
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        masks = gen_mask(polygons, width, height)
        
        img = cv2.imread(image_path)
        img = cv2.flip(img, 1)
        if height < width:
            img = cv2.flip(img, 0)

        result = color_splash(img, masks.transpose(1, 2, 0))
        plt.imshow(result)
        plt.show()


def gen_mask(polygons, width, height):
    """Generate instance masks for an image.
    Returns:
    masks: A bool array of shape [instance count, height, width] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    mask = np.zeros([len(polygons), height, width], dtype=np.uint8)

    for i, p in enumerate(polygons):
        # Get indexes of pixels inside the polygon and set them to 1
        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        if height < width:
            mask[i, rr, cc] = 1
        else:
            mask[i, cc, rr] = 1

    # Return mask, and array of class IDs of each instance. 
    return mask.astype(np.bool)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash

# def get_transform(train):
#     transforms = []
#     transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)

# dataset = UNIMIBDataset('./data/UNIMIB2016', get_transform(train=True))