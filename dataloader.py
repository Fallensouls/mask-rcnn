import os
import numpy as np
import torch
from PIL import Image, ImageOps
import json
import skimage
import random
import cv2
import matplotlib.pyplot as plt

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
        
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)

        width = image.width
        height = image.height
        
        image = image.convert("RGB")

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
            image, target = self.transforms(image, target)

        return image, target
    
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
    
        return target

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
        points_y = p['all_points_y']
        points_x = p['all_points_x']
        rr, cc = skimage.draw.polygon(points_x, points_y)
        cc = width - cc
        mask[i, rr, cc] = 1

    # Return mask, and array of class IDs of each instance. 
    return mask.astype(np.bool)


class Food201Dataset(object):
    def __init__(self, root, transforms, train=True):
        self.root = root
        self.transforms = transforms
        self.train = train
        # load all image files, sorting them to
        # ensure that they are aligned
        if self.train:
            with open('./data/food201/train_pixel_annotations.txt', 'r') as f:
                lines = f.readlines()
            self.masks = [line.strip() for line in lines]
            self.imgs = [line.strip().split('.')[0]+'.jpg' for line in lines]
        else:
            with open('./data/food201/test_pixel_annotations.txt', 'r') as f:
                lines = f.readlines()
            self.masks = [line.strip()[23:] for line in lines]
            self.imgs = [line.strip().split('.')[0][23:]+'.jpg' for line in lines]


    def __getitem__(self, idx):
        # load images ad masks
        if self.train:
            img_path = os.path.join(self.root, "segmented_train", self.imgs[idx])
            mask_path = os.path.join(self.root, "new_masks_train", self.masks[idx])
        else:
            img_path = os.path.join(self.root, "segmented_test", self.imgs[idx])
            mask_path = os.path.join(self.root, "new_masks_test", self.masks[idx])
        # print(img_path)
        # print(mask_path)
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
        obj_ids = obj_ids[1:] if obj_ids[0] == 0 else obj_ids
        # print(mask)
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

        # num_objs_type = len(obj_ids)
        # boxes = []
        # labels = []
        # num_objs = 0
        # split_masks = []
        # for i in range(num_objs_type):
        #     _, binary_mask = cv2.threshold(masks[i].astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
        #     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     for j in range(len(contours)):
        #         x, y, w, h = cv2.boundingRect(contours[j])   
        #         boxes.append([x, y, x+w, y+h])
        #         labels.append(obj_ids[i])
        #         num_objs += 1
        #         split_mask = np.zeros(masks[i].shape)
        #         split_mask[y:y+h,x:x+w] = masks[i][y:y+h,x:x+w]
        #         split_masks.append(split_mask)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # print(labels)
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

        # # 可视化代码
        # class_map = get_class_map()
        # pred_class = [class_map[i] for i in labels.numpy()]
        # print(pred_class)
        # boxes = boxes.int()
        # boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(boxes.detach().numpy())]
        # image = np.asarray(img)
        # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # for i in range(len(masks)):
        #     rgb_mask = random_colour_masks(masks[i])
            
        #     image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
        #     cv2.rectangle(image, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=3)
        #     location = (boxes[i][0][0], boxes[i][0][1]-3)
        #     cv2.putText(image, pred_class[i], location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)

        # plt.imshow(image)
        # plt.show()
        # plt.imshow(img)
        # plt.show()

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        # print(target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_class_map(dataset_name='food201'):
    if dataset_name == 'food201':
        with open('./data/food201/labels.txt', 'r') as f:
            classes = f.readlines()
        f.close()
        classes = [x.strip() for x in classes][1:]
    elif dataset_name == 'unimib':
        classes = json.load(open('/home/hatsunemiku/dev/mask-rcnn/data/UNIMIB2016/classes.json'))
        classes = {value:key for key,value in classes.items()} 
    else:
        raise ValueError('invalid dataset name')
    return classes

def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


# image_path = './data/food201/segmented_test/dumplings/834049.jpg'
# mask_path = './data/food201/new_masks_test/dumplings/834049.png'
# image = Image.open(image_path).convert("RGB")
# image = np.asarray(image)
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# mask = Image.open(mask_path)
# # convert the PIL Image into a numpy array
# mask = np.array(mask)
# # instances are encoded as different colors
# obj_ids = np.unique(mask)
# # print(obj_ids)
# # print(obj_ids[0])

# obj_ids = obj_ids[1:] if obj_ids[0] == 0 else obj_ids
# # print(obj_ids)
# # np.set_printoptions(threshold=np.inf)
# # print(mask)
# # split the color-encoded mask into a set
# # of binary masks
# masks = mask == obj_ids[:, None, None]

# # get bounding box coordinates for each mask
# num_objs_type = len(obj_ids)
# boxes = []
# labels = []
# num_obj = 0
# split_masks = []
# for i in range(num_objs_type):
#     _, binary_mask = cv2.threshold(masks[i].astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
#     contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for j in range(len(contours)):
#         x, y, w, h = cv2.boundingRect(contours[j])   
#         boxes.append([x, y, x+w, y+h])
#         labels.append(obj_ids[i])
#         num_obj += 1
#         split_mask = np.zeros(masks[i].shape)
#         split_mask[y:y+h,x:x+w] = masks[i][y:y+h,x:x+w]
#         split_masks.append(split_mask)
# print(num_obj)
# print(boxes)
# labels = torch.as_tensor(labels, dtype=torch.int64)
# print(labels)
# class_map = get_class_map()
# pred_class = [class_map[i] for i in labels.numpy()]
# boxes = torch.as_tensor(boxes, dtype=torch.float32)

# boxes = boxes.int()
# boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(boxes.detach().numpy())]
# plt.imshow(image)
# plt.show()
# for i in range(num_obj):
#     rgb_mask = random_colour_masks(split_masks[i])
#     image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
#     cv2.rectangle(image, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=3)
#     location = (boxes[i][0][0], boxes[i][0][1]-3)
#     cv2.putText(image, pred_class[i], location, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), thickness=2)

# plt.imshow(image)
# plt.show()