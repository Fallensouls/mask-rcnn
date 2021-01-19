import numpy as np
import datetime
from PIL import Image, ImageOps
from model import get_model_instance_segmentation
import torch
import argparse
import cv2
import random
import transforms as T
import json
import os
import matplotlib.pyplot as plt

def detect_and_color_splash(model, image_path=None, video_path=None, threshold=0.5, dataset_name='unimib'):
    assert image_path or video_path
    rect_th = 5
    text_th = 5
    text_size = 2
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = Image.open(image_path)
        # image = image.transpose(Image.ROTATE_90)
        # image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        transform = T.Compose([T.ToTensor()])
        img, _ = transform(image, None)
        # Detect objects
        images = [img]
        pred = model(images)[0]
        # Color splash
        print(pred)
        pred_score = list(pred['scores'].detach().numpy())

        pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        # pred_t = [pred_score.index(x) for x in pred_score][-1]

        
        labels = pred['labels']
        labels = labels[:pred_t+1]
        
        class_map = get_class_map(dataset_name)
        pred_class = [class_map[i] for i in labels.numpy()]

        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].int().detach().numpy())]
        pred_boxes = pred_boxes[:pred_t+1]
        
        masks = pred['masks'].squeeze(1).detach().numpy() >= 0.5
        masks = masks[:pred_t+1]
        image = np.asarray(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for i in range(len(masks)):
            rgb_mask = random_colour_masks(masks[i])
            image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(image, pred_boxes[i][0], pred_boxes[i][1], color=(0, 255, 0), thickness=rect_th)
            # location = (pred_boxes[i][0][0], pred_boxes[i][0][1]-3)
            # cv2.putText(image, pred_class[i], location, cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0), thickness=text_th)
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(
            datetime.datetime.now())
        cv2.imwrite('result_'+image_path.split('/')[-1], image)
        print("Saved to ", file_name)

def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_class_map(dataset_name='food201'):
    if dataset_name == 'food201':
        with open('./data/food201/labels.txt', 'r') as f:
            classes = f.readlines()
        f.close()
        classes = [x.strip() for x in classes][1:]
    elif dataset_name == 'unimib':
        classes = json.load(open('/home/hatsunemiku/dev/mask-rcnn/data/UNIMIB2016/classes.json'))
        classes = {value:key for key,value in classes.items()} 
    elif dataset_name == 'chfood':
        classes = json.load(open('/home/hatsunemiku/dev/mask-rcnn/data/ch_food/classes.json'))
        classes = {value:key for key,value in classes.items()} 
    else:
        raise ValueError('invalid dataset name')
    
    return classes

def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN')
    parser.add_argument('--image_path', default='/home/hatsunemiku/dev/mask-rcnn/data/food201/segmented_test/pizza/309892.jpg', type=str,
                        help='image path')
    args = parser.parse_args()
    model = get_model_instance_segmentation(num_classes=2)
    model.load_state_dict(torch.load('food201_model.pkl'))
    model.eval()
    detect_and_color_splash(model, args.image_path, dataset_name='food201')

if __name__ == "__main__":
    main()
