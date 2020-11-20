import skimage
import numpy as np
import datetime
from PIL import Image
from train import get_transform
from model import get_model_instance_segmentation
import torch
import argparse
import cv2
import random

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path
    model.eval()
    rect_th = 5
    text_th = 3
    text_size = 3
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = Image.open(image_path).convert("RGB")
        images, _ = get_transform(False)(image, None)
        # Detect objects
        images = [images]
        r = model(images)[0]
        # Color splash
        image = cv2.imread(image_path)
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(r['boxes'].detach().numpy())]
        pred_class = [category_names[i] for i in list(r['labels'].numpy())]

        masks = r['masks'].squeeze(1).detach().numpy() >= 0.5
        for i in range(len(masks)):
            rgb_mask = random_colour_masks(masks[i])
            image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)
            cv2.rectangle(image, pred_boxes[i][0], pred_boxes[i][1], color=(0, 255, 0), thickness=rect_th)
            cv2.putText(image, pred_class[i], pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)

        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(
            datetime.datetime.now())
        cv2.imwrite('result.jpg', image)
        print("Saved to ", file_name)


def random_colour_masks(image):
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN')
    parser.add_argument('--image_path', default='', type=str,
                        help='image path')
    
    args = parser.parse_args()
    model = get_model_instance_segmentation(num_classes=2)
    model.load_state_dict(torch.load('model.pkl'))
    detect_and_color_splash(model, args.image_path)


if __name__ == "__main__":
    category_names = []
    with open("./classes.txt", "r") as f:
        line = f.readline()
        while(line):
            category_names.append(line.replace("\n",""))
            line = f.readline()
    main()