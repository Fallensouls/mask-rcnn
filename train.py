import torch

import utils
import transforms as T
from dataloader import UNIMIBDataset, Food201Dataset, CHFoodDataset
from engine import train_one_epoch, evaluate
from model import get_model_instance_segmentation
import argparse

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main():
    parser = argparse.ArgumentParser(description='Mask R-CNN')
    parser.add_argument('--dataset', default='unimib', type=str,
                        choices=['unimib', 'food201', 'chfood'],
                        help='dataset name')
    args = parser.parse_args()
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.dataset == 'unimib':
        num_classes = 74
        # use our dataset and defined transformations
        dataset = UNIMIBDataset('/home/hatsunemiku/dev/mask-rcnn/data/UNIMIB2016', get_transform(train=True))
        dataset_test = UNIMIBDataset('/home/hatsunemiku/dev/mask-rcnn/data/UNIMIB2016', get_transform(train=False), False)

    elif args.dataset == 'food201':
        num_classes = 2
        # use our dataset and defined transformations
        dataset = Food201Dataset('/home/hatsunemiku/dev/mask-rcnn/data/food201', get_transform(train=True))
        dataset_test = Food201Dataset('/home/hatsunemiku/dev/mask-rcnn/data/food201', get_transform(train=False), False)

    elif args.dataset == 'chfood':
        num_classes = 24
        # use our dataset and defined transformations
        dataset = CHFoodDataset('/home/hatsunemiku/dev/mask-rcnn/data/ch_food', get_transform(train=True))
        dataset_test = CHFoodDataset('/home/hatsunemiku/dev/mask-rcnn/data/ch_food', get_transform(train=False), False)
    else:
        raise Exception()
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=10,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 30

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        print('epoch %d, lr -----------> %.6f'%(epoch,optimizer.param_groups[0]['lr']))
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save(model.state_dict(), 'chfood_model.pkl')

if __name__ == "__main__":
    main()