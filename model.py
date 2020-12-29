import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch
import torch.nn.functional as F

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_model_partially_supervised_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    mlp = bbox2mask_weight_transfer(5120, 1024, 256, 3)
    class_embed = concat_cls_score_bbox_pred(model.roi_heads.box_predictor, num_classes)
    mask_w_flat = mlp(class_embed)

    # mask_w has shape (num_cls, dim_out, 1, 1)
    mask_w = mask_w_flat.unsqueeze(-1).unsqueeze(-1)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    model.roi_heads.mask_predictor.mask_fcn_logits.apply(conv_init(mask_w))

    return model


def conv_init(conv, weight):
    conv.weight.data = weight


def concat_cls_score_bbox_pred(box_predictor, num_classes):
    # flatten 'bbox_pred_w'
    # bbox_pred_w has shape (324, 1024), where 324 is (81, 4) memory structure
    # reshape to (81, 4 * 1024)

    bbox_pred_w = box_predictor.bbox_pred.weight.reshape(num_classes, -1)
    cls_score_w = box_predictor.cls_score.weight
    cls_score_bbox_pred = torch.cat([cls_score_w, bbox_pred_w], 1)
    return cls_score_bbox_pred


def bbox2mask_weight_transfer(dim_in, dim_h, dim_out, mlp_layers=1):
    if mlp_layers == 1:
        mlp = torch.nn.Linear(dim_in, dim_out)
    elif mlp_layers == 2:
        mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )
    elif mlp_layers == 3:
        mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )
    else:
        raise ValueError('unknown mlp_layers {}'.format(mlp_layers))
    
    return mlp
