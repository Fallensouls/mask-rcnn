import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import types  # for bound new forward function for RoIHeads
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.models.detection.roi_heads as roi_heads
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import (fastrcnn_loss,
                                                    keypointrcnn_inference,
                                                    keypointrcnn_loss,
                                                    maskrcnn_inference,
                                                    maskrcnn_loss)
from torchvision.ops import boxes as box_ops
from torchvision.ops import roi_align
import functools

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # FL = FocalLoss(gamma=2, alpha=0.25, magnifier=1)  
    # FL = FocalLoss(gamma=2, alpha=0.5)  
    # FL_wrapped = functools.partial(maskrcnn_loss_focal, focal_loss_func=FL)
    

    # RoIHeads_loss_customized.set_customized_loss(
    #     model.roi_heads, maskrcnn_loss_customized=FL_wrapped
    # )
    # RoIHeads_loss_customized.update_forward_func(model.roi_heads)

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

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=0.5, eps=1e-7, magnifier=1.0, from_logits=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.eps = eps
#         self.alpha = alpha
#         self.magnifier = magnifier
#         assert self.alpha >= 0
#         assert self.alpha <= 1
#         assert self.magnifier > 0
#         self.from_logits = from_logits

#     def forward(self, input, target):
#         if self.from_logits:
#             input = torch.sigmoid(input)

#         y = target
#         not_y = 1 - target

#         y_hat = input
#         not_y_hat = 1 - input

#         y_hat = y_hat.clamp(self.eps, 1.0 - self.eps)
#         not_y_hat = not_y_hat.clamp(self.eps, 1.0 - self.eps)

#         loss = (
#             -1 * self.alpha * not_y_hat ** self.gamma * y * torch.log(y_hat)
#         )  # cross entropy
#         loss += (
#             -1 * (1 - self.alpha) * y_hat ** self.gamma * not_y * torch.log(not_y_hat)
#         )
#         loss *= self.magnifier

#         return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=-1, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class RoIHeads_loss_customized(roi_heads.RoIHeads):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        maskrcnn_loss_customized=None,
        fastrcnn_loss_customized=None,
        keypointrcnn_loss_customized=None,
    ):
        super(RoIHeads_loss_customized, self).__init__()

        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(
            fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False
        )

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
            batch_size_per_image, positive_fraction
        )

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0)
        self.box_coder = det_utils.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.keypoint_roi_pool = keypoint_roi_pool
        self.keypoint_head = keypoint_head
        self.keypoint_predictor = keypoint_predictor

        self.maskrcnn_loss_customized = maskrcnn_loss_customized
        self.fastrcnn_loss_customized = fastrcnn_loss_customized
        self.keypointrcnn_loss_customized = keypointrcnn_loss_customized

    @staticmethod
    def set_customized_loss(
        head,
        maskrcnn_loss_customized=None,
        fastrcnn_loss_customized=None,
        keypointrcnn_loss_customized=None,
    ):
        head.maskrcnn_loss_customized = maskrcnn_loss_customized
        head.fastrcnn_loss_customized = fastrcnn_loss_customized
        head.keypointrcnn_loss_customized = keypointrcnn_loss_customized

    @staticmethod
    def update_forward_func(head):
        head.forward = types.MethodType(
            RoIHeads_loss_customized.forward, head
        )  # bound the method to our head

    # def forward(self, features, proposals, image_shapes, targets=None):
    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Args:
          features: List
          proposals: List
          image_shapes: List
          targets: List (Default value = None)
        Returns:
        """
        maskrcnn_loss_func = maskrcnn_loss
        fastrcnn_loss_func = fastrcnn_loss
        keypointrcnn_loss_func = keypointrcnn_loss

        eval_when_train = not self.training
        try:
            if self._eval_when_train:
                eval_when_train = True
        except AttributeError:
            pass

        if self.maskrcnn_loss_customized is not None:
            maskrcnn_loss_func = self.maskrcnn_loss_customized
        if self.fastrcnn_loss_customized is not None:
            fastrcnn_loss_func = self.fastrcnn_loss_customized
        if self.keypointrcnn_loss_customized is not None:
            keypointrcnn_loss_func = self.keypointrcnn_loss_customized

        if self.training:
            (
                proposals,
                matched_idxs,
                labels,
                regression_targets,
            ) = self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss_func(
                class_logits, box_regression, labels, regression_targets
            )
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        if eval_when_train:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(dict(boxes=boxes[i], labels=labels[i], scores=scores[i],))

        if self.has_mask:
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
            mask_features = self.mask_head(mask_features)
            mask_logits = self.mask_predictor(mask_features)

            loss_mask = {}
            if self.training:
                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                loss_mask = maskrcnn_loss_func(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )
                loss_mask = dict(loss_mask=loss_mask)
            if eval_when_train:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        if self.has_keypoint():
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.nonzero(labels[img_id] > 0).squeeze(1)
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])

            keypoint_features = self.keypoint_roi_pool(
                features, keypoint_proposals, image_shapes
            )
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                gt_keypoints = [t["keypoints"] for t in targets]
                loss_keypoint = keypointrcnn_loss_func(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = dict(loss_keypoint=loss_keypoint)
            if eval_when_train:
                keypoints_probs, kp_scores = keypointrcnn_inference(
                    keypoint_logits, keypoint_proposals
                )
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        return result, losses

def project_masks_on_boxes(gt_masks, boxes, matched_idxs, M):
    """Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.
    Args:
      gt_masks: 
      boxes: 
      matched_idxs: 
      M: 
    Returns:
    """
    matched_idxs = matched_idxs.to(boxes)
    rois = torch.cat([matched_idxs[:, None], boxes], dim=1)
    gt_masks = gt_masks[:, None].to(rois)
    return roi_align(gt_masks, rois, (M, M), 1)[:, 0]


def maskrcnn_loss_focal(
    mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs, focal_loss_func=None
):
    """
    Args:
      proposals: list
      mask_logits: Tensor
      targets: list
      gt_masks: 
      gt_labels: 
      mask_matched_idxs: 
      focal_loss_func: (Default value = None)
    Returns:
      Tensor: scalar tensor containing the loss
    """

    discretization_size = mask_logits.shape[-1]
    labels = [l[idxs] for l, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets = [
        project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0

    loss_func = F.binary_cross_entropy_with_logits
    if focal_loss_func is not None:
        loss_func = focal_loss_func

    mask_loss = loss_func(
        mask_logits[torch.arange(labels.shape[0], device=labels.device), labels],
        mask_targets,
    )
    return mask_loss