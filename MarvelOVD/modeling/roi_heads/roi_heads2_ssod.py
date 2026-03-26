import numpy as np
import torch
from torch import nn
from typing import Dict, List, Tuple, Optional
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.config import configurable
from .fast_rcnn_ssod import SSODFastRCNNOutputLayers
from ..matcher import Matcher2
from detectron2.modeling.sampling import subsample_labels
from detectron2.layers import cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from torch.nn import functional as F
import torch.nn.functional as F
import math

@ROI_HEADS_REGISTRY.register()
class SSODROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
             
        concept_path = 'meta_datasets/coco_description.pkl'
        concept_pkl = torch.load(concept_path, map_location="cuda")
        fine_concept_feats = concept_pkl['all_noun_chunks']['fine_text_features']
        fine_concept_feats = fine_concept_feats.to(torch.float32).contiguous()
        coarse_concept_feats = concept_pkl['all_noun_chunks']['coarse_text_features']
        coarse_concept_feats = coarse_concept_feats.to(torch.float32).contiguous()
        fine_concept_feats = F.normalize(fine_concept_feats, p=2, dim=1)
        coarse_concept_feats = F.normalize(coarse_concept_feats, p=2, dim=1)
        
        self.register_buffer('fine_concept_feats', fine_concept_feats)
        self.register_buffer('coarse_concept_feats', coarse_concept_feats)

        import clip
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
        from ..lgva import LGVA
        lgva_model = LGVA()
        lgva_model.load_state_dict(torch.load("meta_datasets/lgva_coco.pth", map_location='cpu'))
        lgva_model.eval()
        for param in lgva_model.parameters():
            param.requires_grad = False
        self.lgva_model = lgva_model
        self.loss_reg_fn = nn.MSELoss(reduction='mean')

    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictor']
        ret['box_predictor'] = SSODFastRCNNOutputLayers(cfg, ret['box_head'].output_shape)
        return ret

    @classmethod
    def from_config(self, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        del ret['proposal_matcher']
        ret['proposal_matcher'] = Matcher2(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )
        return ret

    def forward_strong(
        self,
        strong_features: Dict[str, torch.Tensor],
        weak_features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        
        assert targets, "'targets' argument is required during training"
        proposals = self.label_and_sample_proposals(proposals, targets)

        with torch.no_grad():
            cls_preds = self.get_preds_from_weak_features(weak_features, proposals)
            for preds_per_image, proposal_per_img in zip(cls_preds, proposals):
                old_confidence = proposal_per_img.gt_confidence
                novel_scores = preds_per_image[:,self.box_predictor.novel].sum(-1)
                new_confidence = old_confidence + novel_scores
                gt_mask = (proposal_per_img.gt_is_base>0) & proposal_per_img.gt_classes<self.box_predictor.num_classes
                # gt_mask = old_confidence>=1
                new_confidence[gt_mask] = 1
                proposal_per_img.gt_confidence = new_confidence

        losses = self._forward_box(strong_features, proposals)
        # Usually the original proposals used by the box head are used by the mask, keypoint
        # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
        # predicted by the box head.
        losses.update(self._forward_mask(strong_features, proposals))
        losses.update(self._forward_keypoint(strong_features, proposals))
        return proposals, losses

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances], images=None, weak_features=None, cls_preds=None, diss_aug=False):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features) # 转到 fast_rcnn_ssod1 166 forward

        if self.training:
            if diss_aug:
                weak_features = [weak_features[f] for f in self.box_in_features]
                weak_features = self.box_pooler(weak_features, [x.proposal_boxes for x in proposals])
                weak_features = self.box_head(weak_features)
                predictions_weak, f_language = self.box_predictor(weak_features,
                                                             need_f=True)  # 转到 fast_rcnn_ssod1 166 forward
                with torch.no_grad():
                    scores = self.box_predictor.predict_probs(predictions_weak, proposals)
                    num_inst_per_image = [len(p) for p in proposals]
                    f_language = f_language.split(num_inst_per_image, dim=0)
                    images = images.tensor.to(scores[0].device)
                    bboxes = self.box_predictor.predict_boxes(predictions_weak, proposals)
                before_fg = []
                after_fg = []
                for i, (logits, img, boxes, proposal_per_img, before_box_features) in enumerate(zip(cls_preds, images, bboxes, proposals, f_language)):
                    bg_scores = logits[:, -1]
                    image_shape = proposal_per_img.image_size
                    fg_mask = bg_scores < 0.1
                    fg_high_prob_mask = fg_mask

                    valid_mask = torch.isfinite(boxes).all(dim=1)
                    if not valid_mask.all():
                        boxes = boxes[valid_mask]
                        fg_high_prob_mask = fg_high_prob_mask[valid_mask]

                    val_mask = fg_high_prob_mask
                    boxes = boxes[val_mask]
                    if boxes.numel() <= 0:
                        continue
                    num_bbox_reg_classes = boxes.shape[1] // 4
                    assert num_bbox_reg_classes == 1, 'error: num_bbox_reg_classes'
                    boxes = Boxes(boxes.reshape(-1, 4))
                    boxes.clip(image_shape)
                    boxes_int = boxes.tensor.to(int)
                    pre_crop = []
                    for box in boxes_int:
                        w = box[2] - box[0]
                        h = box[3] - box[1]
                        assert w >= 0 and h >= 0, 'error: width or height < 0'
                        assert box[2] <= image_shape[1], 'error: width overflow'
                        assert box[3] <= image_shape[0], 'error: height overflow'
                        if w < 1:
                            if box[0] >= 1:
                                box[0] -= 1
                            else:
                                box[2] += 1
                        if h < 1:
                            if box[1] >= 1:
                                box[1] -= 1
                            else:
                                box[3] += 1
                        crop = img[:, box[1]:box[3], box[0]:box[2]]
                        pre_crop.append(self.resize_and_pad(crop))
                    pre_crops = torch.stack(pre_crop, dim=0)
                    with torch.no_grad():
                        v_f = self.clip_model.encode_image(pre_crops).to(torch.float32)
                        v_f /= v_f.norm(dim=-1, keepdim=True)
                        after_box_features = self.lgva_model(v_f, self.fine_concept_feats, self.coarse_concept_feats)
                    norm_before_box_features = F.normalize(before_box_features[val_mask], dim=-1)
                    before_fg.append(norm_before_box_features)   # [N_fg, D]
                    after_fg.append(after_box_features)    # [N_fg, D]
                if len(before_fg) == 0:
                    loss_aug = f_language[0].sum() * 0.0
                else:
                    before_fg = torch.cat(before_fg, dim=0)
                    after_fg = torch.cat(after_fg, dim=0)
                    loss_aug = self.loss_reg_fn(before_fg, after_fg)

            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            if diss_aug:
                losses["loss_augmentation"] = (loss_aug) * 1.0
            else:
                losses["loss_augmentation"] = box_features[0].sum() * 0.0
            del box_features
            return losses
        else:
            del box_features
            pred_instances, _ = self.box_predictor.inference(predictions, proposals, images)
            return pred_instances

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        # if self.training:
            # del images

        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals, images)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals, images)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_weak_strong(
        self,
        images,
        features,
        weak_features,
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        assert targets, "'targets' argument is required during training"


        proposals = self.label_and_sample_proposals(proposals, targets)

        with torch.no_grad():
            cls_preds = self.get_preds_from_weak_features(weak_features, proposals)
            for preds_per_image, proposal_per_img in zip(cls_preds, proposals):
                bg_scores = preds_per_image[:,-1]
                old_confidence = proposal_per_img.gt_confidence
                new_confidence = ((1-bg_scores)*0.5 + old_confidence*0.5)*2
                gt_mask = (proposal_per_img.gt_is_base>0) & (proposal_per_img.gt_classes<self.box_predictor.num_classes)
                new_confidence[gt_mask] = 1
                proposal_per_img.gt_confidence = new_confidence


        losses = self._forward_box(features, proposals, images, weak_features=weak_features, cls_preds=cls_preds, diss_aug=True)
        # Usually the original proposals used by the box head are used by the mask, keypoint
        # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
        # predicted by the box head.
        losses.update(self._forward_mask(features, proposals))
        losses.update(self._forward_keypoint(features, proposals))
        return proposals, losses

    @torch.no_grad()
    def resize_and_pad(self, img, n_px=224):
        _, h, w = img.shape
        if h == 0: h = 1
        if w == 0: w = 1
        max_hw = max(h, w)
        new_h = math.ceil(h / max_hw * n_px)
        new_w = math.ceil(w / max_hw * n_px)
        resized = F.interpolate(img.unsqueeze(0).to(torch.float32), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0).to(torch.uint8)

        padding_h = n_px - new_h
        padding_w = n_px - new_w
        top = padding_h // 2
        bottom = padding_h - top
        left = padding_w // 2
        right = padding_w - left
        padded = F.pad(resized, pad=(left, right, top, bottom), mode='constant', value=0)

        padded = padded / 255
        px_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(img.device)
        px_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(img.device)
        padded = (padded - px_mean) / px_std
        return padded


    @torch.no_grad()
    def pred_boxes(
        self, predictions, boxes
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(boxes):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [p.tensor.size(0) for p in boxes]
        proposal_boxes = cat([p.tensor for p in boxes], dim=0)
        predict_boxes = self.box_predictor.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)


    @torch.no_grad()
    def get_preds_from_weak_features(self, features, proposals):
        features = [features[f] for f in self.box_in_features]
        # print(len(features))
        # print(proposals)
        try:
            box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        except:
            box_features = self.box_pooler(features, [x for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        scores = self.box_predictor.predict_probs(predictions, proposals)
        # scores: List[Tensor]
        return scores

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:

        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            # targets_per_image.keys(): gt_boxes; gt_classes; gt_confidence; gt_use_seg
            # 划分一下base物体和novel物体
            # novel_mask = targets_per_image.gt_confidence<1
            base_mask = targets_per_image.gt_is_base>0
            novel_mask = ~base_mask
            # base_boxes = targets_per_image.gt_boxes[base_mask]
            # novel_boxes = targets_per_image.gt_boxes[novel_mask]

            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # 修改一下proposal_matcher的操作，让其优先匹配base
            # matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix, base_mask, novel_mask)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )


            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.
            if not proposals_per_image.has('gt_confidence'):
                proposals_per_image.gt_confidence = torch.ones_like(gt_classes).float()
            if not proposals_per_image.has('gt_use_seg'):
                proposals_per_image.gt_use_seg = torch.ones_like(gt_classes).float()
            if not proposals_per_image.has('gt_is_base'):
                proposals_per_image.gt_is_base = torch.ones_like(gt_classes).float()

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def _forward_mask(self, features: Dict[str, torch.Tensor], instances: List[Instances]):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """

        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(features, instances)


def select_foreground_proposals(
        proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    assert proposals[0].has("gt_use_seg")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        gt_use_seg = proposals_per_image.gt_use_seg
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label) & (gt_use_seg > 0.)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks
