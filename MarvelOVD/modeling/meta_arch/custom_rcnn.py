from typing import Dict, List, Optional, Tuple
import torch
import torchvision
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.structures import Instances, Boxes, ImageList, pairwise_iou
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from torch.cuda.amp import autocast
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.layers import cat
from torch.nn import functional as F

@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results, _ = self.roi_heads(images, features, proposals, None)

            # # -------- 这里开始统计 proposal vs GT --------

            # 建议在 __init__ 中先初始化这些统计量
            # self.recall_k = 100   # 或 300 / 1000
            # self.iou_thresh = 0.5
            # self.base_gt_count = 0
            # self.base_hit_count = 0
            # self.novel_gt_count = 0
            # self.novel_hit_count = 0

            # for input_per_image, proposal_per_image in zip(batched_inputs, proposals):
            #     image_id = input_per_image["image_id"]
            #
            #     ann_ids = self.coco_api.getAnnIds(imgIds=[image_id], iscrowd=None)
            #     anns = self.coco_api.loadAnns(ann_ids)
            #
            #     base_gt_boxes = []
            #     novel_gt_boxes = []
            #
            #     for ann in anns:
            #         x, y, w, h = ann["bbox"]
            #         box_xyxy = [x, y, x + w, y + h]
            #
            #         if ann["thing_isNovel"]:
            #             novel_gt_boxes.append(box_xyxy)
            #         else:
            #             base_gt_boxes.append(box_xyxy)
            #
            #     prop_boxes_all = proposal_per_image.proposal_boxes
            #     if len(prop_boxes_all) == 0:
            #         continue
            #
            #     device = prop_boxes_all.tensor.device
            #
            #     if len(base_gt_boxes) > 0:
            #         self.base_gt_count += len(base_gt_boxes)
            #         base_gt_boxes_tensor = Boxes(
            #             torch.tensor(base_gt_boxes, dtype=torch.float32, device=device)
            #         )
            #
            #         for k in self.recall_ks:
            #             K = min(k, len(prop_boxes_all))
            #             prop_boxes = Boxes(prop_boxes_all.tensor[:K])
            #
            #             iou_matrix = pairwise_iou(base_gt_boxes_tensor, prop_boxes)
            #             max_ious, _ = iou_matrix.max(dim=1)
            #             hits = (max_ious >= self.iou_thresh).sum().item()
            #
            #             self.base_hit_count_dict[k] += hits
            #
            #     if len(novel_gt_boxes) > 0:
            #         self.novel_gt_count += len(novel_gt_boxes)
            #         novel_gt_boxes_tensor = Boxes(
            #             torch.tensor(novel_gt_boxes, dtype=torch.float32, device=device)
            #         )
            #
            #         for k in self.recall_ks:
            #             K = min(k, len(prop_boxes_all))
            #             prop_boxes = Boxes(prop_boxes_all.tensor[:K])
            #
            #             iou_matrix = pairwise_iou(novel_gt_boxes_tensor, prop_boxes)
            #             max_ious, _ = iou_matrix.max(dim=1)
            #             hits = (max_ious >= self.iou_thresh).sum().item()
            #
            #             self.novel_hit_count_dict[k] += hits

            # for input_per_image, proposal_per_image in zip(batched_inputs, proposals):
            #     image_id = input_per_image["image_id"]
            #
            #     ann_ids = self.coco_api.getAnnIds(imgIds=[image_id], iscrowd=None)
            #     anns = self.coco_api.loadAnns(ann_ids)
            #     gt_boxes = []
            #     for ann in anns:
            #         if ann['thing_isNovel']:
            #         # COCO bbox: [x, y, w, h] -> [x1, y1, x2, y2]
            #             x, y, w, h = ann["bbox"]
            #             gt_boxes.append([x, y, x + w, y + h])
            #             self.count += 1
            #     if len(gt_boxes) == 0:
            #         continue
            #     prop_boxes = proposal_per_image.proposal_boxes
            #     if len(prop_boxes) == 0:
            #         continue
            #     device = prop_boxes.tensor.device
            #     gt_boxes = Boxes(torch.tensor(gt_boxes, dtype=torch.float32, device=device))
            #     # [num_gt, num_prop]
            #     iou_matrix = pairwise_iou(gt_boxes, prop_boxes)
            #     # 每个GT的最大IoU
            #     max_ious, _ = iou_matrix.max(dim=1)
            #     self.rpn_gt_max_ious.extend(max_ious.detach().cpu().tolist())
            # # -------- 统计结束 - -------

        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def forward(self, batched_inputs, style = None):
        # batched_inputs: List[Dict[str, torch.Tensor]]
        
        if not self.training:
            return self.inference(batched_inputs)
        
        images, weak_images = self.preprocess_all_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        features = self.backbone(images.tensor)
        # for k, v in features.items():
        #     print(k, v.shape)
        # p2
        # torch.Size([32, 256, 200, 296])
        # p3
        # torch.Size([32, 256, 100, 148])
        # p4
        # torch.Size([32, 256, 50, 74])
        # p5
        # torch.Size([32, 256, 25, 37])
        # p6
        # torch.Size([32, 256, 13, 19])
        # print(self-1)
        if style=="burn-in":
            filtered_gt_instances = self.filter_pseudo_labels(gt_instances)
            filtered_gt_instances = filtered_gt_instances + filtered_gt_instances
            proposals, proposal_losses = self.proposal_generator(images, features, filtered_gt_instances)
            # for i, p in enumerate(proposals):
            #     print(f"Image {i}: proposal_boxes = {p.proposal_boxes.tensor.shape}")
            # print(self-1)
            proposals, detector_losses = self.roi_heads(images, features, proposals, filtered_gt_instances)
        else:
            with torch.no_grad():
                weak_features = self.backbone(weak_images.tensor)
            gt_instances = self.refine_pseudo_labels(weak_features, gt_instances)
            gt_instances = gt_instances + gt_instances
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
            for key in weak_features.keys():
                weak_features[key] = torch.cat([weak_features[key], weak_features[key]],dim=0)
            _, detector_losses = self.roi_heads.forward_weak_strong(images, features, weak_features, proposals, gt_instances)

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    
    def filter_pseudo_labels(self, gt_instances):
        filtered_gt_instances = []
        for gt_per_img in gt_instances:
            mask = gt_per_img.gt_confidence>=0.9
            new_instance = Instances(gt_per_img.image_size)
            new_instance.gt_boxes = gt_per_img.gt_boxes[mask]
            new_instance.gt_classes = gt_per_img.gt_classes[mask]
            new_instance.gt_confidence = gt_per_img.gt_confidence[mask]
            new_instance.gt_use_seg = gt_per_img.gt_use_seg[mask]
            new_instance.gt_is_base = gt_per_img.gt_is_base[mask]
            new_instance.gt_clip_score = gt_per_img.gt_clip_score[mask]
            filtered_gt_instances.append(new_instance)
        return filtered_gt_instances



    

    @torch.no_grad()
    def refine_pseudo_labels(self, weak_features, gt_instances):
        # preds_on_gt = self.roi_heads.get_preds_for_pseudo_labels(weak_features, [x.gt_boxes for x in gt_instances])
        preds_on_gt = self.roi_heads.get_preds_from_weak_features(weak_features, [x.gt_boxes for x in gt_instances])
        refined_gt_instances = []
        for preds_per_image, gt_per_img in zip(preds_on_gt, gt_instances):
            
            base_mask = gt_per_img.gt_is_base>0
            novel_scores = preds_per_image[:,self.roi_heads.box_predictor.novel].sum(-1)
            if novel_scores.numel()==0:
                refined_gt_instances.append(gt_per_img)
                continue
            ratio = novel_scores.max()
            new_confidence = (gt_per_img.gt_clip_score + (novel_scores / ratio))*0.5
           
            new_confidence[base_mask] = 1

            mask = new_confidence>=0.9
            new_instance = Instances(gt_per_img.image_size)
            new_instance.gt_boxes = gt_per_img.gt_boxes[mask]
            new_instance.gt_classes = gt_per_img.gt_classes[mask]
            new_instance.gt_confidence = new_confidence[mask]
            new_instance.gt_use_seg = gt_per_img.gt_use_seg[mask]
            new_instance.gt_is_base = gt_per_img.gt_is_base[mask]
            new_instance.gt_clip_score = gt_per_img.gt_clip_score[mask]
            refined_gt_instances.append(new_instance)
        return refined_gt_instances


    def forward_simple(self, batched_inputs):
        images, _ = self.preprocess_all_image(batched_inputs)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        # gt_instances = self.filter_pseudo_labels(gt_instances)
        gt_instances = gt_instances + gt_instances

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)        
        proposals, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        # proposals, detector_losses = self.roi_heads()

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


    def preprocess_all_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        # images_strong = [self._move_to_current_device(x["image_strong"]) for x in batched_inputs]
        images_strong = [x["image_strong"].to(self.device) for x in batched_inputs]
        images_strong = [(x - self.pixel_mean) / self.pixel_std for x in images_strong]
        # images_weak = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images_weak = [x["image"].to(self.device) for x in batched_inputs]
        images_weak = [(x - self.pixel_mean) / self.pixel_std for x in images_weak]
        # print(self.backbone)
        images = ImageList.from_tensors(
            images_weak+images_strong,
            self.backbone.size_divisibility,
            # padding_constraints=self.backbone.padding_constraints,
        )
        weak_images = ImageList.from_tensors(
            images_weak,
            self.backbone.size_divisibility,
            # padding_constraints=self.backbone.padding_constraints,
        )
        return images, weak_images
