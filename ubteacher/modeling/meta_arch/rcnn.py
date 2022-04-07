# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from xml.sax.xmlreader import InputSource
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n
from copy import deepcopy


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results

@META_ARCH_REGISTRY.register()
class TwoStageDoubleHeadPseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        super(TwoStageDoubleHeadPseudoLabGeneralizedRCNN,self).__init__(backbone,proposal_generator,roi_heads,pixel_mean,pixel_std,input_format,vis_period)
        self.proposal_generator_a = proposal_generator
        self.proposal_generator_b = deepcopy(proposal_generator)
        self.roi_heads_a = roi_heads
        self.roi_heads_b = deepcopy(roi_heads)

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator_a(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads_a(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_strong":
            #split data
            features_a,features_b = features.chunk(2)
            images_a,images_b = images.chunk(2)
            gt_instances_a,gt_instances_b= gt_instances.chunk(2)

            proposals_rpn_a, proposal_losses_a = self.proposal_generator_a(
                images_a, features_a, gt_instances_a
            )
            _, detector_losses_a = self.roi_heads_a(
                images_a, features_a, proposals_rpn_a, gt_instances_a, branch=branch
            )
            proposals_rpn_b, proposal_losses_b = self.proposal_generator_b(
                images_b, features_b, gt_instances_b
            )
            _, detector_losses_b = self.roi_heads_b(
                images_b, features_b, proposals_rpn_b, gt_instances_b, branch=branch
            )

            losses_a = {}
            losses_a.update(detector_losses_a)
            losses_a.update(proposal_losses_a)
            losses_b = {}
            losses_b.update(detector_losses_b)
            losses_b.update(proposal_losses_b)
            return losses_a,losses_b, [], [], None



        elif branch == "unsup_data_weak":
            #split data
            features_a,features_b = features.chunk(2)
            images_a,images_b = images.chunk(2)
            # Region proposal network
            proposals_rpn_a, _ = self.proposal_generator_b(
                images_a, features_a, None, compute_loss=False
            )
            proposals_rpn_b, _ = self.proposal_generator_a(
                images_b, features_b, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih_a, ROI_predictions_a = self.roi_heads_b(
                images_a,
                features_a,
                proposals_rpn_a,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            proposals_roih_b, ROI_predictions_b = self.roi_heads_a(
                images_b,
                features_b,
                proposals_rpn_b,
                targets=None,
                compute_loss=False,
                branch=branch,
            )
            proposals_rpn = torch.cat((proposals_rpn_a, proposals_rpn_b),0)
            proposals_roih = torch.cat((proposals_roih_a,proposals_roih_b),0)
            ROI_predictions = torch.cat((ROI_predictions_a,ROI_predictions_b),0)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
            #return {}, (proposals_rpn_a, proposals_rpn_b), (proposals_roih_a,proposals_roih_b), (ROI_predictions_a,ROI_predictions_b)

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator_a(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads_a(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None