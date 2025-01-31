#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
import copy
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() > comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


####
def gaussian_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply Gaussian blur to input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    ksize = random_state.randint(0, max_ksize, size=(2,))
    ksize = tuple((ksize * 2 + 1).tolist())

    ret = cv2.GaussianBlur(
        img, ksize, sigmaX=0, sigmaY=0, borderType=cv2.BORDER_REPLICATE
    )
    ret = np.reshape(ret, img.shape)
    ret = ret.astype(np.uint8)
    return [ret]


####
def median_blur(images, random_state, parents, hooks, max_ksize=3):
    """Apply median blur to input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    ksize = random_state.randint(0, max_ksize)
    ksize = ksize * 2 + 1
    ret = cv2.medianBlur(img, ksize)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_hue(images, random_state, parents, hooks, range=None):
    """Perturbe the hue of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    hue = random_state.uniform(*range)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if hsv.dtype.itemsize == 1:
        # OpenCV uses 0-179 for 8-bit images
        hsv[..., 0] = (hsv[..., 0] + hue) % 180
    else:
        # OpenCV uses 0-360 for floating point images
        hsv[..., 0] = (hsv[..., 0] + 2 * hue) % 360
    ret = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_saturation(images, random_state, parents, hooks, range=None):
    """Perturbe the saturation of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = 1 + random_state.uniform(*range)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret = img * value + (gray * (1 - value))[:, :, np.newaxis]
    ret = np.clip(ret, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_contrast(images, random_state, parents, hooks, range=None):
    """Perturbe the contrast of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    ret = img * value + mean * (1 - value)
    ret = np.clip(img, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


####
def add_to_brightness(images, random_state, parents, hooks, range=None):
    """Perturbe the brightness of input images."""
    img = images[0]  # aleju input batch as default (always=1 in our case)
    value = random_state.uniform(*range)
    ret = np.clip(img + value, 0, 255)
    ret = ret.astype(np.uint8)
    return [ret]


def customed_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    # image
    image, transforms = T.apply_transform_gens([
        T.RandomApply(T.RandomCrop(crop_type="relative_range", crop_size=(0.9, 0.9)), prob=0.50),
        T.RandomApply(T.Resize(shape=(512, 512)), prob=1.0),
        T.RandomFlip(prob=0.50, horizontal=True, vertical=False),
        T.RandomFlip(prob=0.50, horizontal=False, vertical=True),
        # # T.RandomApply(transform=T.RandomRotation(angle=[-30,30], expand=True, center=None, sample_style="range", interp=None), prob=0.20)
        # T.RandomApply(transform=T.RandomBrightness(intensity_min=0.75, intensity_max=1.25), prob=0.20),
        # T.RandomApply(transform=T.RandomContrast(intensity_min=0.75, intensity_max=1.25), prob=0.20),
        # T.RandomApply(transform=T.RandomSaturation(intensity_min=0.75, intensity_max=1.25), prob=0.20),
    ], image)

    input_augs = iaa.Sequential([
        iaa.OneOf([
            iaa.Lambda(
                seed=0,
                func_images=lambda *args: gaussian_blur(*args, max_ksize=3),
            ),
            iaa.Lambda(
                seed=0,
                func_images=lambda *args: median_blur(*args, max_ksize=3),
            ),
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
            ),
        ]),
        iaa.Sequential([
            iaa.Lambda(
                seed=0,
                func_images=lambda *args: add_to_hue(*args, range=(-8, 8)),
            ),
            iaa.Lambda(
                seed=0,
                func_images=lambda *args: add_to_saturation(*args, range=(-0.2, 0.2)),
            ),
            iaa.Lambda(
                seed=0,
                func_images=lambda *args: add_to_brightness(*args, range=(-26, 26)),
            ),
            iaa.Lambda(
                seed=0,
                func_images=lambda *args: add_to_contrast(*args, range=(0.75, 1.25)),
            ),
        ], random_order=True,),
    ])
    input_augs = input_augs.to_deterministic()
    image = input_augs.augment_image(image) 

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    # annos
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=customed_mapper)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model, fold=args.fold, step=args.step)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
