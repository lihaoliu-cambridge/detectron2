# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
import pandas as pd
import numpy as np
import cv2
import pathlib
import shutil
from sklearn.metrics import r2_score
from scipy.optimize import linear_sum_assignment

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def _in_bbox(target_bbox, current_pos=[32, 32, 480, 480]): 
    # bbox (x0, top, x1, bottom)
    # 获取两bbox中心的坐标
    mid_x = abs((target_bbox[0] + target_bbox[2]) / 2 - (current_pos[0] + current_pos[2]) / 2)
    mid_y = abs((target_bbox[1] + target_bbox[3]) / 2 - (current_pos[1] + current_pos[3]) / 2)

    # 获取两bbox边长之和
    width  = abs(target_bbox[0] - target_bbox[2]) + abs(current_pos[0] - current_pos[2])
    height = abs(target_bbox[1] - target_bbox[3]) + abs(current_pos[1] - current_pos[3])

    if (mid_x <= width/2 and mid_y <= height/2):
        return True
    else:
        return False


def get_multi_r2(true, pred):
    """Get the correlation of determination for each class and then 
    average the results.
    
    Args:
        true (pd.DataFrame): dataframe indicating the nuclei counts for each image and category.
        pred (pd.DataFrame): dataframe indicating the nuclei counts for each image and category.
    
    Returns:
        multi class coefficient of determination
        
    """
    # first check to make sure that the appropriate column headers are there
    class_names = [
        "neutrophil",
        "eosinophil",
    ]
    for col in true.columns:
        if col not in class_names:
            raise ValueError("%s column header not recognised")

    for col in pred.columns:
        if col not in class_names:
            raise ValueError("%s column header not recognised")

    # for each class, calculate r2 and then take the average
    r2_list = []
    for class_ in class_names:
        true_oneclass = true[class_].tolist()
        pred_oneclass = pred[class_].tolist()
        r2_list.append(r2_score(true_oneclass, pred_oneclass))
    print(r2_list)

    return np.mean(np.array(r2_list))


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred (ndarray): the 2d array contain instances where each instances is marked
            by non-zero integer.
        by_size (bool): renaming such that larger nuclei have a smaller id (on-top).

    Returns:
        new_pred (ndarray): Array with continguous ordering of instances.

    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred


def get_bounding_box(img):
    """Get the bounding box coordinates of a binary input- assumes a single object.

    Args:
        img: input binary image.

    Returns:
        bounding box coordinates

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def get_multi_pq_info(true, pred, nr_classes=2, match_iou=0.5):
    """Get the statistical information needed to compute multi-class PQ.
    
    CoNIC multiclass PQ is achieved by considering nuclei over all images at the same time, 
    rather than averaging image-level results, like was done in MoNuSAC. This overcomes issues
    when a nuclear category is not present in a particular image.
    
    Args:
        true (ndarray): HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map. 
        pred: HxWx2 array. First channel is the instance segmentation map
            and the second channel is the classification map. 
        nr_classes (int): Number of classes considered in the dataset. 
        match_iou (float): IoU threshold for determining whether there is a detection.
    
    Returns:
        statistical info per class needed to compute PQ.
    
    """

    assert match_iou >= 0.0, "Cant' be negative"

    true_inst = true[..., 0]
    pred_inst = pred[..., 0]
    ###
    true_class = true[..., 1]
    pred_class = pred[..., 1]

    pq = []
    for idx in range(nr_classes):
        pred_class_tmp = pred_class == idx + 1
        pred_inst_oneclass = pred_inst * pred_class_tmp
        pred_inst_oneclass = remap_label(pred_inst_oneclass)
        ##
        true_class_tmp = true_class == idx + 1
        true_inst_oneclass = true_inst * true_class_tmp
        true_inst_oneclass = remap_label(true_inst_oneclass)

        pq_oneclass_info = get_pq(true_inst_oneclass, pred_inst_oneclass, remap=False)

        # add (in this order) tp, fp, fn iou_sum
        pq_oneclass_stats = [
            pq_oneclass_info[1][0],
            pq_oneclass_info[1][1],
            pq_oneclass_info[1][2],
            pq_oneclass_info[2],
        ]
        pq.append(pq_oneclass_stats)
    print(pq_oneclass_stats)
    return pq


def get_pq(true, pred, match_iou=0.5, remap=True):
    """Get the panoptic quality result. 
    
    Fast computation requires instance IDs are in contiguous orderding i.e [1, 2, 3, 4] 
    not [2, 3, 6, 10]. Please call `remap_label` beforehand. Here, the `by_size` flag 
    has no effect on the result.

    Args:
        true (ndarray): HxW ground truth instance segmentation map
        pred (ndarray): HxW predicted instance segmentation map
        match_iou (float): IoU threshold level to determine the pairing between
            GT instances `p` and prediction instances `g`. `p` and `g` is a pair
            if IoU > `match_iou`. However, pair of `p` and `g` must be unique 
            (1 prediction instance to 1 GT instance mapping). If `match_iou` < 0.5, 
            Munkres assignment (solving minimum weight matching in bipartite graphs) 
            is caculated to find the maximal amount of unique pairing. If 
            `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
            the number of pairs is also maximal.  
        remap (bool): whether to ensure contiguous ordering of instances.
    
    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]: 
                      pairing information to perform measurement
        
        paired_iou.sum(): sum of IoU within true positive predictions
                    
    """
    assert match_iou >= 0.0, "Cant' be negative"
    # ensure instance maps are contiguous
    if remap:
        pred = remap_label(pred)
        true = remap_label(true)

    true = np.copy(true)
    pred = np.copy(pred)
    true = true.astype("int32")
    pred = pred.astype("int32")
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))
    # prefill with value
    pairwise_iou = np.zeros([len(true_id_list), len(pred_id_list)], dtype=np.float64)

    # caching pairwise iou
    for true_id in true_id_list[1:]:  # 0-th is background
        t_mask_lab = true == true_id
        rmin1, rmax1, cmin1, cmax1 = get_bounding_box(t_mask_lab)
        t_mask_crop = t_mask_lab[rmin1:rmax1, cmin1:cmax1]
        t_mask_crop = t_mask_crop.astype("int")
        p_mask_crop = pred[rmin1:rmax1, cmin1:cmax1]
        pred_true_overlap = p_mask_crop[t_mask_crop > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  # ignore
                continue  # overlaping background
            p_mask_lab = pred == pred_id
            p_mask_lab = p_mask_lab.astype("int")

            # crop region to speed up computation
            rmin2, rmax2, cmin2, cmax2 = get_bounding_box(p_mask_lab)
            rmin = min(rmin1, rmin2)
            rmax = max(rmax1, rmax2)
            cmin = min(cmin1, cmin2)
            cmax = max(cmax1, cmax2)
            t_mask_crop2 = t_mask_lab[rmin:rmax, cmin:cmax]
            p_mask_crop2 = p_mask_lab[rmin:rmax, cmin:cmax]

            total = (t_mask_crop2 + p_mask_crop2).sum()
            inter = (t_mask_crop2 * p_mask_crop2).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou

    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  # index is instance id - 1
        paired_pred += 1  # hence return back to original
    else:  # * Exhaustive maximal unique pairing
        #### Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensure
        # inverse pair to get high IoU as minimum
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        ### extract the paired cost and remove invalid pair
        paired_iou = pairwise_iou[paired_true, paired_pred]

        # now select those above threshold level
        # paired with iou = 0.0 i.e no intersection => FP or FN
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    # get the actual FP and FN
    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
    # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

    #
    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    # get the F1-score i.e DQ
    dq = tp / ((tp + 0.5 * fp + 0.5 * fn) + 1.0e-6)
    # get the SQ, no paired has 0 iou so not impact
    sq = paired_iou.sum() / (tp + 1.0e-6)

    return (
        [dq, sq, dq * sq],
        [tp, fp, fn],
        paired_iou.sum(),
    )

def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()

        segment_flag = False
        all_counts = []
        segmentation_results = np.zeros((1018, 256, 256, 2))
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            # print(inputs[0]["height"])
            outputs = model(inputs)
            # print("-", len(outputs[0]), len(outputs[0]['instances'].pred_boxes), len(outputs[0]['instances'].scores), len(outputs[0]['instances'].pred_classes))
            pred_results = [0, 0]
            # print(outputs[0]['instances'].pred_boxes.tensor.size())
            
            for r_idx in range(outputs[0]['instances'].pred_boxes.tensor.size(0)):
                single_bbox = outputs[0]['instances'].pred_boxes.tensor[r_idx, :]
                r_score = outputs[0]['instances'].scores[r_idx]
                # print(torch.unique(outputs[0]['instances'].pred_classes))

                if r_score >= 0.5 and _in_bbox(target_bbox=single_bbox[:4], current_pos=[32, 32, 480, 480]):  # [16, 16, 240, 240]
                    pred_results[int(outputs[0]['instances'].pred_classes[r_idx])] += 1

                segment_flag = outputs[0]['instances'].has("pred_masks")
                if segment_flag:
                    single_instance_mask = outputs[0]['instances'].pred_masks[r_idx, :]
                    if r_score >= 0.5:
                        single_mask = single_instance_mask.cpu().numpy()
                        # print(single_mask.shape, single_mask.min(), single_mask.max())
                        instance_part_resized = cv2.resize((single_mask*255).astype('uint8'), (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.bool)
                        # print(instance_part_resized.shape, instance_part_resized.min(), instance_part_resized.max())

                        segmentation_results[idx, ..., 0][instance_part_resized] = r_idx+1
                        segmentation_results[idx, ..., 1][instance_part_resized] = (int(outputs[0]['instances'].pred_classes[r_idx])+1)
            
            all_counts.append(pred_results)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

        all_counts_np = np.asarray(all_counts)

        df = pd.DataFrame(data=all_counts_np, columns=["neutrophil", "eosinophil"])
        df.to_csv("/home/ll610/Onepiece/code_cs138/github/mmdetection/conic/dataset/conic/counts_pred_15.csv", index=False)

        # middle_seg = segmentation_results[:, 16:240, 26:240, :]

        # all_counts_2 = []
        # for i in range(middle_seg.shape[0]):
        #     counts_2 = [0,0,0,0,0,0]

        #     instance_and_type = middle_seg[i]
        #     instance_map = instance_and_type[..., 0]
        #     type_map = instance_and_type[..., 1]
            
        #     instance_ids = np.unique(instance_map)
        #     # if len(instance_ids) == 1:
        #     #     print("---", idx)
        #     for instance_id in instance_ids:
        #         if instance_id == 0:
        #             continue

        #         # category_id
        #         instance_part = (instance_map == instance_id)
        #         category_ids_in_instance = np.unique(type_map[instance_part])
        #         assert len(category_ids_in_instance) == 1
        #         category_id = int(category_ids_in_instance[0])
        #         if category_id > 6 or category_id == 0:
        #             raise Exception("Only 6 types")

        #         counts_2[category_id-1] += 1
        #     all_counts_2.append(counts_2)
        
        # all_counts_2_np = np.asarray(all_counts_2)
        # df = pd.DataFrame(data=all_counts_2_np, columns=["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"])
        # df.to_csv("/home/ll610/Onepiece/code_cs138/github/mmdetection/conic/dataset/conic/counts_pred.csv", index=False)

        pred_csv = pd.read_csv("/home/ll610/Onepiece/code_cs138/github/mmdetection/conic/dataset/conic/counts_pred_15.csv")
        true_csv = pd.read_csv("/home/ll610/Onepiece/code_cs138/github/mmdetection/conic/dataset/conic/counts_valid_15.csv")
        all_metrics = {}
        reg_metrics_names = ["r2"]
        for idx, metric in enumerate(reg_metrics_names):
            if metric == "r2":
                # calculate multiclass coefficient of determination
                r2 = get_multi_r2(true_csv, pred_csv)
                all_metrics["multi_r2"] = [r2]
            else:
                raise ValueError("%s is not supported!" % metric)
        df = pd.DataFrame(all_metrics)
        df = df.to_string(index=False)
        print(df)
        with open("log_15.txt", "a") as text_file:
            text_file.write("\n"+str(df))

        if segment_flag:
            np.save(f'/home/ll610/Onepiece/code_cs138/github/mmdetection/conic/dataset/conic/valid_pred_15.npy', segmentation_results)

            all_metrics = {}
            # initialise empty placeholder lists
            pq_list = []
            mpq_info_list = []
            # load the prediction and ground truth arrays
            pred_array = np.load("/home/ll610/Onepiece/code_cs138/github/mmdetection/conic/dataset/conic/valid_pred_15.npy")
            true_array = np.load("/home/ll610/Onepiece/code_cs138/github/mmdetection/conic/dataset/conic/valid_true_15.npy")

            nr_patches = pred_array.shape[0]
            seg_metrics_names = ["pq", "multi_pq+"]

            for patch_idx in range(nr_patches):
                # get a single patch
                pred = pred_array[patch_idx]
                true = true_array[patch_idx]

                # instance segmentation map
                pred_inst = pred[..., 0]
                true_inst = true[..., 0]
                # classification map
                pred_class = pred[..., 1]
                true_class = true[..., 1]

                # ===============================================================

                for idx, metric in enumerate(seg_metrics_names):
                    if metric == "pq":
                        # get binary panoptic quality
                        pq = get_pq(true_inst, pred_inst)
                        pq = pq[0][2]
                        pq_list.append(pq)
                    elif metric == "multi_pq+":
                        # get the multiclass pq stats info from single image
                        mpq_info_single = get_multi_pq_info(true, pred)
                        mpq_info = []
                        # aggregate the stat info per class
                        for single_class_pq in mpq_info_single:
                            tp = single_class_pq[0]
                            fp = single_class_pq[1]
                            fn = single_class_pq[2]
                            sum_iou = single_class_pq[3]
                            mpq_info.append([tp, fp, fn, sum_iou])
                        mpq_info_list.append(mpq_info)
                    else:
                        raise ValueError("%s is not supported!" % metric)

            pq_metrics = np.array(pq_list)
            pq_metrics_avg = np.mean(pq_metrics, axis=-1)  # average over all images
            if "multi_pq+" in seg_metrics_names:
                mpq_info_metrics = np.array(mpq_info_list, dtype="float")
                # sum over all the images
                total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)

            for idx, metric in enumerate(seg_metrics_names):
                if metric == "multi_pq+":
                    mpq_list = []
                    # for each class, get the multiclass PQ
                    for cat_idx in range(total_mpq_info_metrics.shape[0]):
                        total_tp = total_mpq_info_metrics[cat_idx][0]
                        total_fp = total_mpq_info_metrics[cat_idx][1]
                        total_fn = total_mpq_info_metrics[cat_idx][2]
                        total_sum_iou = total_mpq_info_metrics[cat_idx][3]

                        # get the F1-score i.e DQ
                        dq = total_tp / (
                            (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6
                        )
                        # get the SQ, when not paired, it has 0 IoU so does not impact
                        sq = total_sum_iou / (total_tp + 1.0e-6)
                        mpq_list.append(dq * sq)
                    mpq_metrics = np.array(mpq_list)
                    all_metrics[metric] = [np.mean(mpq_metrics)]
                else:
                    all_metrics[metric] = [pq_metrics_avg]

            df = pd.DataFrame(all_metrics)
            df = df.to_string(index=False)
            print(df)
            with open("log_15.txt", "a") as text_file:
                text_file.write("\n"+str(df))

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
