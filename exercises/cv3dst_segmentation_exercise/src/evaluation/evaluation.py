from pathlib import Path
from typing import Dict, List

import torch
from src.evaluation.metrics import confusion_matrix, accuracy, mean_precision, mean_recall, mean_iou


def load_submission_tensor(paths: List[Path], gt: bool) -> Dict[str, torch.Tensor]:
    assert len(paths) == 4, "Incorrect number of files"

    def load_tensor(name: str) -> torch.Tensor:
        tensor_path = None
        for path in paths:
            if path.stem.startswith(name) and path.is_file():
                tensor_path = path
        if tensor_path is None:
            if gt:
                raise ValueError("Oops. Something went wrong on our side.")
            else:
                raise ValueError(f"Could not find {name} output file.")
        return torch.load(tensor_path)

    output: Dict[str, torch.Tensor] = {}
    output["obj_seg"] = load_tensor("object_segmentation")
    output["sem_seg"] = load_tensor("semantic_segmentation")
    output["clustering_0"] = load_tensor("clustering_0")
    output["clustering_1"] = load_tensor("clustering_1")

    return output


def evaluate_obj_seg(submission: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    conf_matrix = confusion_matrix(submission, gt, 2)

    metrics: Dict[str, float] = {}
    metrics["Acc"] = accuracy(conf_matrix)
    metrics["mPrcn"] = mean_precision(conf_matrix)
    metrics["mRcll"] = mean_recall(conf_matrix)
    metrics["mIOU"] = mean_iou(conf_matrix)

    return metrics


def evaluate_sem_seg(submission: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    mask = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    conf_matrix = confusion_matrix(submission, gt, 9)

    metrics: Dict[str, float] = {}
    metrics["Acc"] = accuracy(conf_matrix)
    metrics["mPrcn"] = mean_precision(conf_matrix, mask)
    metrics["mRcll"] = mean_recall(conf_matrix, mask)
    metrics["mIOU"] = mean_iou(conf_matrix, mask)

    return metrics


def evaluate_clustering(submission: torch.Tensor, gt: torch.Tensor) -> Dict[str, float]:
    def find_best_class() -> int:
        best_accuracy = 0
        best_class = -1
        for class_num in range(int(submission.max().item()) + 1):
            bin_labels = (submission == class_num).to(torch.int16)
            acc = accuracy(confusion_matrix(bin_labels, gt, 2))

            if acc > best_accuracy:
                best_accuracy = acc
                best_class = class_num

        return best_class

    best_class = find_best_class()
    bin_labels = (submission == best_class).to(torch.int16)

    conf_matrix = confusion_matrix(bin_labels, gt, 2)

    metrics: Dict[str, float] = {}
    metrics["Acc"] = accuracy(conf_matrix)
    metrics["mPrcn"] = mean_precision(conf_matrix)
    metrics["mRcll"] = mean_recall(conf_matrix)
    metrics["mIOU"] = mean_iou(conf_matrix)

    return metrics
