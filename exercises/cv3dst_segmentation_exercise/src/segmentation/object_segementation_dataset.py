from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset

from src.segmentation.utils import load_annotation, load_feature_map, load_img, read_split


class ObjectSegmentationDataset(Dataset):
    def __init__(self, path: Path, split: str = "train") -> None:
        self.resolution = "480p"
        self.no_gt = False
        if split in ["test", "unsupervised_0", "unsupervised_1"]:
            self.no_gt = True
        self.dataset_path = path
        self.image_paths: List[Tuple[str, str]] = []
        self.visualization_paths: List[Tuple[str, str]] = []

        split_path = path.joinpath(f"{split}_seqs.txt")
        self.split_classes = read_split(split_path)

        for split_class in self.split_classes:
            image_path = self.dataset_path.joinpath("JPEGImages", self.resolution, split_class)
            annotation_path = self.dataset_path.joinpath("Annotations", self.resolution, split_class)
            data_path = self.dataset_path.joinpath("FeatureMaps", self.resolution, split_class)

            for img_path in sorted(data_path.iterdir()):
                img_name = img_path.stem
                if annotation_path.joinpath(f"{img_name}.png").is_file() or self.no_gt:
                    self.image_paths.append((split_class, img_name))

                    if image_path.joinpath(f"{img_name}.jpg").is_file():
                        self.visualization_paths.append((split_class, img_name))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        img_class, img_name = self.image_paths[index]

        sample: Dict[str, torch.Tensor] = {}
        sample["data"] = load_feature_map(
            self.dataset_path.joinpath("FeatureMaps", self.resolution, img_class, f"{img_name}.pt")
        )
        if not self.no_gt:
            sample["annotations"] = load_annotation(
                self.dataset_path.joinpath("Annotations", self.resolution, img_class, f"{img_name}.png")
            )
        return sample

    def get_visualization_example(self, index) -> Dict[str, torch.Tensor]:
        sample: Dict[str, torch.Tensor] = {}
        img_class, img_name = self.visualization_paths[index]
        sample["data"] = load_feature_map(
            self.dataset_path.joinpath("FeatureMaps", self.resolution, img_class, f"{img_name}.pt")
        )
        if not self.no_gt:
            sample["color_annotations"] = load_annotation(
                self.dataset_path.joinpath("Annotations", self.resolution, img_class, f"{img_name}.png")
            )
        sample["images"] = load_img(
            self.dataset_path.joinpath("JPEGImages", self.resolution, img_class, f"{img_name}.jpg")
        )

        return sample
