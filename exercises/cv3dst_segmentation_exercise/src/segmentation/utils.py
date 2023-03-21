import csv
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from sklearn.metrics import confusion_matrix as sk_conf_matrix
from skimage import io

# =================================================== TEST VARIABLES ===================================================
int_1 = torch.tensor([[0, 2, 6, 10], [6, 2, 4, 8], [-4, 0, 4, 4], [0, 0, 0, 0]], dtype=torch.float32)
int_2 = torch.tensor([[4, 2, 8], [4, 20, 6], [4, -6, 0]], dtype=torch.float32)

int_1_nearest = torch.tensor(
    [
        [0, 0, 2, 2, 6, 6, 10, 10],
        [0, 0, 2, 2, 6, 6, 10, 10],
        [6, 6, 2, 2, 4, 4, 8, 8],
        [6, 6, 2, 2, 4, 4, 8, 8],
        [-4, -4, 0, 0, 4, 4, 4, 4],
        [-4, -4, 0, 0, 4, 4, 4, 4],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=torch.float32,
)
int_2_nearest = torch.tensor(
    [
        [4, 4, 2, 2, 8, 8],
        [4, 4, 2, 2, 8, 8],
        [4, 4, 20, 20, 6, 6],
        [4, 4, 20, 20, 6, 6],
        [4, 4, -6, -6, 0, 0],
        [4, 4, -6, -6, 0, 0],
    ],
    dtype=torch.float32,
)

int_1_bilinear = torch.tensor(
    [
        [0.0000, 0.5000, 1.5000, 3.0000, 5.0000, 7.0000, 9.0000, 10.0000],
        [1.5000, 1.6250, 1.8750, 2.8750, 4.6250, 6.5000, 8.5000, 9.5000],
        [4.5000, 3.8750, 2.6250, 2.6250, 3.8750, 5.5000, 7.5000, 8.5000],
        [3.5000, 3.0000, 2.0000, 2.1250, 3.3750, 4.7500, 6.2500, 7.0000],
        [-1.5000, -1.0000, 0.0000, 1.3750, 3.1250, 4.2500, 4.7500, 5.0000],
        [-3.0000, -2.2500, -0.7500, 0.7500, 2.2500, 3.0000, 3.0000, 3.0000],
        [-1.0000, -0.7500, -0.2500, 0.2500, 0.7500, 1.0000, 1.0000, 1.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    ],
    dtype=torch.float32,
)
int_2_bilinear = torch.tensor(
    [
        [4.0000, 3.5000, 2.5000, 3.5000, 6.5000, 8.0000],
        [4.0000, 4.6250, 5.8750, 6.7500, 7.2500, 7.5000],
        [4.0000, 6.8750, 12.6250, 13.2500, 8.7500, 6.5000],
        [4.0000, 6.3750, 11.1250, 11.2500, 6.7500, 4.5000],
        [4.0000, 3.1250, 1.3750, 0.7500, 1.2500, 1.5000],
        [4.0000, 1.5000, -3.5000, -4.5000, -1.5000, 0.0000],
    ],
    dtype=torch.float32,
)
int_1_bilinear_alt = torch.tensor(
    [
        [0.0000, 0.8571, 1.7143, 3.1429, 4.8571, 6.5714, 8.2857, 10.0000],
        [2.5714, 2.3265, 2.0816, 2.8980, 4.2449, 5.7143, 7.4286, 9.1429],
        [5.1429, 3.7959, 2.4490, 2.6531, 3.6327, 4.8571, 6.5714, 8.2857],
        [3.1429, 2.4082, 1.6735, 2.1633, 3.2653, 4.4082, 5.6327, 6.8571],
        [-1.1429, -0.4082, 0.3265, 1.5510, 3.0204, 4.1633, 4.6531, 5.1429],
        [-3.4286, -1.9592, -0.4898, 0.9796, 2.4490, 3.4286, 3.4286, 3.4286],
        [-1.7143, -0.9796, -0.2449, 0.4898, 1.2245, 1.7143, 1.7143, 1.7143],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
    ],
    dtype=torch.float32,
)
int_2_bilinear_alt = torch.tensor(
    [
        [4.0000e00, 3.2000e00, 2.4000e00, 3.2000e00, 5.6000e00, 8.0000e00],
        [4.0000e00, 6.0800e00, 8.1600e00, 8.8000e00, 8.0000e00, 7.2000e00],
        [4.0000e00, 8.9600e00, 1.3920e01, 1.4400e01, 1.0400e01, 6.4000e00],
        [4.0000e00, 8.3200e00, 1.2640e01, 1.2800e01, 8.8000e00, 4.8000e00],
        [4.0000e00, 4.1600e00, 4.3200e00, 4.0000e00, 3.2000e00, 2.4000e00],
        [4.0000e00, 5.9605e-08, -4.0000e00, -4.8000e00, -2.4000e00, 0.0000e00],
    ],
    dtype=torch.float32,
)

sub_conv_1 = torch.tensor(
    [
        [[-3.0, 2.0], [3.0, 4.0], [-3.0, 4.0], [0.0, 0.0]],
        [[0.0, 2.0], [4.0, 4.0], [3.0, 0.0], [1.0, -1.0]],
        [[-2.0, -3.0], [2.0, 1.0], [0.0, -1.0], [3.0, 2.0]],
        [[3.0, -1.0], [4.0, 1.0], [-2.0, -1.0], [4.0, 0.0]],
        [[-2.0, -1.0], [0.0, 2.0], [2.0, 0.0], [-1.0, 1.0]],
        [[0.0, -3.0], [-1.0, 2.0], [0.0, 2.0], [-1.0, 2.0]],
        [[4.0, 4.0], [2.0, 2.0], [-1.0, -1.0], [-1.0, 4.0]],
        [[-3.0, 4.0], [-3.0, -2.0], [4.0, 1.0], [-3.0, -1.0]],
    ],
    dtype=torch.float32,
)

sub_conv_2 = torch.tensor(
    [
        [
            [2.0, -1.0, -3.0, 2.0],
            [-1.0, 4.0, -2.0, 1.0],
            [-2.0, 0.0, -3.0, 3.0],
            [0.0, -1.0, 4.0, 3.0],
            [1.0, 1.0, 4.0, 2.0],
            [3.0, 1.0, 0.0, 3.0],
        ],
        [
            [4.0, -1.0, -1.0, -1.0],
            [0.0, -3.0, 0.0, -1.0],
            [2.0, -3.0, -1.0, 0.0],
            [-3.0, 3.0, -2.0, 0.0],
            [0.0, 2.0, 4.0, 3.0],
            [0.0, 0.0, -1.0, 2.0],
        ],
        [
            [0.0, 1.0, -2.0, -3.0],
            [-1.0, -2.0, -1.0, 4.0],
            [0.0, -1.0, 4.0, -1.0],
            [4.0, -1.0, -2.0, 2.0],
            [1.0, -1.0, 2.0, -3.0],
            [4.0, 3.0, 2.0, 4.0],
        ],
        [
            [-1.0, 0.0, 2.0, 4.0],
            [-2.0, -3.0, -3.0, -1.0],
            [3.0, -2.0, -3.0, 2.0],
            [3.0, 4.0, 4.0, 2.0],
            [-1.0, -1.0, 1.0, 0.0],
            [2.0, -1.0, 0.0, 3.0],
        ],
    ],
    dtype=torch.float32,
)

sub_conv_1_out = torch.tensor(
    [
        [
            [-3.0, 0.0, 2.0, 2.0],
            [-2.0, 3.0, -3.0, -1.0],
            [3.0, 4.0, 4.0, 4.0],
            [2.0, 4.0, 1.0, 1.0],
            [-3.0, 3.0, 4.0, 0.0],
            [0.0, -2.0, -1.0, -1.0],
            [0.0, 1.0, 0.0, -1.0],
            [3.0, 4.0, 2.0, 0.0],
        ],
        [
            [-2.0, 0.0, -1.0, -3.0],
            [4.0, -3.0, 4.0, 4.0],
            [0.0, -1.0, 2.0, 2.0],
            [2.0, -3.0, 2.0, -2.0],
            [2.0, 0.0, 0.0, 2.0],
            [-1.0, 4.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, 2.0],
            [-1.0, -3.0, 4.0, -1.0],
        ],
    ],
    dtype=torch.float32,
)

sub_conv_2_out = torch.tensor(
    [
        [
            [2.0, 4.0, -1.0, -1.0, -3.0, -1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0, 0.0, -2.0, 2.0, -3.0, 4.0],
            [-1.0, 0.0, 4.0, -3.0, -2.0, 0.0, 1.0, -1.0],
            [-1.0, -2.0, -2.0, -3.0, -1.0, -3.0, 4.0, -1.0],
            [-2.0, 2.0, 0.0, -3.0, -3.0, -1.0, 3.0, 0.0],
            [0.0, 3.0, -1.0, -2.0, 4.0, -3.0, -1.0, 2.0],
            [0.0, -3.0, -1.0, 3.0, 4.0, -2.0, 3.0, 0.0],
            [4.0, 3.0, -1.0, 4.0, -2.0, 4.0, 2.0, 2.0],
            [1.0, 0.0, 1.0, 2.0, 4.0, 4.0, 2.0, 3.0],
            [1.0, -1.0, -1.0, -1.0, 2.0, 1.0, -3.0, 0.0],
            [3.0, 0.0, 1.0, 0.0, 0.0, -1.0, 3.0, 2.0],
            [4.0, 2.0, 3.0, -1.0, 2.0, 0.0, 4.0, 3.0],
        ]
    ],
    dtype=torch.float32,
)


def test_interpolation(interpolation: Callable[[torch.Tensor, int], torch.Tensor], mode: str):
    if not mode in ["nearest", "bilinear"]:
        raise ValueError("Please chose a valid interpolation strategy")
    out_1 = interpolation(int_1[None, None], 2)
    out_2 = interpolation(int_2[None, None], 2)

    if out_1.shape != (1, 1, 8, 8):
        raise ValueError(f"Expected shape\n{(1, 1, 8, 8)}\nGot\n{out_1.shape}")
    if out_2.shape != (1, 1, 6, 6):
        raise ValueError(f"Expected shape\n{(1, 1, 6, 6)}\nGot\n{out_2.shape}")
    print("Shapes are ok.")

    if mode == "nearest":
        if not out_1[0, 0].allclose(int_1_nearest):
            raise ValueError(f"Expected\n{int_1_nearest}\nGot\n{out_1[0, 0]}")
        if not out_2[0, 0].allclose(int_2_nearest):
            raise ValueError(f"Expected\n{int_2_nearest}\nGot\n{out_2[0, 0]}")

    if mode == "bilinear":
        if not (out_1[0, 0].allclose(int_1_bilinear) or out_1[0, 0].allclose(int_1_bilinear_alt, 1.0e-3)):
            raise ValueError(f"Expected\n{int_1_bilinear}\nor\n{int_1_bilinear_alt}\nGot\n{out_1[0, 0]}")
        if not (out_2[0, 0].allclose(int_2_bilinear) or out_2[0, 0].allclose(int_2_bilinear_alt, 1.0e-3)):
            raise ValueError(f"Expected\n{int_2_bilinear}\nor\n{int_2_bilinear_alt}\nGot\n{out_2[0, 0]}")
    print("Values are ok.")


def test_subconv(subpixel_convolution: Callable[[torch.Tensor, int], torch.Tensor]):
    out_1 = subpixel_convolution(sub_conv_1[None], 2)
    out_2 = subpixel_convolution(sub_conv_2[None], 2)

    if out_1.shape != (1, 2, 8, 4):
        raise ValueError(f"Expected shape\n{(1, 2, 8, 4)}\nGot\n{out_1.shape}")
    if out_2.shape != (1, 1, 12, 8):
        raise ValueError(f"Expected shape\n{(1, 1, 12, 8)}\nGot\n{out_2.shape}")
    print("Shapes are ok.")

    if not out_1[0].allclose(sub_conv_1_out):
        raise ValueError(f"Expected\n{sub_conv_1_out}\nGot\n{out_1[0]}")
    if not out_2[0].allclose(sub_conv_2_out):
        raise ValueError(f"Expected\n{sub_conv_2_out}\nGot\n{out_2[0]}")
    print("Values are ok.")


# ===================================================== DATALOADING ====================================================
def read_split(path: Path) -> List[str]:
    with open(path) as f:
        lines = f.read().splitlines()
    return lines


def load_img(path: Path) -> torch.tensor:
    return torch.from_numpy(io.imread(path)).permute(2, 0, 1) / 255.0


def load_annotation(path: Path) -> torch.Tensor:
    img = io.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    return torch.from_numpy(img) / 255.0


def load_semantic_annotation(path: Path) -> torch.Tensor:
    img = io.imread(path)
    return torch.from_numpy(img).permute(2, 0, 1)


def load_feature_map(path: Path) -> torch.Tensor:
    f_map = torch.load(path).to(torch.float32)
    return f_map


def read_class_dict(path: Path) -> Dict[Tuple[int, int, int], Tuple[int, str]]:
    names = []
    colors = []
    class_dict = {}
    with open(path.joinpath("reduced_class_dict.csv"), "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        # skip header
        next(csv_reader, None)
        for line in csv_reader:
            names.append(line[0])
            colors.append((int(line[1]), int(line[2]), int(line[3])))

    for i, (name, color) in enumerate(zip(names, colors)):
        class_dict[color] = (i, name)

    return class_dict


# ==================================================== MISCELLANEOUS ===================================================
def colors_to_labels(
    color_annotation: torch.Tensor, class_dict: Dict[Tuple[int, int, int], Tuple[int, str]]
) -> torch.Tensor:
    label_annotation = -torch.ones_like(color_annotation)[0]
    for color, (label, name) in class_dict.items():
        mask = torch.all(color_annotation == torch.tensor([color[0], color[1], color[2]])[:, None, None], dim=0)
        label_annotation[mask] = label
    return label_annotation


def labels_to_color(
    label_annotation: torch.Tensor, class_dict: Dict[Tuple[int, int, int], Tuple[int, str]]
) -> torch.Tensor:
    color_annotation = -torch.ones(label_annotation.shape + (3,))
    for color, (label, name) in class_dict.items():
        mask = label_annotation == label
        color_annotation[mask] = torch.tensor([color[0], color[1], color[2]], dtype=torch.float32)
    return color_annotation.permute(2, 0, 1)


def metrics_header():
    print(
        f"{f'Epoch' : >8} {f'Split' : >10} {f'Loss' : >6} {f'Acc' : >5} {f'mPrcn' : >5} {f'mRcll' : >5} {f'mIOU' : >5}"
    )


def print_metrics(metrics: Dict[str, float], epoch: int, split_name: str):
    metric_names = ["loss", "acc", "m_prcn", "m_rcll", "m_iou"]
    str_metrics: Dict[str, str] = {}
    for metric in metric_names:
        value = metrics.get(metric, None)
        if value:
            str_metrics[metric] = f"{value :.2f}"
        else:
            str_metrics[metric] = "-"

    loss = str_metrics["loss"]
    acc = str_metrics["acc"]
    m_prcn = str_metrics["m_prcn"]
    m_rcll = str_metrics["m_rcll"]
    m_iou = str_metrics["m_iou"]

    print(
        f"{f'{epoch}' : >8} {f'{split_name}' : >10} {f'{loss}' : >6} {f'{acc}' : >5} {f'{m_prcn}' : >5} {f'{m_rcll}' : >5} {f'{m_iou}' : >5}"
    )


# ===================================================== PREDICTION =====================================================
def logits_to_labels(logits: torch.Tensor) -> torch.Tensor:
    labels = logits.max(1).indices
    return labels


def binary_output_to_labels(logits: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros_like(logits)
    labels[logits >= 0.5] = 1.0
    return labels


# ======================================================= METRICS ======================================================
def pixelwise_accuracy(prediction: torch.Tensor, annotations: torch.Tensor) -> Tuple[int, int]:
    # prediction: B,H,W
    # annotations: B,H,W
    pred_correct = int(((prediction - annotations) == 0).sum().item())
    total = int(prediction.view(-1).shape[0])

    return pred_correct, total


def confusion_matrix(prediction: torch.Tensor, annotations: torch.Tensor, num_classes: int) -> torch.Tensor:
    conf_matrix = torch.zeros(num_classes, num_classes)

    cm = torch.from_numpy(sk_conf_matrix(prediction.view(-1).cpu(), annotations.view(-1).cpu()))
    conf_matrix[: cm.shape[0], : cm.shape[1]] = cm

    return conf_matrix


def precision(true_positive: int, false_positive: int) -> float:
    return (true_positive) / (true_positive + false_positive)


def recall(true_positive: int, false_negative: int) -> float:
    return (true_positive) / (true_positive + false_negative)
