import torch
import pandas as pd
from datasets.dwi_dataset import get_test_loader
from utils.validation import validation


def set_seed(seed: int = 42) -> None:
    """Set seeds for the libraries numpy, random, torch and torch.cuda.

    Args:
        seed (int, optional): Seed to be used. Defaults to `42`.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


test_csv = pd.read_csv("datasets/dwi_2D_test.csv")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("/path/to/model").to(device)
model.eval()

set_seed(42)

sampling_factors = [0, 2, 4, 8, 12, 16, 24, 32, 48]
bootstrap = True

for i in sampling_factors:
    loader = get_test_loader(test_csv, batch_size=1, sampling_factor=i)

    pred_sum = []
    label_sum = []

    for batch_idx, data_dict in enumerate(loader):
        input = data_dict["image"].to(device)
        label = data_dict["label"]

        with torch.no_grad():
            pred = model(input)

        pred = pred.to("cpu").detach()
        pred_sum.append(pred)
        label_sum.append(label)

    pred_sum = torch.cat(pred_sum)
    label_sum = torch.cat(label_sum)

    metrics = validation(pred_sum, label_sum, 2, bootstrap=bootstrap)

    print(f"x{i}: AUROC: {metrics['auroc']} | AUPRC: {metrics['auprc']}")
