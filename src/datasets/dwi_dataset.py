from typing import Literal
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import random
from torchvision.transforms import v2
from pathlib import Path
from utils.fourier import ifft


def undersample(kspace: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Undersamples the k-space data by removing every n-th line, depending on the factor.

    Args:
        kspace (torch.Tensor): The input k-space data.
        factor (int): The undersampling factor.

    Returns:
        torch.Tensor: The undersampled k-space data.
    """

    kspace = kspace.clone()
    if factor > 1:
        mask = np.ones(kspace.shape, dtype=bool)
        midline = kspace.shape[1] // 2
        mask[0, midline::factor] = 0
        mask[0, midline::-factor] = 0
        kspace[torch.tensor(mask)] = 0

    return kspace


class KSpace_Dataset(Dataset):
    """
    A Dataset class for handling k-Space data.

    Attributes:
        csv: A CSV file containing the dataset.
        train: A boolean indicating if the dataset is for training.
        test: A boolean indicating if the dataset is for testing. Defaults to False.
        grappa: A boolean indicating if GRAPPA (GeneRalized Autocalibrating Partial Parallel Acquisition) is used. Defaults to False.
        data_type: A string indicating the type of data. It can be "magnitude", "magnitude_phase", or "magnitude_kspace".
        sampling_factor: An integer indicating the sampling factor. Defaults to None.
    """
    def __init__(
        self,
        csv,
        train: bool,
        test: bool = False,
        grappa: bool = False,
        data_type: str = Literal["magnitude", "magnitude_phase", "magnitude_kspace"],
        sampling_factor: int | None = None,
    ) -> None:
        super().__init__(csv, train, test)

        assert data_type in [
            "magnitude",
            "magnitude_phase",
            "magnitude_kspace",
        ], "Invalid data type"

        self.csv = csv
        self.train = train
        self.test = test
        self.grappa = grappa
        self.data_type = data_type
        self.sampling_factor = sampling_factor

    def __getitem__(self, idx: int):
            
        kspace = np.load(Path(self.csv['fastmri_rawfile'][idx]))

        self.y = self.csv["PIRADS"][idx] - 1
        self.y = int(self.y > 1)

        data_dict = {
            "image": torch.from_numpy(kspace).cfloat().unsqueeze(0),
            "label": self.y,
        }
        if self.train:
            if random.random() < 0.5:
                data_dict["image"] = v2.RandomHorizontalFlip(p=1)(data_dict["image"])

            data_dict["image"] = v2.Lambda(
                lambda x: undersample(x, random.randint(0, 8))
            )(data_dict["image"])
        else:
            if self.sampling_factor:
                data_dict["image"] = v2.Lambda(
                    lambda x: undersample(x, self.sampling_factor)
                )(data_dict["image"])

        data_dict["image"] = v2.functional.center_crop(data_dict["image"], (224, 224))

        if self.data_type == "magnitude":
            data_dict["image"] = v2.Lambda(lambda x: ifft(x).abs())(data_dict["image"])
        else:
            data_dict["image"] = v2.Lambda(lambda x: KSpace_Dataset.stack_complex(x))(
                data_dict["image"]
            )

        data_dict["image"] = v2.Lambda(lambda x: KSpace_Dataset.normalization(x))(
            data_dict["image"]
        )
        data_dict["image"] = v2.Lambda(lambda x: KSpace_Dataset.standardization(x))(
            data_dict["image"]
        )

        return data_dict

    def stack_complex(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stack the complex tensor `x` along the specified dimension.

        Args:
            x (torch.Tensor): The input complex tensor.

        Returns:
            torch.Tensor: The stacked tensor.
        """
        x_ifft = ifft(x)
        if self.data_type == "magnitude_phase":
            x = torch.cat([x_ifft.abs(), x_ifft.angle()], dim=0)
        else:
            x = torch.cat([x_ifft.abs(), x.real, x.imag], dim=0)

        return x

    @staticmethod
    def normalization(x: torch.Tensor) -> torch.Tensor:
        """
        Normalize the input tensor along each channel.

        Args:
            x (torch.Tensor): The input tensor to be normalized.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        for channel in range(x.size(0)):
            x[channel] = (x[channel] - x[channel].min()) / (
                x[channel].max() - x[channel].min()
            )

        return x

    @staticmethod
    def standardization(x: torch.Tensor) -> torch.Tensor:
        """
        Applies standardization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to be standardized.

        Returns:
            torch.Tensor: The standardized tensor.
        """
        for channel in range(x.size(0)):
            x[channel] = (x[channel] - x[channel].mean()) / x[channel].std()

        return x


def get_loaders(
    train_csv=None,
    val_csv=None,
    batch_size: int = 32,
    sampling_factor=None,
):
    """
    Get data loaders for training and validation datasets.

    Args:
        train_csv (str): Path to the CSV file containing training data.
        val_csv (str): Path to the CSV file containing validation data.
        batch_size (int): Number of samples per batch.
        sampling_factor (float): Sampling factor for the validation dataset.

    Returns:
        train_loader (torch.utils.data.DataLoader): Data loader for the training dataset.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation dataset.
    """
    dataloader_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
    }

    train_dataset = KSpace_Dataset(train_csv, train=True)
    val_dataset = KSpace_Dataset(val_csv, train=False, sampling_factor=sampling_factor)

    train_loader = DataLoader(train_dataset, **dataloader_params)
    val_loader = DataLoader(val_dataset, **dataloader_params)

    return train_loader, val_loader


def get_test_loader(
    test_csv: Path | str,
    batch_size: int = 32,
    sampling_factor: int | None = None,
):
    """
    Returns a DataLoader object for the test dataset.

    Args:
        test_csv (Path or str): Path to the CSV file containing test dataset information.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        sampling_factor (int or None, optional): Sampling factor for the test dataset. Defaults to None.

    Returns:
        DataLoader: DataLoader object for the test dataset.
    """
    dataloader_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
    }

    test_dataset = KSpace_Dataset(
        test_csv, train=False, test=True, sampling_factor=sampling_factor
    )

    test_loader = DataLoader(test_dataset, **dataloader_params)

    return test_loader
