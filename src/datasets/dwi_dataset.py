from typing import Literal
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
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
    Dataset class for working with k-space data.

    Args:
        data_path (Path | str): The path to the data.
        csv (pd.DataFrame): The DataFrame containing the data information.
        train (bool): Flag indicating whether the dataset is for training or not.
        grappa (bool, optional): Flag indicating whether GRAPPA reconstruction is used. Defaults to False.
        data_type (str, optional): The type of data to be used. Defaults to "magnitude".
        sampling_factor (int | None, optional): The sampling factor for undersampling. Defaults to None.
    """

    def __init__(
        self,
        data_path: Path | str,
        csv: pd.DataFrame,
        train: bool,
        grappa: bool = False,
        data_type: str = Literal["magnitude", "magnitude_phase", "magnitude_kspace"],
        sampling_factor: int | None = None,
    ) -> None:
        self.data_path = data_path
        self.csv = csv
        self.train = train
        self.grappa = grappa
        self.data_type = data_type
        self.sampling_factor = sampling_factor

    def __getitem__(self, idx: int):
        """
        Get the item at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: A dictionary containing the image and label.
        """
        kspace = np.load(Path(self.data_path, self.csv['fastmri_rawfile'][idx]))

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
    
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.csv['fastmri_rawfile'])

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
    data_path: Path | str,
    train_csv: pd.DataFrame,
    val_csv: pd.DataFrame,
    data_type: str = Literal["magnitude", "magnitude_phase", "magnitude_kspace"],
    batch_size: int = 32,
    sampling_factor: int = None,
):
    """
    Returns the train and validation data loaders for a given dataset.

    Args:
        data_path (Path | str): The path to the dataset.
        train_csv (pd.DataFrame): The DataFrame containing the training data.
        val_csv (pd.DataFrame): The DataFrame containing the validation data.
        data_type (str, optional): The type of data to load. Defaults to "magnitude".
        batch_size (int, optional): The batch size for the data loaders. Defaults to 32.
        sampling_factor (int, optional): The sampling factor for the data. Defaults to None.

    Returns:
        tuple: A tuple containing the train and validation data loaders.
    """

    dataloader_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
    }

    train_dataset = KSpace_Dataset(data_path, train_csv, train=True, data_type=data_type, sampling_factor=sampling_factor)
    val_dataset = KSpace_Dataset(data_path, val_csv, train=False, data_type=data_type, sampling_factor=sampling_factor)

    train_loader = DataLoader(train_dataset, **dataloader_params)
    val_loader = DataLoader(val_dataset, **dataloader_params)

    return train_loader, val_loader


def get_test_loader(
    data_path: Path | str,
    test_csv: pd.DataFrame,
    batch_size: int = 32,
    sampling_factor: int | None = None,
):
    """
    Returns a DataLoader object for the test dataset.

    Args:
        data_path (Path | str): The path to the data.
        test_csv (pd.DataFrame): The DataFrame containing the test data.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 32.
        sampling_factor (int | None, optional): The sampling factor for the dataset. Defaults to None.

    Returns:
        DataLoader: The DataLoader object for the test dataset.
    """

    dataloader_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 8,
        "pin_memory": True,
        "prefetch_factor": 4,
    }

    test_dataset = KSpace_Dataset(
        data_path, test_csv, train=False, sampling_factor=sampling_factor
    )

    test_loader = DataLoader(test_dataset, **dataloader_params)

    return test_loader
