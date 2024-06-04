# -*- coding: utf-8 -*-
import os
import shutil
import yaml
from pathlib import Path
import torch, random
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
from datasets.dwi_dataset import get_loaders
from model import ConvNext
from utils.validation import validation
from utils.IKIMLogger import IKIMLogger

parser = argparse.ArgumentParser(
    prog="Training",
    description="Train a neural network to predict the likelihood of PCa on the FastMRI Prostate raw dataset.",
)

parser.add_argument("--e", type=int, default=100, help="Number of epochs for training")
parser.add_argument(
    "--log", type=str, default="INFO", help="Define debug level. Defaults to INFO."
)
parser.add_argument(
    "--gpu",
    type=int,
    default=0,
    help="GPU used for training.",
)
parser.add_argument(
    "--config",
    type=str,
    help="Path to configuration file",
    default="train_dwi.yaml",
)


def set_seed(seed: int = 42) -> None:
    """Set seeds for the libraries numpy, random, torch and torch.cuda.

    Args:
        seed (int, optional): Seed to be used. Defaults to `42`.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug(f"Random seed set as {seed}")


class EarlyStopping(object):
    """Early stops the training if performance doesn't improve after a given patience."""

    def __init__(
        self,
        patience=10,
        verbose=True,
        delta=0,
        monitor="val_loss",
        op_type="min",
        logger=None,
    ):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped. Default is 10.
            verbose (bool): If True, prints a message for each epoch where the validation loss decreases. Default is True.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
            monitor (str): Quantity to be monitored. Default is "val_loss".
            op_type (str): Type of optimization. "min" for minimizing the monitored quantity, "max" for maximizing it. Default is "min".
            logger (object): Logger object to log messages. Default is None.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.monitor = monitor
        self.op_type = op_type
        self.logger = logger

        if self.op_type == "min":
            self.val_score_min = np.Inf
        else:
            self.val_score_min = 0

    def __call__(self, val_score):
        score = -val_score if self.op_type == "min" else val_score

        if self.best_score is None:
            self.best_score = score
            self.print_and_update(val_score)
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.print_and_update(val_score)
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def print_and_update(self, val_score):
        """print_message when validation score decrease."""
        if self.verbose:
            logger.info(
                f"{self.monitor} optimized ({self.val_score_min:.6f} --> {val_score:.6f}).  Saving model ...",
            )
        self.val_score_min = val_score


class TrainNetwork:
    """Train a neural network based on PyTorch architecture.

    Args:
        args (dict): Dictionary containing user-specified settings.
        config (dict): Dictionary containing settings set in a yaml-config file.
    """

    def __init__(self, args: dict, config: dict) -> None:
        """
        Initializes the TrainNetwork class.

        Args:
            args (dict): A dictionary containing the command-line arguments.
            config (dict): A dictionary containing the configuration settings.

        Attributes:
            args (dict): A dictionary containing the command-line arguments.
            config (dict): A dictionary containing the configuration settings.
            train_path (str): The path to the training data.
            val_path (str): The path to the validation data.
            base_output (Path): The base output path.
            init_lr (float): The initial learning rate.
            epochs (int): The number of epochs.
            model_name (str): The name of the model.
            device (torch.device): The device to be used for training.

        Returns:
            None
        """
        self.args: dict = args
        self.config: dict = config
        self.data_path: str = config["data_path"]
        self.base_output: Path = config["base_output"]
        self.init_lr: float = config["lr"]
        self.data_type: str = config["data_type"]
        self.in_channel: int = config["in_channel"]
        self.num_classes: int = config["num_classes"]
        self.batch_size: int = config["batch_size"]
        self.epochs: int = args["e"]
        self.model_name: str = (
            f"{config['data_type']}_{config['lr']}_{config['comment']}"
        )
        self.device = torch.device(
            f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"
        )
        self.model = ConvNext(
            in_channels=self.in_channel, num_classes=self.num_classes
        ).to(self.device)

    def train_fn(self) -> None:
        """Train function.

        Calculates loss per batch, performs backpropagation and optimizer step.

        Args:
            self: self object of the class.

        Returns:
            None.
        """
        loop = tqdm(self.train_loader)
        self.total_train_loss = 0

        for batch_idx, data_dict in enumerate(loop):

            data = data_dict["image"].to(device=self.device, non_blocking=True)

            predictions = self.model(data)

            targets = data_dict["label"].to(device=data.device, non_blocking=True)

            loss = self.loss(predictions, targets)

            self.total_train_loss += loss.item()

            if torch.isnan(loss):
                logger.warning("-- Loss nan --")
                break

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            loop.set_postfix(loss=loss.item())

        self.scheduler.step()
        self.lr = self.scheduler.get_last_lr()[0]
        self.total_train_loss = self.total_train_loss / len(self.train_loader)

    def validation(self) -> None:
        """Performs validation after each epoch.

        This method saves one batch of the validation set in the save-folder
        and calculates the dice score as well as the validation loss.
        The results are logged to Weights & Biases.

        Args:
            self: Instance of `TrainNetwork` class.

        Returns:
            None
        """
        self.model.eval()
        total_validation_loss = 0
        loop = tqdm(self.val_loader)
        targets_sum = []
        preds_sum = []

        for batch_idx, data_dict in enumerate(loop):
            data = data_dict["image"].to(device=self.device, non_blocking=True)

            # forward
            with torch.no_grad():
                predictions = self.model(data)

            targets = data_dict["label"].to(device=data.device, non_blocking=True)

            targets_sum.append(targets)
            preds_sum.append(predictions)

            loss = self.loss(predictions, targets)

            loop.set_postfix(loss=loss.item())

            total_validation_loss += loss.item()

        val_metrics = validation(
            predictions=torch.cat(preds_sum),
            targets=torch.cat(targets_sum),
            num_classes=2,
        )
        total_validation_loss /= len(self.val_loader)
        logger.info(f"Val-loss: {total_validation_loss:.3f}")
        logger.info(
            f"Accuracy: {val_metrics['accuracy']:.3f} | Recall: {val_metrics['recall']:.3f} | Precision: {val_metrics['precision']:.3f} | AUROC: {val_metrics['auroc']:.3f} | AUPRC: {val_metrics['auprc']:.3f} | F1: {val_metrics['f1']:.3f}"
        )
        stop_metric = val_metrics["auroc"]

        self.early_stopping(stop_metric)

        if stop_metric > self.metric:
            self.metric = stop_metric
            torch.save(self.model, Path(self.save_folder) / self.model_name)

        self.model.train()

    def main(self) -> None:
        """
        Main method for training the model.

        Reads train and validation CSV files, initializes loaders, optimizer, loss function,
        and starts the training loop. Also performs early stopping based on the validation
        performance.

        Returns:
            None
        """
        train_csv = pd.read_csv("datasets/dwi_2D_train.csv")
        val_csv = pd.read_csv("datasets/dwi_2D_val.csv")

        self.metric_list = []
        self.save_folder = f"{self.base_output}/train_{self.model_name}"
        Path(self.save_folder).mkdir(exist_ok=True)

        logger.info(f"Device: {self.device}")

        self.train_loader, self.val_loader = get_loaders(
            data_path=self.data_path,
            train_csv=train_csv,
            val_csv=val_csv,
            data_type=self.data_type,
            batch_size=self.batch_size,
            sampling_factor=None,
        )

        self.early_stopping = EarlyStopping(
            patience=50, verbose=True, monitor="auroc", op_type="max"
        )
        self.metric = 0.0
        self.lr = self.init_lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=3, eta_min=0
        )
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 17.0])).to(
            self.device
        )

        # Copy config file to save folder
        shutil.copyfile(args.config, Path(self.save_folder, Path(args.config).name))
        # Log save folder information
        logger.info(f"Save folder: {str(self.save_folder)}")

        # Start epoch loop
        for self.epoch in range(self.epochs):
            logger.info(f"Now training epoch {self.epoch}!")
            TrainNetwork.train_fn(self)

            logger.info(f"Train-loss: {self.total_train_loss:.3f}")

            # Validate the model
            TrainNetwork.validation(self)
            if self.early_stopping.early_stop == True:
                logger.info("Early stopping ...")
                break


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parser.parse_args()
    args.config = "src/configs/" + args.config

    with open(args.config, "r") as conf:
        config = yaml.safe_load(conf)

    torch.set_num_threads(5)

    ikim_logger = IKIMLogger(
        level=args.log,
        log_dir="logs/",
        comment=(f"train_{config['lr']}_{config['comment']}"),
    )
    logger = ikim_logger.create_logger()

    try:
        set_seed(42)
        training = TrainNetwork(
            args=args.__dict__,
            config=config,
        )
        logger.info(training.__repr__())
        training.main()
    except Exception as e:
        logger.exception(e)
