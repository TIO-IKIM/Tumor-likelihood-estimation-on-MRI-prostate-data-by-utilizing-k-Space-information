import torch
from sklearn import metrics as M
import torchmetrics
from torchmetrics.classification import MulticlassAveragePrecision, F1Score
from torchmetrics.wrappers import BootStrapper

def validation(predictions: list, targets: list, num_classes: int, bootstrap: bool=False):
    """
    Perform validation on the predictions using various metrics.

    Args:
        predictions (list): The predicted values.
        targets (list): The target values.
        num_classes (int): The number of classes.
        bootstrap (bool, optional): Whether to use bootstrap resampling for metrics computation. Defaults to False.

    Returns:
        dict: A dictionary containing the computed validation metrics.
    """

    val_metrics = {}

    predictions_softmax = torch.softmax(predictions, dim=1)

    F1 = F1Score(task="multiclass", num_classes=num_classes).to(targets.device)
    AUPRC = MulticlassAveragePrecision(num_classes=num_classes).to(targets.device)
    AUROC = torchmetrics.AUROC(task="multiclass", num_classes=num_classes).to(targets.device)

    if bootstrap:
        bootstrap_auprc = BootStrapper(AUPRC, num_bootstraps=1000).to(targets.device)
        bootstrap_auroc = BootStrapper(AUROC, num_bootstraps=1000).to(targets.device)
        bootstrap_auprc.update(predictions_softmax, targets)
        bootstrap_auroc.update(predictions_softmax, targets)
        val_metrics['auroc'] = bootstrap_auroc.compute()
        val_metrics['auprc'] = bootstrap_auprc.compute()
    else:
        val_metrics['auroc'] = AUROC(predictions_softmax, targets)
        val_metrics['auprc'] = AUPRC(predictions_softmax, targets)   

    val_metrics['f1'] = F1(predictions_softmax, targets)  

    score_pred = torch.argmax(predictions_softmax, dim=1).cpu()
    score_target = targets.cpu()
    val_metrics['accuracy'] = round(M.accuracy_score(score_target, score_pred), 3)
    val_metrics['recall'] = round(M.recall_score(score_target, score_pred, average="weighted"), 3)
    val_metrics['precision'] = M.precision_score(score_target, score_pred, average="weighted")

    return val_metrics