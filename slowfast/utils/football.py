import torch
import slowfast.visualization.utils as vis_utils
from sklearn.metrics import confusion_matrix, average_precision_score, cohen_kappa_score

def get_xG_and_goals(preds, labels):
    """
    Calculate expected goals and goals.
    Args:
        preds (tensor or lists of tensors): predictions. Each tensor is in
            in the shape of (n_batch, num_classes). Tensor(s) must be on CPU.
        labels (tensor or lists of tensors): corresponding labels. Each tensor is
            in the shape of either (n_batch,) or (n_batch, num_classes).
    Returns:
        xG (float): the expected goals
        goals (int): the number of goals
    """
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)

    preds = torch.flatten(preds[:, 1])
    labels = torch.flatten(labels)

    xG = torch.sum(preds).item()
    goals = torch.sum(labels).item()

    return xG, goals

def get_figure_metrics(preds, labels):
    if isinstance(preds, list):
        preds = torch.cat(preds, dim=0)
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)

    preds_probs = torch.flatten(preds[:, 1]).cpu().detach().numpy().tolist()
    preds = torch.flatten(torch.argmax(preds, dim=1)).cpu().detach().numpy().tolist()
    labels = torch.flatten(labels).cpu().detach().numpy().tolist()

    cm = confusion_matrix(labels, preds)
    figure = vis_utils.plot_confusion_matrix(cm,
                                             num_classes=2,
                                             class_names=['no_goal', 'goal'])

    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))

    auc_pr = average_precision_score(labels, preds_probs)
    auc_pr_random = sum(labels) / len(labels)
    cohen_kappa = cohen_kappa_score(labels, preds)

    return figure, precision, recall, f1, auc_pr_random, auc_pr, cohen_kappa
