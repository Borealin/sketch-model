import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, r2_score


def accuracy(scores, targets):
    S = targets.cpu().numpy()
    C = scores.cpu().numpy()
    S, C = S.flatten(), C.flatten()
    CM = confusion_matrix(S, C).astype(np.float32)
    nb_classes = CM.shape[0]
    targets = targets.cpu().detach().numpy()
    nb_non_empty_classes = 0
    pr_classes = np.zeros(nb_classes)
    for r in range(nb_classes):
        cluster = np.where(targets == r)[0]
        if cluster.shape[0] != 0:
            pr_classes[r] = CM[r, r] / float(cluster.shape[0])
            if CM[r, r] > 0:
                nb_non_empty_classes += 1
        else:
            pr_classes[r] = 0.0
    acc = 100. * np.sum(pr_classes) / float(nb_classes)
    return acc


def f1score(scores, targets, average='micro'):
    """Computes the F1 score using scikit-learn for binary class labels. 

    Returns the F1 score for the positive class, i.e. labelled '1'.
    """
    y_true = targets.cpu().numpy()
    y_pred = scores.cpu().numpy()
    return f1_score(y_true, y_pred, average=average)


def r2score(scores, targets):
    y_true = targets.cpu().numpy()
    y_pred = scores.cpu().numpy()
    return r2_score(y_true, y_pred)
