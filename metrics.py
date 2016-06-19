import numpy as np
from sklearn.metrics import make_scorer


def log_loss_score_function(truth, prediction, **kwargs):
    t = truth
    if len(truth.shape) == 1:
        t = convert_to_one_hot(truth)

    zero = np.math.pow(10, -15)
    total_predictions_per_image = np.sum(prediction, axis=1)
    normalized_predictions = prediction / total_predictions_per_image[:, None]
    updated_predictions = np.maximum(np.minimum(normalized_predictions, 1 - zero), zero)
    log_losses = t * np.log(updated_predictions)
    return -1 * np.average(np.sum(log_losses, axis=1))


def convert_to_one_hot(truth):
    t = (truth + 1) / 2
    t = t.astype(int)
    a = np.zeros((truth.shape[0], 2))
    a[:, 0][t == 0] = 1
    a[:, 1][t == 1] = 1
    return a


multi_class_log_loss = make_scorer(log_loss_score_function, greater_is_better=False, needs_proba=True)
