import numpy as np

def probability(X_test, model, n):
    clf_preds, reg_preds = [], []
    for _ in range(n):
        clf_pred, reg_pred = model(X_test)
        clf_preds.append(clf_pred)
        reg_preds.append(reg_pred)

    return np.stack(clf_preds).mean(axis=0), np.hstack(reg_preds)
