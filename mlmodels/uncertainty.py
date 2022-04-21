import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score


def probability(X_test, model, n):
    reg_preds = []
    for _ in range(n):
        reg_pred = model(X_test)
        reg_preds.append(reg_pred)

    return np.hstack(reg_preds)


def dnn_uncertainty(X_test, Y_test, model, n):
    acc = []
    for _ in range(n):
        pred = model.predict(X_test)[0]
        pred = np.argmax(pred, axis=1)
        acc.append(accuracy_score(Y_test, pred))
    
    return np.mean(acc), np.std(acc)


def rmse(X_test, Y_test, model, n):
    mae = []
    for _ in range(n):
        reg_pred = model.predict(X_test)[1]
        mae.append(mean_squared_error(Y_test, reg_pred, squared=False))
        
    return np.mean(mae), np.std(mae)
