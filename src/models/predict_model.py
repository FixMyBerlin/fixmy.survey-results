import numpy as np

def ordinal_predict(y_pred, threshold=0.5):
    output = np.zeros(np.shape(y_pred)[0])
    max_value = np.shape(y_pred)[1]
    for i, y in enumerate(output):
        found = False
        for j, value in enumerate(y_pred[i]):
            if value < threshold:
                output[i] = j
                found = True
                break
        if not found:
            output[i] = max_value
    return output