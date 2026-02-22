import numpy as np

def check_drift(new_data, reference_mean, threshold=0.5):
    new_mean = np.mean(new_data)

    if abs(new_mean - reference_mean) > threshold:
        return "Drift detected"
    return "No drift"
