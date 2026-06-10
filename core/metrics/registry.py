from .types.regression import MaxAccuracy, FScore



REGRESSION_METRICS = {
    "max_accuracy": MaxAccuracy,
    "f_score": FScore,
}


CLASSIFICATION_METRICS = {
    None,
}