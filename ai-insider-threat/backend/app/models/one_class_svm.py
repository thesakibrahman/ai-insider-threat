import pandas as pd
from sklearn.svm import OneClassSVM

def train_one_class_svm(X: pd.DataFrame) -> OneClassSVM:
    """
    Trains a One-Class SVM anomaly detection model.
    """
    # Using nu roughly equal to the expected anomaly rate (0.05)
    model = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    model.fit(X)
    return model

def predict_one_class_svm(model: OneClassSVM, X: pd.DataFrame):
    """
    Returns anomaly scores from the OCSVM model.
    In sklearn, negative scores are anomalies. We invert and normalize to 0-1 range.
    """
    scores = model.decision_function(X)
    
    # smaller scores are more anomalous. We'll invert them for risk representation
    inverted_scores = -scores 
    min_score, max_score = inverted_scores.min(), inverted_scores.max()
    
    if max_score > min_score:
        normalized_scores = (inverted_scores - min_score) / (max_score - min_score)
    else:
        normalized_scores = inverted_scores * 0.0
        
    return normalized_scores
