import pandas as pd
import numpy as np
from app.data.simulator import get_simulated_data
from app.services.preprocessing import preprocess_logs
from app.services.feature_engineering import engineer_features, get_feature_matrix
from app.models.isolation_forest import train_isolation_forest, predict_isolation_forest
from app.models.autoencoder import train_autoencoder, predict_autoencoder
from app.services.explain_service import generate_shap_explanation
from app.services.graph_service import build_behavioral_graph, export_graph_to_pyvis
import os

# Global state to hold models and data (In a real app, use a DB and proper ML model registry)
GLOBAL_STATE = {
    'raw_df': None,
    'features_df': None,
    'if_model': None,
    'ae_model': None,
    'graph_html_path': None
}

def run_pipeline(custom_df=None):
    """
    Runs the full ingestion, modeling, scoring, and graphing pipeline.
    """
    # 1. Ingestion
    if custom_df is not None:
        raw_df = custom_df.copy()
        raw_df['is_malicious_simulated'] = False # For real data, we don't know the ground truth
    else:
        raw_df = get_simulated_data()
    
    # 2. Preprocessing
    processed_df = preprocess_logs(raw_df.copy())
    
    # 3. Feature Engineering
    features_df = engineer_features(processed_df)
    X = get_feature_matrix(features_df)
    
    # 4. Model Training (Continual learning simulated by retraining on current window)
    if_model = train_isolation_forest(X)
    ae_model = train_autoencoder(X, epochs=30) # Train quickly for demo
    
    # 5. Scoring
    if_scores = predict_isolation_forest(if_model, X)
    ae_scores = predict_autoencoder(ae_model, X)
    
    # Ensemble Score: Simple average (equal weighting)
    # Both are normalized to 0-1
    ensemble_score = (if_scores + ae_scores) / 2.0
    
    raw_df['anomaly_score'] = ensemble_score
    raw_df['if_score'] = if_scores
    raw_df['ae_score'] = ae_scores
    
    # 6. Graph Generation
    graph = build_behavioral_graph(raw_df)
    # Ensure static dir exists
    static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    os.makedirs(static_dir, exist_ok=True)
    graph_path = export_graph_to_pyvis(graph, output_dir=static_dir)
    
    # Update Global State
    GLOBAL_STATE['raw_df'] = raw_df
    GLOBAL_STATE['features_df'] = features_df
    GLOBAL_STATE['if_model'] = if_model
    GLOBAL_STATE['ae_model'] = ae_model
    GLOBAL_STATE['graph_html_path'] = graph_path
    
    return raw_df

def get_latest_anomalies(top_n=50):
    df = GLOBAL_STATE.get('raw_df')
    if df is None:
        return []
    
    # Sort by anomaly score descending
    anomalies = df.sort_values(by='anomaly_score', ascending=False).head(top_n)
    
    results = []
    for idx, row in anomalies.iterrows():
        results.append({
            'log_id': idx, # passing dataframe index as log_id
            'timestamp': str(row['timestamp']),
            'user': row['user'],
            'role': row.get('role', 'Unknown'),
            'event_type': row['event_type'],
            'details': row.get('details', ''),
            'anomaly_score': float(row['anomaly_score']),
            'is_simulated_attack': bool(row.get('is_malicious_simulated', False))
        })
    return results

def get_anomaly_explanation(log_id: int):
    features_df = GLOBAL_STATE.get('features_df')
    if_model = GLOBAL_STATE.get('if_model')
    
    if features_df is None or if_model is None:
        return {"error": "Pipeline not run yet."}
        
    try:
        # Get the feature matrix for the specific log
        instance_features = features_df[features_df['log_id'] == log_id]
        if instance_features.empty:
            return {"error": "Log ID not found"}
            
        X_instance = get_feature_matrix(instance_features)
        X_train = get_feature_matrix(features_df)
        
        explanations = generate_shap_explanation(if_model, X_train, X_instance)
        return explanations
    except Exception as e:
        return {"error": str(e)}

def get_metrics():
    df = GLOBAL_STATE.get('raw_df')
    if df is None:
        return {"total_events": 0, "anomalies_detected": 0, "accuracy": 0}
        
    total = len(df)
    # Consider anomaly_score > 0.6 as flagged
    flagged = df[df['anomaly_score'] > 0.6]
    
    # Calculate simulated accuracy (metrics vs simulation flags)
    true_positives = len(flagged[flagged['is_malicious_simulated'] == True])
    false_positives = len(flagged[flagged['is_malicious_simulated'] == False])
    actual_malicious = len(df[df['is_malicious_simulated'] == True])
    
    accuracy = 0.0
    if actual_malicious > 0:
        accuracy = (true_positives / actual_malicious) * 100
        
    precision = 0.0
    if (true_positives + false_positives) > 0:
        precision = (true_positives / (true_positives + false_positives)) * 100
        
    return {
        "total_events": total,
        "anomalies_detected": len(flagged),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "simulated_recall": round(accuracy, 2),
        "simulated_precision": round(precision, 2)
    }