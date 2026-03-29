from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from app.services.anomaly_service import run_pipeline, get_latest_anomalies, get_anomaly_explanation, get_metrics, GLOBAL_STATE
import os

router = APIRouter()

def auto_map_columns(df):
    """
    Heuristically maps uploaded CSV columns to exactly match the required schema:
    'timestamp', 'user', 'event_type', 'details'
    """
    col_mapping = {}
    
    aliases = {
        'timestamp': ['time', 'date', 'datetime', 'logtime', 'created_at', '_time', 'ts', 'timestamp'],
        'user': ['username', 'employee', 'employee_id', 'userid', 'account', 'actor', 'src_user', 'user'],
        'event_type': ['event', 'action', 'activity', 'type', 'category', 'operation', 'event_type', 'eventtype'],
        'details': ['description', 'message', 'msg', 'info', 'note', 'content', 'data', 'details', 'file_tree', 'filename', 'to', 'pc']
    }
    
    available_cols = list(df.columns)
    
    for target, target_aliases in aliases.items():
        if target in available_cols:
            continue
            
        match_found = False
        for col in available_cols:
            clean_col = str(col).lower().replace(' ', '').replace('_', '')
            for alias in target_aliases:
                clean_alias = alias.replace('_', '')
                if clean_col == clean_alias or clean_alias in clean_col:
                    col_mapping[col] = target
                    available_cols.remove(col)
                    match_found = True
                    break
            if match_found:
                break
                
    if col_mapping:
        df = df.rename(columns=col_mapping)
        print(f"Auto-mapped columns using aliases: {col_mapping}")
        
    # Fully dynamic fallback: If we still don't have the required columns,
    # generate synthetic equivalents so ANY file template analyzes successfully
    required_cols = {'timestamp', 'user', 'event_type', 'details'}
    missing = required_cols - set(df.columns)
    
    if missing:
        print(f"Missing strict template columns {missing}. Synthesizing dynamically for universal analysis...")
        import datetime
        
        if 'timestamp' in missing:
            # Provide sequential recent timestamps
            df['timestamp'] = [datetime.datetime.now() - datetime.timedelta(minutes=i) for i in range(len(df))]
            
        if 'user' in missing:
            df['user'] = "Custom_Entity"
            
        if 'event_type' in missing:
            df['event_type'] = "generic_activity"
            
        if 'details' in missing:
            # Compress all unused data columns into 'details' so the AI algorithms 
            # (especially intent scoring and vectorization) still analyze the source payload
            other_cols = [c for c in df.columns if c not in {'timestamp', 'user', 'event_type'}]
            if other_cols:
                df['details'] = df[other_cols].astype(str).agg(' | '.join, axis=1)
            else:
                df['details'] = "Custom Payload"

    return df


@router.post("/simulate")
async def trigger_simulation():
    """
    Triggers the data generation and ML pipeline.
    """
    try:
        run_pipeline()
        metrics = get_metrics()
        return {"status": "success", "message": "Pipeline completed successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload_csv")
async def upload_and_analyze(file: UploadFile = File(...)):
    """
    Accepts a custom CSV file and runs it through the ML pipeline.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    try:
        import pandas as pd
        # The user requested to allow huge files, so we expand the cap to 100,000 rows.
        # This will take ~2-3 minutes to run the IsolationForest and Autoencoder on standard machines.
        df = pd.read_csv(file.file, nrows=100000)
        df.columns = df.columns.astype(str).str.strip().str.lower()
        
        # Check if the file had data URI prefix on the first column header due to Safari issue
        if len(df.columns) > 0 and df.columns[0].startswith("data:"):
            df.columns.values[0] = df.columns[0].split(',')[-1]
            
        # Auto-map columns using heuristics
        df = auto_map_columns(df)
        
        # basic validation
        required_cols = {'timestamp', 'user', 'event_type', 'details'}
        if not required_cols.issubset(set(df.columns)):
            missing = required_cols - set(df.columns)
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")
            
        # run pipeline with custom df
        run_pipeline(custom_df=df)
        metrics = get_metrics()
        return {"status": "success", "message": "Custom data analyzed successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anomalies")
async def fetch_anomalies():
    """
    Returns the most recent anomalies detected.
    """
    anomalies = get_latest_anomalies(top_n=50)
    return {"anomalies": anomalies}

@router.get("/metrics")
async def fetch_metrics():
    """
    Returns system metrics evaluated on the held-out TEST split (15%).
    """
    return get_metrics()

@router.get("/split_info")
async def fetch_split_info():
    """
    Returns the live data split breakdown (train / val / test sizes and percentages).
    Matches the paper's Experimental Setup: 70% train, 15% validation, 15% test.
    """
    train_df    = GLOBAL_STATE.get('raw_df')
    val_df      = GLOBAL_STATE.get('val_df')
    test_df     = GLOBAL_STATE.get('test_df')
    data_source = GLOBAL_STATE.get('data_source', 'none')

    if train_df is None:
        return {
            "status": "pipeline_not_run",
            "message": "Run simulation or upload data first.",
            "split": None
        }

    train_size = len(train_df)
    val_size   = len(val_df)   if val_df   is not None and not val_df.empty   else 0
    test_size  = len(test_df)  if test_df  is not None and not test_df.empty  else 0
    total      = train_size + val_size + test_size

    source_labels = {
        'cert':      'CERT v6.2 Dataset',
        'custom':    'Custom Upload',
        'simulator': 'Simulated Data',
        'none':      'Unknown Source'
    }

    return {
        "status":      "ok",
        "data_source": data_source,
        "source_label": source_labels.get(data_source, 'Dataset'),
        "split": {
            "total":      total,
            "train":      {"size": train_size, "pct": 70,
                           "label": "Training",
                           "note": "Models learn ONLY from this data"},
            "validation": {"size": val_size,   "pct": 15,
                           "label": "Validation",
                           "note": "Used by Autoencoder during training to prevent overfitting. NOT used for metrics."},
            "test":       {"size": test_size,  "pct": 15,
                           "label": "Test — Metrics Evaluated Here",
                           "note": "Held-out. Precision / Recall / F1 / MTTD computed here ONLY."}
        }
    }

@router.get("/explain/{log_id}")
async def explain_anomaly(log_id: int):
    """
    Returns SHAP feature importance for a specific anomaly.
    """
    explanation = get_anomaly_explanation(log_id)
    if "error" in explanation:
        raise HTTPException(status_code=400, detail=explanation["error"])
    return {"explanation": explanation}

@router.get("/graph")
async def get_graph_html(filter: str = "all"):
    """
    Returns the interactive PyVis HTML file dynamically based on the filter.
    """
    df = GLOBAL_STATE.get('raw_df')
    if df is None:
        raise HTTPException(status_code=404, detail="Graph not generated yet. Run simulation first.")
        
    from app.services.graph_service import build_behavioral_graph, export_graph_to_pyvis
    
    # Build graph with the selected filter
    graph = build_behavioral_graph(df, filter_type=filter)
    
    static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    filename = f"graph_{filter}.html"
    graph_path = export_graph_to_pyvis(graph, output_dir=static_dir, filename=filename)
    
    return FileResponse(graph_path)