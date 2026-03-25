from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from app.services.anomaly_service import run_pipeline, get_latest_anomalies, get_anomaly_explanation, get_metrics, GLOBAL_STATE
import os

router = APIRouter()

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
        df = pd.read_csv(file.file)
        df.columns = df.columns.astype(str).str.strip().str.lower()
        
        # Check if the file had data URI prefix on the first column header due to Safari issue
        if len(df.columns) > 0 and df.columns[0].startswith("data:"):
            df.columns.values[0] = df.columns[0].split(',')[-1]
        
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
    Returns system metrics.
    """
    return get_metrics()

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