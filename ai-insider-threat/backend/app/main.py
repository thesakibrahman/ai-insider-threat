from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import anomaly, health
import os

app = FastAPI(title="AI Insider Threat Detection API")

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Since frontend might run on a different port (e.g., Live Server)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(anomaly.router, prefix="/api", tags=["Anomaly"])
app.include_router(health.router, prefix="/api", tags=["Health"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Powered Insider Threat Detection API"}