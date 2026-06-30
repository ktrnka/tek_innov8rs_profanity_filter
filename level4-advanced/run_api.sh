#!/bin/bash
# Start the FastAPI server

echo "Starting Profanity Filter API..."
echo "================================"
echo ""
echo "Interactive docs: http://localhost:8000/docs"
echo "ReDoc:           http://localhost:8000/redoc"
echo "Health check:    http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo ""

source .venv/bin/activate
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
