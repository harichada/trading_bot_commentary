"""Run: python -m intraday_lab"""
from .app import app, PORT
import uvicorn

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=PORT)
