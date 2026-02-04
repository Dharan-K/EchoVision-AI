"""
Minimal ASGI wrapper to avoid lifespan issues on Windows
"""
import asyncio
import os
import sys

# Set event loop policy before importing FastAPI/Uvicorn
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ.setdefault("OPENROUTER_API_KEY", "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")

from webapp import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        lifespan="off"  # Disable lifespan to avoid Windows asyncio issues
    )
