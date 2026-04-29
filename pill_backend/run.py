#!/usr/bin/env python3
# run.py — Start the Pill Recognition API server

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,       # single worker (model is in memory)
    )
