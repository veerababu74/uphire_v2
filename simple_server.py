#!/usr/bin/env python3
"""
Simple FastAPI server for testing manual search only
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apis.manual_search import router as manual_search_router

# Create a simple FastAPI app
app = FastAPI(
    title="Manual Search Test API",
    description="Testing manual search functionality",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include only the manual search router
app.include_router(manual_search_router)


@app.get("/")
async def root():
    return {
        "message": "Manual Search Test API",
        "docs_url": "/docs",
        "manual_search_url": "/manualsearch/",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("simple_server:app", host="127.0.0.1", port=8000, reload=False)
