#!/usr/bin/env python3
"""
Simplified MetaQore Orchestrator Server
Lightweight coordination service for TerraQore Studio agents.
"""

import uvicorn
from metaqore.config import MetaQoreConfig
from metaqore.api.app import app

def main():
    """Start the MetaQore orchestrator server."""
    config = MetaQoreConfig()

    print("🚀 Starting MetaQore Orchestrator...")
    print(f"📍 Host: {config.host}")
    print(f"📍 Port: {config.port}")
    print(f"📍 Debug: {config.debug}")
    print()

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        reload=config.debug
    )

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\user\Desktop\Vault\Agentic Ai\GoAI\metaqore\main.py