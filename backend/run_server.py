#!/usr/bin/env python3

import os
import sys

def main():
    """
    Starts the API server for the FR Machine II Demo
    """
    print("Starting FR Machine II Demo API Server...")
    
    # Add enclave directory to path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'enclave'))
    
    # Import and run the API server
    from enclave.api_server import app
    app.run(debug=True, port=5001, host='0.0.0.0')

if __name__ == "__main__":
    main() 