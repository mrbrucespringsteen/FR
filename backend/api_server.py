#!/usr/bin/env python3

# Make sure matplotlib use Agg backend before importing pyplot
import os
os.environ['MPLBACKEND'] = 'Agg'

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
matplotlib.rcParams['figure.max_open_warning'] = 0

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Add this import for CORS support
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from collections import Counter
import sys
import traceback  # For better error reporting

# Add parent directory to path to import pristineapp functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Print the current path for debugging
print("Python path:", sys.path)

# Try importing from pristineapp with more detailed error handling
try:
    # Import core functionality from pristineapp
    from pristineapp import (
        sample_cauchy,
        safe_logistic_cdf,
        generate_second_order_cauchy_chain_until_M,
        generate_second_order_cauchy_chain_fixed_length,
        reindex_and_map_binary,
        plot_binary_runs
    )
    print("Successfully imported pristineapp functions")
except ImportError as e:
    print(f"ERROR importing pristineapp: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    print(traceback.format_exc())
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Add explicit CORS headers
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/api/test', methods=['GET'])
def test():
    """Simple test endpoint to verify server is working"""
    return jsonify({
        'success': True,
        'message': 'API server is working!'
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate PNG visualization based on input values"""
    try:
        # Parse JSON data from request
        data = request.json
        print(f"Received data: {data}")
        input1 = float(data.get('input1', 0))
        input2 = float(data.get('input2', 0))
        
        # Set these inputs as L and U values
        L = input1
        U = input2
        
        print(f"Generating visualization for L={L}, U={U}")
        
        # Use same parameters as pristineapp
        phi = 0.001
        psi = 0.001
        phi_prime = 0.001
        psi_prime = 0.001
        M = 50
        n = 10000
        
        # Number of runs for the grid
        num_runs = 50
        
        all_runs = []
        
        for i in range(num_runs):
            # 1) Generate the first chain until |Z_t|>M
            Z_list, stop_time = generate_second_order_cauchy_chain_until_M(phi, psi, M)
            
            # 2) Generate the second chain of length n
            J_list = generate_second_order_cauchy_chain_fixed_length(phi_prime, psi_prime, n)
            
            # 3) Reindex & map to binary values
            values = reindex_and_map_binary(Z_list, J_list, L, U)
            
            # Save
            all_runs.append(values)
        
        # Ensure non-interactive backend is used
        plt.ioff()
        
        # Clear any existing plots
        plt.clf()
        
        # Create the binary runs plot with explicit figure creation
        fig = plt.figure(figsize=(40, 30))
        rows, cols = 5, 10
        fig.suptitle(f'Distribution of Values L={L}, U={U} over {len(all_runs)} Replications', fontsize=32)
        
        for i, run in enumerate(all_runs):
            if i >= rows * cols:
                break
                
            ax = fig.add_subplot(rows, cols, i+1)
            
            counts = Counter(run)
            freq_L = counts.get(L, 0)
            freq_U = counts.get(U, 0)
            
            ax.bar([str(L), str(U)], [freq_L, freq_U], color=['skyblue','salmon'])
            ax.set_title(f'Run {i+1}', fontsize=24)
            ax.tick_params(axis='both', which='major', labelsize=18)
            
            ax.text(0, freq_L+0.05*max(freq_L,freq_U), str(freq_L), ha='center', fontsize=18)
            ax.text(1, freq_U+0.05*max(freq_L,freq_U), str(freq_U), ha='center', fontsize=18)
        
        plt.tight_layout()
        
        # Save plot to a byte buffer with increased DPI
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)  # Close the figure to free memory
        
        # Encode the image for web display
        buf.seek(0)
        img_data = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_data
        })
    
    except Exception as e:
        print(f"Error in generate: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/single', methods=['POST'])
def single_draw():
    """Generate a single binary value based on input values"""
    try:
        # Parse JSON data from request
        data = request.json
        print(f"Received data: {data}")
        input1 = float(data.get('input1', 0))
        input2 = float(data.get('input2', 0))
        
        # Set these inputs as L and U values
        L = input1
        U = input2
        
        print(f"Generating single draw for L={L}, U={U}")
        
        # Parameters for the chain
        phi = 0.001
        psi = 0.001
        warmup_steps = 10
        
        # Warm-up: Generate some initial draws and discard
        for _ in range(5):
            _ = np.random.standard_cauchy()
        
        # Start chain with small random perturbation
        initial_offset = np.random.normal(0, 0.1)
        initial_scale = 1.0 + np.random.uniform(-0.1, 0.1)
        Z_prev2 = sample_cauchy(location=initial_offset, scale=initial_scale)
        Z_prev1 = sample_cauchy(location=Z_prev2, scale=1.0)
        
        # Continue chain for warmup_steps
        Z_current = Z_prev1
        for _ in range(warmup_steps):
            scale = phi * abs(Z_prev2) + psi
            Z_current = sample_cauchy(location=Z_prev1, scale=scale)
            Z_prev2, Z_prev1 = Z_prev1, Z_current
        
        # Map final value based on mode (binary)
        p = safe_logistic_cdf(Z_current)
        result = L if p < 0.5 else U
        
        return jsonify({
            'success': True,
            'result': result
        })
    
    except Exception as e:
        print(f"Error in single_draw: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    print("Starting API server on http://localhost:5001")
    app.run(debug=True, port=5001, host='0.0.0.0') 