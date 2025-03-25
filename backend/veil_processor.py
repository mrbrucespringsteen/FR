#!/usr/bin/env python3

import numpy as np
import json
from efficientveil import (
    generate_second_order_cauchy_chain_until_M,
    generate_second_order_cauchy_chain_fixed_length,
    reindex_and_map
)

def generate_extreme_runs(phi, psi, num_runs=50, samples_per_run=10000):
    """
    Generate runs with extreme distributions by directly implementing app.py's approach.
    """
    all_runs = []
    
    for _ in range(num_runs):
        # Generate Z chain
        Z_chain = np.zeros(1000)  # Pre-allocate
        Z_chain[0] = np.random.standard_cauchy()
        Z_chain[1] = np.random.standard_cauchy() + Z_chain[0]
        
        t = 2
        while t < 1000 and abs(Z_chain[t-1]) <= 50:
            scale_t = phi * abs(Z_chain[t-2]) + psi
            Z_chain[t] = np.random.standard_cauchy() * scale_t + Z_chain[t-1]
            t += 1
        
        Z_chain = Z_chain[:t]  # Truncate
        
        # Generate J chain
        J_chain = np.zeros(samples_per_run)
        J_chain[0] = np.random.standard_cauchy()
        J_chain[1] = np.random.standard_cauchy() + J_chain[0]
        
        for k in range(2, samples_per_run):
            scale_k = phi * abs(J_chain[k-2]) + psi
            J_chain[k] = np.random.standard_cauchy() * scale_k + J_chain[k-1]
        
        # Map to binary
        n_prime = len(Z_chain)
        u_values = 1.0 / (1.0 + np.exp(-np.clip(J_chain, -30, 30)))
        indices = np.clip(np.floor(u_values * n_prime).astype(int), 0, n_prime-1)
        z_prime_values = Z_chain[indices]
        cdf_values = 1.0 / (1.0 + np.exp(-np.clip(z_prime_values, -30, 30)))
        binary_values = np.where(cdf_values < 0.5, 0, 1).tolist()
        
        all_runs.append(binary_values)
    
    return all_runs

def process_data(input1, input2, mode='single'):
    """
    Process inputs using the efficient veil algorithm.
    
    Args:
        input1: First numeric input (used as phi parameter and as first output value)
        input2: Second numeric input (used as psi parameter and as second output value)
        mode: 'single' for one result, 'multiple' for 100,000 samples
        
    Returns:
        For single mode: A single numeric result (either input1 or input2)
        For multiple mode: A dict with raw results data
    """
    # Convert inputs to appropriate parameters but use very small values
    phi = 0.001  # Fixed small value like in app.py
    psi = 0.001  # Fixed small value like in app.py
    
    # Add slight variation based on inputs (optional)
    phi += float(input1) / 1e14  # Tiny adjustment based on input
    psi += float(input2) / 1e14  # Tiny adjustment based on input
    
    # No clamping - let the jittering in efficientveil.py handle the variation
    
    # Define L and U values for the algorithm (0 and 1)
    L, U = 0, 1
    
    # Define the actual output values (the user's inputs)
    output_L, output_U = int(input1), int(input2)
    
    # Add this near the other parameter initializations
    SHOW_EXTREME_RUNS_ONLY = True  # Set to True to filter for extreme runs
    
    if mode == 'single':
        # Generate a single result
        M = 50  # Threshold for stopping (same as app.py)
        Z_chain, length = generate_second_order_cauchy_chain_until_M(phi, psi, M)
        J_chain = generate_second_order_cauchy_chain_fixed_length(phi, psi, length)
        
        # Get just one mapped value (0 or 1)
        binary_result = reindex_and_map(Z_chain, [J_chain[-1]], L, U)[0]
        
        # Map the binary result to the actual output value
        result = output_L if binary_result == L else output_U
        return result
    else:
        # Generate multiple results
        all_runs = generate_extreme_runs(phi, psi, num_runs=50, samples_per_run=10000)
        
        # Calculate statistics for each run
        grid_stats = []
        total_count_0 = 0
        total_count_1 = 0
        
        for i, run in enumerate(all_runs):
            # Count occurrences in this run
            count_0 = run.count(0)
            count_1 = run.count(1)
            
            # Update totals
            total_count_0 += count_0
            total_count_1 += count_1
            
            # Add run statistics
            grid_stats.append({
                'batch_number': i + 1,
                'sample_range': [i * 10000 + 1, (i + 1) * 10000],
                'counts': {
                    'first': count_0,
                    'second': count_1
                },
                'ratio': count_1 / len(run) if run else 0
            })
        
        # Flatten all runs for total count (if needed)
        binary_results = [item for sublist in all_runs for item in sublist]
        
        # Run extremity analysis
        print("DEBUG - Run extremity analysis:")
        extreme_runs = 0
        for i, run in enumerate(all_runs):
            count_0 = run.count(0) 
            count_1 = run.count(1)
            ratio = max(count_0, count_1) / len(run)
            if ratio > 0.99:
                extreme_runs += 1
                print(f"Run {i+1}: {ratio:.2%} extreme ({count_0} vs {count_1})")

        print(f"Found {extreme_runs} extreme runs out of {len(all_runs)}")

        # Then, right after generating all runs but before creating grid_stats
        if SHOW_EXTREME_RUNS_ONLY:
            filtered_runs = []
            for run in all_runs:
                count_0 = run.count(0)
                count_1 = run.count(1)
                ratio = max(count_0, count_1) / len(run)
                if ratio > 0.95:  # If run is >95% one value
                    filtered_runs.append(run)
            
            # If we found enough extreme runs, use those
            if len(filtered_runs) >= 25:
                all_runs = filtered_runs[:25]  # Use the first 25 extreme runs
                print(f"FILTERED: Using {len(all_runs)} extreme runs")

        # Return summary statistics and grid data
        return {
            'total_samples': len(binary_results),
            'counts': {
                'first': total_count_0,
                'second': total_count_1
            },
            'overall_ratio': total_count_1 / len(binary_results) if binary_results else 0,
            'grid_stats': grid_stats,
            'output_values': [output_L, output_U],
            'all_runs': all_runs,  # Include all runs for plotting if needed
            'extreme_runs': extreme_runs,
            'extreme_threshold': 0.99,
            'num_runs': len(all_runs)
        } 