from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.gridspec import GridSpec
import os

app = Flask(__name__)

# Configuration parameters
DIRECT_SAMPLING_PROBABILITY = 0.001  # Probability of using direct sampling instead of batch
BASE_BATCH_SIZE = 10000              # Base size for Cauchy batches
BATCH_SIZE_JITTER = 1000             # Amount of random jitter to apply to batch size

# Initialize the first batch
cauchy_batch = np.random.standard_cauchy(BASE_BATCH_SIZE)
batch_index = 0

def safe_logistic_cdf(x):
    """
    A safe logistic CDF mapping R -> (0,1), clipped to avoid overflow.
    """
    x_clipped = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def create_cauchy_batch(base_size=BASE_BATCH_SIZE):
    """
    Create a batch of Cauchy samples with randomized size for unpredictability.
    """
    jitter = np.random.randint(-BATCH_SIZE_JITTER, BATCH_SIZE_JITTER)
    actual_size = max(1000, base_size + jitter)
    return np.random.standard_cauchy(actual_size)

def sample_cauchy(location, scale):
    """
    Sample from a Cauchy distribution with given location and scale.
    """
    global cauchy_batch, batch_index
    
    # Occasionally use direct sampling instead of batch
    if np.random.random() < DIRECT_SAMPLING_PROBABILITY:
        return location + scale * np.random.standard_cauchy()
        
    # Otherwise use batch method
    if batch_index >= len(cauchy_batch):
        cauchy_batch = create_cauchy_batch()
        batch_index = 0
    
    value = cauchy_batch[batch_index]
    batch_index += 1
    return location + scale * value

def generate_second_order_cauchy_chain_until_M(phi, psi, M, max_steps=1_000_000):
    """
    Generate a second-order Cauchy chain Z_1, Z_2, ... 
    stopping as soon as |Z_t| > M.
    """
    # Pre-allocate array with a reasonable initial size
    initial_size = min(1000, max_steps)
    Z_array = np.zeros(initial_size)
    
    # Z_0 ~ C[0,1]
    Z_array[0] = sample_cauchy(location=0.0, scale=1.0)
    
    # Z_1 ~ C[Z_0, 1]
    Z_array[1] = sample_cauchy(location=Z_array[0], scale=1.0)
    
    # Check if we already exceed threshold
    if abs(Z_array[1]) > M:
        return Z_array[:2], 2
    
    t = 2  # Next index to fill (0-based)
    while t < max_steps:
        # Resize array if needed
        if t >= len(Z_array):
            Z_array = np.resize(Z_array, min(len(Z_array) * 2, max_steps))
        
        scale_t = phi * abs(Z_array[t-2]) + psi
        loc_t = Z_array[t-1]
        Z_array[t] = sample_cauchy(location=loc_t, scale=scale_t)
        
        if abs(Z_array[t]) > M:
            return Z_array[:t+1], t+1
        
        t += 1
    
    # If we reach here, we never exceeded M in max_steps draws
    return Z_array[:t], t

def generate_second_order_cauchy_chain_fixed_length(phi, psi, n):
    """
    Generate exactly n draws in a second-order Cauchy chain.
    """
    Z_array = np.zeros(n)  # Pre-allocate full array
    
    # First two values
    Z_array[0] = sample_cauchy(location=0.0, scale=1.0)
    Z_array[1] = sample_cauchy(location=Z_array[0], scale=1.0)
    
    # Remaining values
    for k in range(2, n):
        scale_k = phi * abs(Z_array[k-2]) + psi
        loc_k = Z_array[k-1]
        Z_array[k] = sample_cauchy(location=loc_k, scale=scale_k)
    
    return Z_array

def reindex_and_map_binary(Z_list, J_list, L, U):
    """
    Vectorized version of reindex_and_map for binary outputs.
    """
    # Convert inputs to numpy arrays if they aren't already
    Z_array = Z_list if isinstance(Z_list, np.ndarray) else np.array(Z_list)
    J_array = J_list if isinstance(J_list, np.ndarray) else np.array(J_list)
    
    n_prime = len(Z_array)
    
    # Map J values to indices
    u_values = safe_logistic_cdf(J_array)
    indices = np.floor(u_values * n_prime).astype(int)
    indices = np.clip(indices, 0, n_prime - 1)
    
    # Get corresponding Z values
    z_prime_values = Z_array[indices]
    
    # Map to L or U based on logistic CDF
    cdf_values = safe_logistic_cdf(z_prime_values)
    final_values = np.where(cdf_values < 0.5, L, U)
    
    return final_values.tolist()

def reindex_and_map_continuous(Z_list, J_list, a, b):
    """
    Reindex using one chain and map to continuous interval [a, b].
    """
    # Convert inputs to numpy arrays if they aren't already
    Z_array = Z_list if isinstance(Z_list, np.ndarray) else np.array(Z_list)
    J_array = J_list if isinstance(J_list, np.ndarray) else np.array(J_list)
    
    n_prime = len(Z_array)
    
    # Map J values to indices
    u_values = safe_logistic_cdf(J_array)
    indices = np.floor(u_values * n_prime).astype(int)
    indices = np.clip(indices, 0, n_prime - 1)
    
    # Get corresponding Z values
    z_prime_values = Z_array[indices]
    
    # Map to continuous interval [a, b]
    p_values = safe_logistic_cdf(z_prime_values)
    return a + (b - a) * p_values

def plot_binary_runs(all_runs, L, U):
    """
    Plot binary runs as bar charts in a grid.
    """
    rows, cols = 5, 10
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    fig.suptitle(f'Distribution of Values L={L}, U={U} over {len(all_runs)} Replications', fontsize=16)
    
    for i, run in enumerate(all_runs):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        counts = Counter(run)
        freq_L = counts.get(L, 0)
        freq_U = counts.get(U, 0)
        
        ax.bar([str(L), str(U)], [freq_L, freq_U], color=['skyblue','salmon'])
        ax.set_title(f'Run {i+1}')
        # show raw counts:
        ax.text(0, freq_L+0.05*max(freq_L,freq_U), str(freq_L), ha='center')
        ax.text(1, freq_U+0.05*max(freq_L,freq_U), str(freq_U), ha='center')
    
    plt.tight_layout()
    return fig

def plot_continuous_cdfs(all_runs, a, b, num_bins=50):
    """
    Plot continuous runs as CDFs in a grid.
    """
    num_runs = min(len(all_runs), 16)  # Limit to 16 for display
    rows = 4
    cols = 4
    
    # Create bins for histograms
    bins = np.linspace(a, b, num_bins + 1)
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(rows, cols, figure=fig)
    fig.suptitle(f'Cumulative Distributions over {num_runs} Runs (Interval [{a}, {b}])', fontsize=16)
    
    for i in range(num_runs):
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        
        run = all_runs[i]
        
        # Calculate cumulative distribution
        hist, bin_edges = np.histogram(run, bins=bins, density=True)
        cumulative = np.cumsum(hist * np.diff(bin_edges))
        
        # Plot cumulative distribution
        ax.plot(bin_edges[1:], cumulative, 'b-', linewidth=1)
        
        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='r', linestyle='-', alpha=0.5)
        
        # Scale y-axis to [0, 1]
        ax.set_ylim(0, 1)
        
        # Scale x-axis to match the interval
        ax.set_xlim(a, b)
        
        # Add run number as title
        ax.set_title(f'Run {i+1}')
        
        # Remove most tick labels to reduce clutter
        if row < rows - 1:
            ax.set_xticklabels([])
        if col > 0:
            ax.set_yticklabels([])
    
    plt.tight_layout()
    return fig

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get user inputs
    L = float(request.form.get('L'))
    U = float(request.form.get('U'))
    mode = request.form.get('mode')
    
    # Parameters
    phi = 0.001
    psi = 0.001
    phi_prime = 0.001
    psi_prime = 0.001
    M = 50
    n = 10000
    
    # Number of runs depends on mode
    num_runs = 50 if mode == 'binary' else 16  # Binary needs more runs for the grid
    
    all_runs = []
    
    for i in range(num_runs):
        # 1) Generate the first chain until |Z_t|>M
        Z_list, stop_time = generate_second_order_cauchy_chain_until_M(phi, psi, M)
        
        # 2) Generate the second chain of length n
        J_list = generate_second_order_cauchy_chain_fixed_length(phi_prime, psi_prime, n)
        
        # 3) Reindex & map based on mode
        if mode == 'binary':
            values = reindex_and_map_binary(Z_list, J_list, L, U)
        else:  # continuous
            values = reindex_and_map_continuous(Z_list, J_list, L, U)
        
        # Save
        all_runs.append(values)
    
    # Clear any existing plots
    plt.clf()
    
    # Create and save the appropriate plot based on mode
    if mode == 'binary':
        fig = plot_binary_runs(all_runs, L, U)
    else:  # continuous
        fig = plot_continuous_cdfs(all_runs, L, U)
    
    # Save plot to a string buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)  # Close the figure to free memory
    
    # Encode the image for web display
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    buf.close()  # Close the buffer
    
    # Calculate statistics for the runs
    stats = []
    for i, run in enumerate(all_runs):
        if mode == 'binary':
            counts = Counter(run)
            total = len(run)
            freq_L = counts.get(L, 0) / total
            freq_U = counts.get(U, 0) / total
            stats.append({
                'run': i+1,
                'L_freq': f"{freq_L:.3f}",
                'U_freq': f"{freq_U:.3f}"
            })
        else:  # continuous
            mean_val = np.mean(run)
            median_val = np.median(run)
            std_val = np.std(run)
            stats.append({
                'run': i+1,
                'mean': f"{mean_val:.4f}",
                'median': f"{median_val:.4f}",
                'std_dev': f"{std_val:.4f}"
            })
    
    return render_template('results.html', 
                          plot_url=plot_url, 
                          mode=mode, 
                          L=L, 
                          U=U,
                          stats=stats)

@app.route('/single_draw', methods=['POST'])
def single_draw():
    """Generate a single draw either binary or continuous"""
    L = float(request.form.get('L'))
    U = float(request.form.get('U'))
    mode = request.form.get('mode')
    
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
    
    # Map final value based on mode
    p = safe_logistic_cdf(Z_current)
    
    if mode == 'binary':
        result = L if p < 0.5 else U
    else:  # continuous
        result = L + (U - L) * p
    
    return render_template('single_draw.html', 
                          result=result, 
                          mode=mode,
                          L=L,
                          U=U)

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True) 