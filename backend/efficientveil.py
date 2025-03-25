#!/usr/bin/env python3

import sys
import numpy as np
import signal
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend for non-GUI rendering
import matplotlib.pyplot as plt
from collections import Counter
import time
import io
import base64

try:
    import numba
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# Configuration parameters
DIRECT_SAMPLING_PROBABILITY = 0.001  # Probability of using direct sampling instead of batch
BASE_BATCH_SIZE = 10000              # Base size for Cauchy batches
BATCH_SIZE_JITTER = 1000             # Amount of random jitter to apply to batch size

# Add default values at the top of the file with the other configuration parameters
DEFAULT_PHI = 0.001
DEFAULT_PSI = 0.001
DEFAULT_M = 50

# Add this flag at the top with other configuration parameters
ENABLE_JITTER = False  # Set to False to disable jittering

# Set a fixed seed for reproducibility (at the top of the file)
np.random.seed(42)  # Try different seed values

seed_value = int(time.time())  # Or any other seed value
np.random.seed(seed_value)

def signal_handler(sig, frame):
    print("\nExiting gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Note: The lookup table approach has been replaced with a more statistically 
# sound batch approach that maintains independence between samples.

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
    # Add ±10% randomness to batch size
    jitter = np.random.randint(-BATCH_SIZE_JITTER, BATCH_SIZE_JITTER)
    actual_size = max(1000, base_size + jitter)
    return np.random.standard_cauchy(actual_size)

# Initialize the first batch
cauchy_batch = create_cauchy_batch()
batch_index = 0

if HAS_NUMBA:
    @numba.jit(nopython=True)
    def _sample_cauchy_batch_impl(location, scale, value):
        return location + scale * value
        
    def sample_cauchy_batch(location, scale):
        global cauchy_batch, batch_index
        if batch_index >= len(cauchy_batch):
            cauchy_batch = create_cauchy_batch()
            batch_index = 0
        
        value = cauchy_batch[batch_index]
        batch_index += 1
        return _sample_cauchy_batch_impl(location, scale, value)
else:
    def sample_cauchy_batch(location, scale):
        global cauchy_batch, batch_index
        if batch_index >= len(cauchy_batch):
            cauchy_batch = create_cauchy_batch()
            batch_index = 0
        
        value = cauchy_batch[batch_index]
        batch_index += 1
        return location + scale * value

def sample_cauchy(location, scale):
    """
    Sample from a Cauchy distribution with given location and scale.
    
    Occasionally produces extreme values to increase chaos.
    """
    # Rarely (0.5% chance), return an extremely large value
    if np.random.random() < 0.005:
        # Return an extreme value (positive or negative)
        extreme_sign = 1 if np.random.random() < 0.5 else -1
        return location + scale * extreme_sign * np.random.uniform(1000, 10000)
    
    # Occasionally (0.1% chance) use direct sampling
    if np.random.random() < DIRECT_SAMPLING_PROBABILITY:
        return location + scale * np.random.standard_cauchy()
        
    # Otherwise use batch method
    return sample_cauchy_batch(location, scale)

# Add parameter jittering function 
def jitter_parameters(phi=DEFAULT_PHI, psi=DEFAULT_PSI, jitter_pct=0.15):
    """
    Apply subtle jitter to parameters with occasional extreme values.
    
    Args:
        phi, psi: Base parameter values (typically around 0.001)
        jitter_pct: Base percentage for jitter
        
    Returns:
        (float, float): Jittered phi and psi values
    """
    # Most of the time (80%), apply very subtle jitter
    if np.random.random() < 0.80:
        # Subtle jitter: ±15% of the base value
        phi_jittered = phi * (1.0 + np.random.uniform(-jitter_pct, jitter_pct))
        psi_jittered = psi * (1.0 + np.random.uniform(-jitter_pct, jitter_pct))
    
    # Occasionally (15%), make phi much smaller (creates more stable chains)
    elif np.random.random() < 0.15:
        phi_jittered = phi * np.random.uniform(0.1, 0.5)  # 10-50% of original
        psi_jittered = psi * (1.0 + np.random.uniform(-jitter_pct, jitter_pct))
    
    # Rarely (5%), make phi much larger (creates more chaotic chains)
    else:
        phi_jittered = phi * np.random.uniform(2.0, 10.0)  # 2-10x larger
        psi_jittered = psi * (1.0 + np.random.uniform(-jitter_pct, jitter_pct))
    
    # Ensure parameters don't become negative
    return max(0.00001, phi_jittered), max(0.00001, psi_jittered)

def generate_second_order_cauchy_chain_until_M(phi=DEFAULT_PHI, psi=DEFAULT_PSI, M=DEFAULT_M, max_steps=1_000_000):
    # Only apply jitter if enabled
    if ENABLE_JITTER:
        phi, psi = jitter_parameters(phi, psi)
    
    # Pre-allocate array with a reasonable initial size
    initial_size = min(1000, max_steps)  # Start with 1000 or max_steps, whichever is smaller
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

def generate_second_order_cauchy_chain_fixed_length(phi=DEFAULT_PHI, psi=DEFAULT_PSI, n=10000):
    # Only apply jitter if enabled
    if ENABLE_JITTER:
        phi, psi = jitter_parameters(phi, psi)
    
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

def reindex_and_map(Z_list, J_list, L, U):
    """
    Vectorized version of reindex_and_map for better performance.
    
    Args:
        Z_list: First Cauchy chain (numpy array or list)
        J_list: Second Cauchy chain (numpy array or list)
        L: Lower value to map to
        U: Upper value to map to
        
    Returns:
        list: Final mapped values (L or U)
    """
    # Convert inputs to numpy arrays if they aren't already
    Z_array = Z_list if isinstance(Z_list, np.ndarray) else np.array(Z_list)
    J_array = J_list if isinstance(J_list, np.ndarray) else np.array(J_list)
    
    n_prime = len(Z_array)  # realized sample size
    
    # Map J values to indices
    u_values = safe_logistic_cdf(J_array)
    indices = np.floor(u_values * n_prime).astype(int)
    indices = np.clip(indices, 0, n_prime - 1)  # ensure valid indices
    
    # Get corresponding Z values
    z_prime_values = Z_array[indices]
    
    # Map to L or U based on logistic CDF
    cdf_values = safe_logistic_cdf(z_prime_values)
    final_values = np.where(cdf_values < 0.5, L, U)
    
    return final_values.tolist()  # Convert back to list for compatibility

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

def plot_runs(all_runs, L, U, grid_size=(5, 5), samples_per_plot=4000):
    """
    Given a list of binary outcomes (0s and 1s), create a grid of bar charts
    showing the distribution in each subplot.
    
    Args:
        all_runs: List of binary outcomes (0s and 1s)
        L: First value to display (typically input1)
        U: Second value to display (typically input2)
        grid_size: Tuple of (rows, cols) for the grid layout
        samples_per_plot: Number of samples to include in each subplot
        
    Returns:
        matplotlib Figure object
    """
    rows, cols = grid_size
    total_plots = rows * cols
    
    # Website color scheme
    purple_color = '#9370DB'  # Purple from the F in FR
    teal_color = '#00CED1'    # Teal from the R in FR
    background_color = '#111111'  # Dark background like the website
    text_color = '#FFFFFF'    # White text
    
    # Create figure with dark background
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    fig.patch.set_facecolor(background_color)
    
    # Adjust the figure title
    fig.suptitle(f'Distribution between {L} and {U} across {total_plots} Samples', 
                fontsize=24, color=text_color, y=0.95)
    
    # Flatten the axes array for easier iteration
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1:
        axes = [axes[i] for i in range(cols)]
    elif cols == 1:
        axes = [axes[i] for i in range(rows)]
    
    # Split the runs into chunks for each subplot
    for i in range(total_plots):
        ax = axes[i]
        
        # Set dark background for subplot
        ax.set_facecolor(background_color)
        
        # Calculate start and end indices for this subplot
        start_idx = i * samples_per_plot
        end_idx = min(start_idx + samples_per_plot, len(all_runs))
        
        if start_idx >= len(all_runs):
            # Hide empty subplots
            ax.axis('off')
            continue
            
        # Get the chunk of data for this subplot
        chunk = all_runs[start_idx:end_idx]
        
        # Count occurrences (0 maps to L, 1 maps to U)
        counts = {0: chunk.count(0), 1: chunk.count(1)}
        
        # Create bar chart with the actual output values
        bars = ax.bar([str(L), str(U)], [counts[0], counts[1]], 
                     color=[purple_color, teal_color], width=0.6)
        
        # Add counts as text on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{height}', ha='center', va='bottom', color=text_color)
        
        # Set title and style
        ax.set_title(f'Samples {start_idx+1}-{end_idx}', color=text_color)
        ax.set_ylim(0, samples_per_plot * 0.8)  # Leave room for text
        
        # Add percentage text
        if sum(counts.values()) > 0:
            ratio = counts[1] / sum(counts.values())
            ax.text(0.5, 0.85, f'{ratio:.2%} {U}s', 
                   transform=ax.transAxes, ha='center', 
                   color=text_color, fontsize=12)
        
        # Style the axes
        for spine in ax.spines.values():
            spine.set_color('#333333')
        
        ax.tick_params(colors=text_color, which='both')
        ax.set_xlabel('Value', color=text_color)
        ax.set_ylabel('Count', color=text_color)
        ax.grid(True, linestyle='--', alpha=0.3, color='#333333')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
    return fig 

def plot_sequential_runs(all_runs, L, U, grid_size=(5, 5)):
    """
    Plot the sequential nature of binary runs to show streaks and patterns.
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    fig.suptitle(f'Sequential Values L={L}, U={U}', fontsize=16)
    
    # Prepare axes
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    elif rows == 1:
        axes = [axes[i] for i in range(cols)]
    elif cols == 1:
        axes = [axes[i] for i in range(rows)]
    
    # Determine how to split the data
    total_samples = len(all_runs)
    samples_per_plot = total_samples // (rows * cols)
    
    for i in range(rows * cols):
        ax = axes[i]
        
        # Calculate start and end indices
        start_idx = i * samples_per_plot
        end_idx = min(start_idx + samples_per_plot, total_samples)
        
        if start_idx >= total_samples:
            ax.axis('off')
            continue
        
        # Get sequence for this plot
        sequence = all_runs[start_idx:end_idx]
        
        # Transform sequence to 0/1 for plotting
        plot_data = [1 if x == U else 0 for x in sequence]
        
        # Plot as a line to show sequential pattern
        ax.plot(range(len(plot_data)), plot_data, 'b-', linewidth=0.5)
        
        # Fill between to highlight streaks
        ax.fill_between(range(len(plot_data)), plot_data, alpha=0.3)
        
        # Set plot limits and labels
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([str(L), str(U)])
        ax.set_title(f'Samples {start_idx+1}-{end_idx}')
    
    plt.tight_layout()
    return fig 

def plot_binary_runs(all_runs, L, U):
    """
    Plot binary runs as bar charts in a grid, where each run is a separate experiment.
    
    Args:
        all_runs: List of lists, where each inner list is a complete sequence
        L: Lower value
        U: Upper value
        
    Returns:
        matplotlib Figure object
    """
    rows, cols = 5, 10
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    fig.suptitle(f'Distribution of Values L={L}, U={U} over {len(all_runs)} Replications', fontsize=16)
    
    # Flatten axes if needed
    if rows > 1 and cols > 1:
        axes = axes.flatten()
    
    for i, run in enumerate(all_runs):
        if i >= rows * cols:
            break
            
        ax = axes[i]
        
        # Count occurrences
        counts = Counter(run)
        freq_L = counts.get(L, 0)
        freq_U = counts.get(U, 0)
        
        # Create bar chart
        ax.bar([str(L), str(U)], [freq_L, freq_U], color=['skyblue','salmon'])
        ax.set_title(f'Run {i+1}')
        
        # Show raw counts
        ax.text(0, freq_L+0.05*max(freq_L,freq_U), str(freq_L), ha='center')
        ax.text(1, freq_U+0.05*max(freq_L,freq_U), str(freq_U), ha='center')
    
    plt.tight_layout()
    return fig 