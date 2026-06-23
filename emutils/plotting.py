import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype
import matplotlib.ticker as ticker

def _white_to_blue_hue(value: float) -> str:
    """
    Interpolate between white (#ffffff) and #1f77b4.
    """
    value = max(0.0, min(1.0, value))**2
    
    white = np.array([255, 255, 255])
    target = np.array([31, 119, 180])
    
    r, g, b = (white + value * (target - white)).astype(int)
    
    return f"#{r:02x}{g:02x}{b:02x}"

def parallel_coordinates_plot(
    data: pd.DataFrame, 
    metric_range: tuple = None,
    highlight_last: bool = True, 
    highlight_best: bool = True
) -> plt.Axes:
    
    n_rows, n_params = data.shape
    xticks = np.linspace(0, 1, n_params)
    xticklabels = data.columns

    fig, ax = plt.subplots(figsize=(n_params / 0.7, 4))
    
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.01, 1.01)
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    
    ax.set_yticks([]) 
    ax.tick_params('x', length=0)

    # 1. Normalize data and capture column metadata for axis drawing
    data_normalized = pd.DataFrame(index=data.index)
    col_info = {}

    for i, col in enumerate(data.columns):
        series = data[col]
        
        # Handle Strings / Categoricals
        if is_string_dtype(series) or series.dtype == object:
            unique_vals = sorted(series.unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            mapped_series = series.map(mapping)
            
            min_val, max_val = 0, len(unique_vals) - 1
            if max_val == 0:
                data_normalized[col] = 0.5  
            else:
                data_normalized[col] = (mapped_series - min_val) / (max_val - min_val)
                
            col_info[col] = {'type': 'string', 'unique_vals': unique_vals}
            
        # Handle Numerics
        else:
            is_int = is_integer_dtype(series)
            min_val, max_val = series.min(), series.max()
            
            if i == n_params - 1 and metric_range:
                # If metric range is explicitly provided, respect it exactly
                nice_min, nice_max = metric_range
                nice_ticks = np.linspace(nice_min, nice_max, 6)
            else:
                # Ask matplotlib for ~5 "pretty" ticks spanning the min/max
                locator = ticker.MaxNLocator(nbins=5, integer=is_int)
                nice_ticks = locator.tick_values(min_val, max_val)
                nice_min, nice_max = nice_ticks[0], nice_ticks[-1]
                
            if nice_max == nice_min:
                data_normalized[col] = 0.5
            else:
                # Normalize the data against the new PRETTY limits, not the exact max/min
                data_normalized[col] = (series - nice_min) / (nice_max - nice_min)
                
            col_info[col] = {
                'type': 'int' if is_int else 'float',
                'ticks': nice_ticks,
                'min': nice_min,
                'max': nice_max
            }

    # 2. Draw lines sorted by the final metric
    hue = data_normalized.iloc[:, -1]
    sorted_indices = data_normalized.sort_values(data.columns[-1]).index

    for idx in sorted_indices:
        row = data_normalized.loc[idx]
        ax.plot(xticks, row, color=_white_to_blue_hue(hue[idx]))

    if highlight_last:
        ax.plot(xticks, data_normalized.iloc[-1], c='tab:green')
    if highlight_best:
        ax.plot(xticks, data_normalized.loc[sorted_indices[-1]], color='tab:red')

    # 3. Draw vertical axes and apply custom tick labels
    ax.vlines(xticks, 0, 1, color='k', lw=1)

    for i, (x, col) in enumerate(zip(xticks, data.columns)):
        info = col_info[col]
        
        if info['type'] == 'string':
            unique_vals = info['unique_vals']
            n_vals = len(unique_vals)
            
            tick_pos = [0.5] if n_vals == 1 else np.linspace(0, 1, n_vals)
            
            ax.hlines(tick_pos, x - 0.01, x, color='k', lw=1)
            for y, label in zip(tick_pos, unique_vals):
                ax.text(x - 0.016, y, str(label), va='center', ha='right')
                
        else:
            nice_ticks = info['ticks']
            nice_min, nice_max = info['min'], info['max']
            
            # Map the true tick values back to their 0-1 normalized Y positions
            tick_pos = (nice_ticks - nice_min) / (nice_max - nice_min)
            
            ax.hlines(tick_pos, x - 0.01, x, color='k', lw=1)
            
            labels = nice_ticks.copy()
            
            max_abs = np.max(np.abs(labels))
            max_exponent = np.floor(np.log10(max_abs)) if max_abs > 0 else 0
            
            eng_exponent = 0
            if abs(max_exponent) >= 3:
                eng_exponent = int((max_exponent // 3) * 3)
                labels /= 10**eng_exponent
                ax.text(x, 1.03, rf"$\times 10^{{{eng_exponent}}}$", va='bottom', ha='center')
                
            for y, label in zip(tick_pos, labels):
                # Format to strip trailing .000s for a much cleaner axis
                if np.isclose(label, np.round(label)):
                    label_str = f"{int(np.round(label))}"
                else:
                    # Keep decimals for floats, but remove unnecessary trailing zeros
                    label_str = f"{label:.3f}".rstrip('0').rstrip('.')
                    
                ax.text(x - 0.016, y, label_str, va='center', ha='right')

    return ax

if __name__ == "__main__":
    # 1. Generate Dummy Data
    np.random.seed(42)
    n_samples = 25

    smoke_data = pd.DataFrame({
        # Test string/categorical parsing
        'Algorithm': np.random.choice(['Random Forest', 'XGBoost', 'SVM', 'Neural Net'], n_samples),
        
        # Test extremely small floats (should trigger x 10^-6)
        'Learning Rate': np.random.uniform(1e-6, 9e-6, n_samples),
        
        # Test exact integers (should not show decimals)
        'Estimators': np.random.randint(10, 250, n_samples),
        
        # Test standard floats
        'Dropout': np.random.uniform(0.1, 0.5, n_samples),
        
        # Test large numbers (should trigger x 10^6)
        'Parameters': np.random.uniform(1e6, 8e6, n_samples),
        
        # Test the target metric used for the hue (Standardized between 0 and 1)
        'Accuracy': np.random.uniform(0.65, 0.99, n_samples)
    })

    # 2. Render the Plot
    # We pass a metric_range for Accuracy just to test the override feature
    ax = parallel_coordinates_plot(
        data=smoke_data,
        metric_range=(0.5, 1.0), 
        highlight_last=True, 
        highlight_best=True
    )

    # 3. Display
    plt.title("Parallel Coordinates Plot - Smoke Test", pad=20)
    plt.tight_layout()
    plt.show()