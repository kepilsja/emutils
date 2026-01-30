import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def _white_to_blue_hue(value):
    """
    Interpolate between white (#ffffff) and #1f77b4.
    
    Parameters:
        value (float): A number between 0 and 1.
                       0 -> white
                       1 -> #1f77b4
    
    Returns:
        str: Hex color string.
    """
    # Clamp value to [0, 1]
    value = max(0.0, min(1.0, value))**2

    # White and target color RGB values
    white = (255, 255, 255)
    target = (0x1f, 0x77, 0xb4)  # (31, 119, 180)

    # Linear interpolation
    r = int(white[0] + value * (target[0] - white[0]))
    g = int(white[1] + value * (target[1] - white[1]))
    b = int(white[2] + value * (target[2] - white[2]))

    return f"#{r:02x}{g:02x}{b:02x}"

def parallel_coordinates_plot(data: pd.DataFrame, metric_range=None,
                              highlight_last=True, highlight_best=True):
    n_rows, n_params = data.shape
    xticks = np.linspace(0, 1, n_params)
    xticklabels = data.columns
    yticks = np.linspace(0, 1, 6)

    fig, ax = plt.subplots(figsize=(n_params/.7, 4))
    ax.spines[['top', 'bottom']].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-.01, 1.01)
    
    ax.set_xticks(xticks, xticklabels)
    ax.set_yticklabels([])
    ax.tick_params('x', size=0)
    
    data_normalized = (data - data.min())/(data.max() - data.min())
    hue = data_normalized.iloc[:,-1]
    if metric_range:
        metric = (data.iloc[:,-1] - metric_range[0])/(metric_range[1] - metric_range[0])
        data_normalized.iloc[:,-1] = metric

    for i, row in data_normalized.sort_values(data.columns[-1]).iterrows():
        ax.plot(xticks, row, color=_white_to_blue_hue(hue[i]))

    if highlight_last:
        ax.plot(xticks, data_normalized.iloc[-1], c='tab:green')
    if highlight_best:
        ax.plot(xticks, data_normalized.sort_values(data.columns[-1]).iloc[-1],
                color='tab:red')
    
    ax.vlines(xticks, 0, 1, color='k', lw=1)
    for i, x in enumerate(xticks):
        ax.hlines(yticks, x-.01, x, color='k', lw=1)
        col = data.iloc[:,i]
        labels = yticks * np.ptp(col) + col.min()
        if i==n_params-1 and metric_range:
            labels = yticks * np.ptp(metric_range) + metric_range[0]
        max_exponent = np.floor(np.log10(abs(labels))).max()
        if abs(max_exponent)>3:
            labels /= 10**max_exponent
            ax.text(x, yticks[-1]+.03, f'1e{int(max_exponent)}', va='bottom', ha='center')
        for y, label in zip(yticks, labels):
            ax.text(x-.016, y, f'{label:.3f}', va='center', ha='right')

    return ax