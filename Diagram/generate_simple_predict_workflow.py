#!/usr/bin/env python3
"""
Simplified Predict.py Workflow Visualization
==========================================

This script generates a concise workflow diagram showing the main
steps of the prediction process in predict.py.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure with a compact size
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Define colors for different components
colors = {
    'input': '#E3F2FD',       # Light blue
    'process': '#E8F5E8',     # Light green
    'model': '#F3E5F5',       # Light purple
    'output': '#FFF3E0',      # Light orange
}

# Helper function to create rounded rectangles
def create_box(ax, x, y, width, height, text, color, text_size=10):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=text_size, wrap=True, weight='bold')
    return box

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, color='black', linewidth=3):
    arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                          arrowstyle='->', shrinkA=5, shrinkB=5,
                          color=color, linewidth=linewidth)
    ax.add_patch(arrow)
    return arrow

# Title
ax.text(5, 5.5, 'Predict.py Workflow (Simplified)', 
        ha='center', va='center', fontsize=16, weight='bold')

# Main workflow steps
# Step 1: Data Fetching
create_box(ax, 0.5, 4, 2, 0.8, 'DATA FETCHING\n\nENTSO-E API\n(or CSV fallback)\nLast 72 hours', colors['input'], 9)

# Step 2: Data Processing
create_box(ax, 3, 4, 2, 0.8, 'DATA PROCESSING\n\nRenewable %\nTime series\nInput window', colors['process'], 9)

# Step 3: Model Loading
create_box(ax, 5.5, 4, 2, 0.8, 'MODEL LOADING\n\nSaved weights\nKeras/TensorFlow\nModel selection', colors['model'], 9)

# Step 4: Prediction
create_box(ax, 8, 4, 1.5, 0.8, 'PREDICTION\n\nNext 72h\nForecast', colors['model'], 9)

# Step 5: Results Processing
create_box(ax, 2, 2.5, 2.5, 0.8, 'RESULTS PROCESSING\n\nStatistics • Plots\nJSON reports', colors['process'], 9)

# Step 6: Output Files
create_box(ax, 5.5, 2.5, 3, 0.8, 'OUTPUT FILES\n\nPredictions/ directory\nforecast_72h.json • plots • reports', colors['output'], 9)

# Final aggregation
create_box(ax, 3, 0.8, 4, 0.8, 'AGGREGATED PREDICTIONS\n\npredictions_<season>.json', colors['output'], 10)

# Create main flow arrows
create_arrow(ax, 2.5, 4.4, 3.0, 4.4, 'blue')      # Data fetch → Processing
create_arrow(ax, 5.0, 4.4, 5.5, 4.4, 'green')     # Processing → Model loading
create_arrow(ax, 7.5, 4.4, 8.0, 4.4, 'purple')    # Model → Prediction
create_arrow(ax, 8.5, 4.0, 4.0, 3.3, 'orange')    # Prediction → Results
create_arrow(ax, 7.0, 2.5, 7.0, 1.6, 'red')       # Results → Output
create_arrow(ax, 5.0, 1.6, 5.0, 1.6, 'red', 0)    # To final aggregation

# Add process flow numbers
positions = [(1.5, 4.8), (4.0, 4.8), (6.5, 4.8), (8.75, 4.8), (3.25, 3.3), (7.0, 3.3), (5.0, 1.6)]
for i, (x, y) in enumerate(positions, 1):
    circle = plt.Circle((x, y), 0.15, color='red', zorder=10)
    ax.add_patch(circle)
    ax.text(x, y, str(i), ha='center', va='center', fontsize=10, weight='bold', color='white', zorder=11)

# Add key features box
features_text = """Key Features:
• Real-time ENTSO-E API
• 72h historical context
• 72h future forecast
• Multiple model support
• Automatic CSV fallback
• Visual plot generation"""

ax.text(0.2, 2.2, features_text, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))

# Add technical details box
tech_text = """Technical Details:
• TensorFlow/Keras models
• SavedModel format
• Country-specific weights
• Season-aware predictions
• JSON + PNG outputs
• Statistical summaries"""

ax.text(7.8, 1.8, tech_text, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.8))

plt.tight_layout()
plt.savefig('predict_workflow_simple.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Simplified predict workflow diagram saved as 'predict_workflow_simple.png'")
plt.show()