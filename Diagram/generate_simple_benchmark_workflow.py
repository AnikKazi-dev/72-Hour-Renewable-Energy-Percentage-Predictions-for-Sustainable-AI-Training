#!/usr/bin/env python3
"""
Simplified Benchmark.py Workflow Visualization
============================================

This script generates a concise workflow diagram showing only the main
steps of the benchmarking process in benchmark.py.
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
ax.text(5, 5.5, 'Benchmark.py Workflow (Simplified)', 
        ha='center', va='center', fontsize=16, weight='bold')

# Main workflow steps
# Step 1: Input Data
create_box(ax, 0.5, 4, 2, 0.8, 'INPUT DATA\n\nResults/emissions/\nResults/reports/', colors['input'], 10)

# Step 2: Data Collection
create_box(ax, 3.5, 4, 2.5, 0.8, 'DATA COLLECTION\n\ncollect_emissions()\ncollect_metrics_from_reports()', colors['process'], 9)

# Step 3: Data Processing
create_box(ax, 7, 4, 2.5, 0.8, 'AGGREGATION\n\nCSV Export\nDataFrame Merge', colors['process'], 10)

# Step 4: Analysis & Visualization
create_box(ax, 1.5, 2.5, 3, 0.8, 'ANALYSIS & VISUALIZATION\n\nBox Plots • Heatmaps • Trade-offs\nTop Models • Country Analysis', colors['process'], 9)

# Step 5: Output
create_box(ax, 6, 2.5, 3, 0.8, 'OUTPUT FILES\n\nResults/Benchmark/\n15+ PNG files + CSV reports', colors['output'], 10)

# Final Summary
create_box(ax, 3, 0.8, 4, 0.8, 'COMPREHENSIVE BENCHMARK REPORT\n\nModel Performance • Emissions • Rankings', colors['output'], 10)

# Create main flow arrows
create_arrow(ax, 2.5, 4.4, 3.5, 4.4, 'blue')      # Input → Collection
create_arrow(ax, 6.0, 4.4, 7.0, 4.4, 'green')     # Collection → Aggregation
create_arrow(ax, 4.5, 4.0, 3.5, 3.3, 'orange')    # To Analysis
create_arrow(ax, 8.0, 4.0, 7.5, 3.3, 'orange')    # Aggregation → Output
create_arrow(ax, 6.0, 2.9, 6.0, 1.6, 'red')       # To Final Report

# Add process flow numbers
positions = [(1.5, 4.8), (4.75, 4.8), (8.25, 4.8), (3.0, 3.3), (7.5, 3.3), (5.0, 1.6)]
for i, (x, y) in enumerate(positions, 1):
    circle = plt.Circle((x, y), 0.15, color='red', zorder=10)
    ax.add_patch(circle)
    ax.text(x, y, str(i), ha='center', va='center', fontsize=10, weight='bold', color='white', zorder=11)

# Add key statistics box
stats_text = """Key Features:
• 40+ Model Variants
• 25+ EU Countries
• Summer/Winter Analysis
• Emissions vs Performance
• Automated Visualization"""

ax.text(0.2, 1.8, stats_text, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('benchmark_workflow_simple.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Simplified benchmark workflow diagram saved as 'benchmark_workflow_simple.png'")
plt.show()