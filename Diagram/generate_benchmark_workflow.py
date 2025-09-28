#!/usr/bin/env python3
"""
Benchmark.py Workflow Visualization
==================================

This script generates a detailed workflow diagram showing the benchmarking
process implemented in benchmark.py, including data collection, aggregation,
and visualization steps.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle, Circle
import numpy as np

# Set up the figure with a larger size for detailed workflow
fig, ax = plt.subplots(1, 1, figsize=(18, 16))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors for different components
colors = {
    'input': '#E3F2FD',       # Light blue for input data
    'processing': '#F3E5F5',  # Light purple for processing
    'analysis': '#E8F5E8',    # Light green for analysis
    'visualization': '#FFF3E0', # Light orange for visualization
    'output': '#FFEBEE',      # Light red for outputs
    'aggregation': '#F0F4C3'  # Light yellow-green for aggregation
}

# Helper function to create rounded rectangles
def create_box(ax, x, y, width, height, text, color, text_size=9, border_color='black'):
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=text_size, wrap=True, weight='bold')
    return box

# Helper function to create diamonds for decision points
def create_diamond(ax, x, y, width, height, text, color, text_size=8):
    diamond = patches.RegularPolygon((x + width/2, y + height/2), 4, 
                                   radius=width/2, orientation=np.pi/4,
                                   facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(diamond)
    ax.text(x + width/2, y + height/2, text, 
            ha='center', va='center', fontsize=text_size, wrap=True, weight='bold')
    return diamond

# Helper function to create circles
def create_circle(ax, x, y, radius, text, color, text_size=8):
    circle = Circle((x, y), radius, facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', fontsize=text_size, wrap=True, weight='bold')
    return circle

# Helper function to create arrows
def create_arrow(ax, x1, y1, x2, y2, color='black', style='->', linewidth=2):
    arrow = ConnectionPatch((x1, y1), (x2, y2), "data", "data",
                          arrowstyle=style, shrinkA=5, shrinkB=5,
                          color=color, linewidth=linewidth)
    ax.add_patch(arrow)
    return arrow

# Title
ax.text(5, 11.5, 'Benchmark.py Workflow: Model Performance Analysis', 
        ha='center', va='center', fontsize=18, weight='bold')

# Stage 1: Input Data Sources
create_box(ax, 0.5, 10.5, 2, 0.6, 'Results/emissions/\n<Model>/<Country>/<Season>/', colors['input'], 8)
create_box(ax, 3, 10.5, 2, 0.6, 'Results/reports/\n<Model>/<Country>/<Season>/', colors['input'], 8)
create_box(ax, 5.5, 10.5, 2, 0.6, '*_emissions.csv\nFiles', colors['input'], 8)
create_box(ax, 7.5, 10.5, 2, 0.6, '*.json\nReport Files', colors['input'], 8)

# Stage 2: Data Collection Functions
create_box(ax, 1, 9.5, 2, 0.6, 'collect_emissions()\nFunction', colors['processing'], 9)
create_box(ax, 4, 9.5, 2.5, 0.6, 'collect_metrics_from_reports()\nFunction', colors['processing'], 9)
create_box(ax, 7, 9.5, 2, 0.6, '_parse_ctx_from_path()\nHelper', colors['processing'], 8)

# Stage 3: Data Parsing and Loading
create_box(ax, 0.5, 8.5, 1.8, 0.6, '_load_emissions_total()\nCSV Parser', colors['processing'], 8)
create_box(ax, 2.8, 8.5, 1.8, 0.6, 'json_load()\nJSON Parser', colors['processing'], 8)
create_box(ax, 5.1, 8.5, 1.8, 0.6, 'Path Structure\nValidation', colors['processing'], 8)
create_box(ax, 7.4, 8.5, 2, 0.6, 'Metrics Extraction\n(MAE, RMSE, etc.)', colors['processing'], 8)

# Stage 4: Data Aggregation
create_box(ax, 1.5, 7.5, 2, 0.6, 'Emissions DataFrame\n(dfe)', colors['aggregation'], 9)
create_box(ax, 4.5, 7.5, 2, 0.6, 'Metrics DataFrame\n(dfm)', colors['aggregation'], 9)
create_box(ax, 7, 7.5, 2, 0.6, 'merge_metrics_emissions()\nCombined Data', colors['aggregation'], 8)

# Decision Points
create_diamond(ax, 1.8, 6.5, 0.8, 0.6, 'dfe\nEmpty?', colors['processing'], 7)
create_diamond(ax, 4.8, 6.5, 0.8, 0.6, 'dfm\nEmpty?', colors['processing'], 7)

# Stage 5: Aggregated CSV Outputs
create_box(ax, 0.5, 5.5, 2, 0.6, 'save_aggregated()\nemissions_aggregated.csv', colors['output'], 8)
create_box(ax, 3, 5.5, 2.5, 0.6, 'save_metrics_aggregated()\nmetrics_aggregated.csv', colors['output'], 8)

# Stage 6: Visualization Functions - Emissions
create_box(ax, 0.2, 4.5, 1.6, 0.5, 'plot_box_all_models()\nBoxplots', colors['visualization'], 7)
create_box(ax, 2, 4.5, 1.6, 0.5, 'plot_heatmaps()\nHeatmaps', colors['visualization'], 7)
create_box(ax, 3.8, 4.5, 1.6, 0.5, 'plot_per_country_box()\nCountry Analysis', colors['visualization'], 7)

# Stage 7: Visualization Functions - Metrics
create_box(ax, 0.2, 3.5, 1.6, 0.5, 'plot_metric_box_all_models()\nMAE/RMSE Boxes', colors['visualization'], 7)
create_box(ax, 2, 3.5, 1.6, 0.5, 'plot_metric_heatmaps()\nMetric Heatmaps', colors['visualization'], 7)
create_box(ax, 3.8, 3.5, 1.6, 0.5, 'plot_top_models_bar()\nTop Models', colors['visualization'], 7)

# Stage 8: Advanced Analysis
create_box(ax, 5.8, 4.5, 1.8, 0.5, 'plot_tradeoff_scatter()\nMAE vs Emissions', colors['analysis'], 7)
create_box(ax, 5.8, 3.5, 1.8, 0.5, 'plot_top10_mae_per_country()\nCountry-specific', colors['analysis'], 7)
create_box(ax, 7.8, 4.5, 1.8, 0.5, 'plot_histogram_mae_per_country()\nMAE Distributions', colors['analysis'], 7)

# Stage 9: Output Files
output_files = [
    'boxplot_emissions_all_models.png',
    'heatmap_mean_emissions_<season>.png',
    'boxplot_emissions_<country>.png',
    'boxplot_mae_all_models.png',
    'heatmap_mean_mae_<season>.png',
    'top10_mae.png',
    'tradeoff_mae_vs_emissions.png',
    'top10_mae_<country>_<season>.png'
]

for i, filename in enumerate(output_files):
    x = 0.3 + (i % 4) * 2.4
    y = 2.3 - (i // 4) * 0.6
    create_box(ax, x, y, 2.2, 0.4, filename, colors['output'], 6)

# Final Output
create_box(ax, 3, 0.8, 4, 0.6, 'Results/Benchmark/ Directory\nComplete Benchmark Report', colors['output'], 10)

# Create arrows for data flow
# Input to collection
create_arrow(ax, 1.5, 10.5, 1.5, 10.1, 'blue')  # emissions input
create_arrow(ax, 4, 10.5, 4.5, 10.1, 'blue')    # reports input
create_arrow(ax, 6.5, 10.5, 6.5, 10.1, 'blue')  # csv files
create_arrow(ax, 8.5, 10.5, 7.5, 10.1, 'blue')  # json files

# Collection to parsing
create_arrow(ax, 2, 9.5, 2, 9.1, 'purple')      # collect_emissions
create_arrow(ax, 5.25, 9.5, 5.25, 9.1, 'purple') # collect_metrics

# Parsing to aggregation
create_arrow(ax, 1.4, 8.5, 1.8, 8.1, 'purple')
create_arrow(ax, 3.7, 8.5, 4.2, 8.1, 'purple')
create_arrow(ax, 6, 8.5, 5.8, 8.1, 'purple')
create_arrow(ax, 8.4, 8.5, 8, 8.1, 'purple')

# Aggregation to decisions
create_arrow(ax, 2.5, 7.5, 2.2, 7.1, 'green')
create_arrow(ax, 5.5, 7.5, 5.2, 7.1, 'green')

# Decisions to outputs
create_arrow(ax, 1.8, 6.5, 1.5, 6.1, 'orange')
create_arrow(ax, 4.8, 6.5, 4.25, 6.1, 'orange')

# Outputs to visualizations
create_arrow(ax, 1.5, 5.5, 1.5, 5.0, 'red')
create_arrow(ax, 4.25, 5.5, 4.25, 5.0, 'red')

# Visualizations to final output
create_arrow(ax, 3, 4.0, 3.5, 1.4, 'red')
create_arrow(ax, 5, 4.0, 5, 1.4, 'red')
create_arrow(ax, 7, 4.0, 6.5, 1.4, 'red')

# Add legend
legend_elements = [
    patches.Patch(color=colors['input'], label='Input Data Sources'),
    patches.Patch(color=colors['processing'], label='Data Collection & Processing'),
    patches.Patch(color=colors['aggregation'], label='Data Aggregation'),
    patches.Patch(color=colors['analysis'], label='Advanced Analysis'),
    patches.Patch(color=colors['visualization'], label='Visualization Generation'),
    patches.Patch(color=colors['output'], label='Output Files')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.35))

# Add detailed process description
description = """
Benchmark.py Process Flow:

1. Data Collection:
   • Scans Results/emissions/ for CSV files
   • Scans Results/reports/ for JSON metrics
   • Validates file paths and structure

2. Data Processing:
   • Parses emissions data (CO₂, energy, duration)
   • Extracts metrics (MAE, RMSE, MSE, R², MAPE)
   • Aggregates by model/country/season

3. Analysis & Visualization:
   • Box plots for distribution analysis
   • Heatmaps for model × country comparison
   • Trade-off analysis (accuracy vs emissions)
   • Top performing models identification

4. Output Generation:
   • Multiple PNG visualizations
   • Aggregated CSV reports
   • Country-specific analyses
   • Season-based comparisons
"""

ax.text(0.1, 0.5, description, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))

# Add process statistics
stats_text = """
Key Statistics Tracked:
• 40+ Model Variants
• 25+ European Countries  
• Summer/Winter Seasons
• Emissions (kg CO₂e)
• Energy (kWh)
• Training Duration
• MAE, RMSE, MSE, R², MAPE
• Model Rankings
"""

ax.text(7.5, 2.5, stats_text, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcyan', alpha=0.8))

plt.tight_layout()
plt.savefig('benchmark_workflow_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('benchmark_workflow_diagram.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Benchmark workflow diagram saved as 'benchmark_workflow_diagram.png' and 'benchmark_workflow_diagram.pdf'")
plt.show()