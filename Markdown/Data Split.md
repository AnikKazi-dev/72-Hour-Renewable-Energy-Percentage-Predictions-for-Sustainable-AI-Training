# Data Split Methodology in BioNet Project

## Overview

This document describes the comprehensive data splitting strategy used across all machine learning models in the BioNet renewable energy forecasting project. The project implements a consistent, time-series-aware approach to data splitting that ensures temporal integrity while providing adequate training, validation, and testing datasets.

## Data Sources and Structure

### Raw Data Organization

- **Data Files**: 8 CSV files covering 4 countries (DE, DK, ES, HU) with seasonal splits
- **File Pattern**: `energy_data_{COUNTRY_CODE}_{YEARS}years_{season}.csv`
- **Seasons**:
  - Summer: April-September (months 4-9)
  - Winter: October-March (months 10-3)
- **Temporal Resolution**: Hourly data points
- **Time Span**: 5 years of historical data per file
- **Data Size**: ~21,133 hourly records per seasonal dataset (for Germany summer data)

### Data Features

- **Primary Feature**: `renewable_percentage` - the percentage of renewable energy in the grid
- **Index**: `startTime` - timestamp with timezone information (UTC+00:00)
- **Single Feature Focus**: All models use only the renewable percentage as input (N_FEATURES = 1)

## Sequence Creation Parameters

### Time Series Configuration

```python
LOOK_BACK = 72          # Use past 72 hours (3 days) of data to predict
FORECAST_HORIZON = 72   # Predict next 72 hours (3 days ahead)
N_FEATURES = 1          # Single feature: renewable_percentage
```

### Sequence Generation Process

The `create_sequences()` function transforms the time series data into supervised learning format:

1. **Input Sequences (X)**: 72 consecutive hourly values representing 3 days of historical renewable energy percentages
2. **Target Sequences (y)**: 72 consecutive future hourly values representing 3 days of forecasted renewable energy percentages
3. **Sliding Window**: Creates overlapping sequences with a step size of 1 hour

**Example**:

- Sequence 1: Hours 1-72 → Hours 73-144
- Sequence 2: Hours 2-73 → Hours 74-145
- Sequence 3: Hours 3-74 → Hours 75-146
- And so on...

## Data Preprocessing Pipeline

### 1. Seasonal Filtering

```python
# Filter data to include only relevant seasonal months
seasonal_data = cached_data[cached_data.index.month.isin(MONTHS)]
```

- **Summer months**: [4, 5, 6, 7, 8, 9] (April to September)
- **Winter months**: [10, 11, 12, 1, 2, 3] (October to March)

### 2. Normalization

```python
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data_values = scaler.fit_transform(data_for_scaling).flatten()
```

- **Method**: Min-Max scaling to range [0, 1]
- **Scope**: Applied to the entire seasonal dataset before splitting
- **Consistency**: Same scaler used for training, validation, and testing

### 3. Sequence Creation

Data is transformed from time series format to supervised learning sequences using a sliding window approach.

## Train-Validation-Test Split Strategy

### Split Ratios (Consistent Across All Models)

- **Training Set**: 70% of sequences
- **Validation Set**: 15% of sequences
- **Test Set**: 15% of sequences

### Chronological Split Implementation

```python
# Chronological split for time series
train_size_idx = int(len(X_seq) * 0.70)
valid_size_idx = int(len(X_seq) * 0.15)

X_train, y_train = X_seq[:train_size_idx], y_seq[:train_size_idx]
X_valid, y_valid = X_seq[train_size_idx : train_size_idx + valid_size_idx], y_seq[train_size_idx : train_size_idx + valid_size_idx]
X_test, y_test = X_seq[train_size_idx + valid_size_idx:], y_seq[train_size_idx + valid_size_idx:]
```

### Temporal Integrity Preservation

- **No Random Shuffling**: Data splits maintain chronological order
- **Sequential Assignment**: Earlier time periods for training, later for validation/testing
- **No Data Leakage**: Future data never used to predict past values
- **Realistic Evaluation**: Test set represents truly unseen future periods

## Split Characteristics by Dataset Size

### Approximate Split Sizes (Example: German Summer Data)

With ~21,133 total hourly data points and creating sequences with LOOK_BACK=72 and FORECAST_HORIZON=72:

- **Total Sequences**: ~20,989 sequences (21,133 - 72 - 72 + 1)
- **Training Sequences**: ~14,692 sequences (70%)
- **Validation Sequences**: ~3,148 sequences (15%)
- **Test Sequences**: ~3,149 sequences (15%)

### Temporal Coverage

- **Training Period**: Covers the earliest ~70% of the seasonal timespan
- **Validation Period**: Covers the middle ~15% of the seasonal timespan
- **Test Period**: Covers the latest ~15% of the seasonal timespan

## Model-Specific Considerations

### Input Shape Variations

While the split methodology is consistent, different models handle input reshaping differently:

1. **DLinear Models**: Use 2D input `(batch_size, look_back)`
2. **CNN/LSTM Models**: Use 3D input `(batch_size, look_back, n_features)`
3. **Transformer Models**: May require additional positional encoding

### Consistency Across Model Types

All models in the project (14+ different architectures) implement identical splitting logic:

- Autoformer, CarbonCast, CNN-LSTM, Cycle-LSTM, DLinear
- EnsembleCI, Hybrid CNN-CycleLSTM-Attention, Informer
- Mamba, N-BEATS, PatchTST, Temporal Fusion Transformer, Transformer

## Validation and Quality Checks

### Data Availability Verification

```python
if len(data_values_scaled) < look_back + forecast_horizon:
    print(f"Not enough data to create sequences...")
    return np.array([]), np.array([])
```

### Split Size Reporting

Each model reports split dimensions:

```python
print(f"\nData Split:")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
```

## Multi-Country and Multi-Season Execution

### Country-Specific Processing

The main.py orchestrates training across multiple countries:

- Each country's data is processed independently
- Same split ratios applied to each country's seasonal data
- Models trained separately for each country-season combination

### Seasonal Processing

- Summer and winter data processed as separate experiments
- Seasonal filtering ensures only relevant months are included
- Environment variables control which season is active during training

## Best Practices and Rationale

### Why Chronological Splitting?

1. **Temporal Dependencies**: Time series data has inherent temporal patterns
2. **Realistic Evaluation**: Mimics real-world deployment where models predict future from past
3. **Avoiding Look-Ahead Bias**: Prevents the model from accidentally learning future patterns
4. **Seasonal Consistency**: Maintains seasonal patterns within each split

### Why 70-15-15 Split?

1. **Adequate Training Data**: 70% provides sufficient historical patterns for learning
2. **Balanced Validation**: 15% offers meaningful validation set for hyperparameter tuning
3. **Robust Testing**: 15% provides sufficient test data for reliable performance assessment
4. **Industry Standard**: Aligns with common practice in time series forecasting

### Data Preprocessing Benefits

1. **Normalization**: Improves model convergence and stability
2. **Seasonal Filtering**: Focuses learning on relevant temporal patterns
3. **Sequence Creation**: Transforms time series into supervised learning problem
4. **Consistent Pipeline**: Ensures reproducible results across all models

## Summary

The BioNet project implements a sophisticated, consistent data splitting methodology that:

- Preserves temporal integrity through chronological splitting
- Maintains consistent 70-15-15 ratios across all models
- Processes seasonal data independently for focused learning
- Supports multi-country comparative analysis
- Follows time series forecasting best practices
- Enables fair comparison between different model architectures

This approach ensures that all model comparisons are conducted on identical data splits, providing reliable and reproducible benchmarking results for renewable energy forecasting performance.
