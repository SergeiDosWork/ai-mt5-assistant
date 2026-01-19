# Training Module for Neural Network Price Forecasting

## Overview
This module implements a training dataset builder for a neural network designed to forecast short-term futures price movements. The module processes Apache Parquet files containing 1-minute candlestick data and creates training examples with sophisticated features and targets.

## Key Features

### Input Requirements
- **File Format**: Apache Parquet with pyarrow backend
- **Data**: 1-minute candles (M1) of a single instrument
- **Columns**: `time` (datetime64[ns]), `open`, `high`, `low`, `close` (float64), `volume` (int64)

### Feature Engineering

#### Input Features (X) - Total: 358 features
1. **Ticker Name** (1 feature)
   - String identifier extracted from filename (placeholder: 0.0)

2. **60 M3 Candles** (300 features)
   - Each M3 candle has 5 relative change features:
     - `Δopen = open[t] - close[t-1]`
     - `Δhigh = high[t] - close[t-1]`
     - `Δlow = low[t] - close[t-1]`
     - `Δclose = close[t] - close[t-1]`
     - `volume[t]` (absolute value)
   - Total: 60 candles × 5 features = 300 features

3. **10 H1 Candles** (50 features)
   - Aggregated from M3 candles (each H1 = ~20 M3 candles)
   - Aggregation: `open=first, high=max, low=min, close=last, volume=sum`
   - Same 5 relative change features as M3
   - Total: 10 candles × 5 features = 50 features

4. **Technical Indicators** (5 features)
   - RSI(14) on last M3 candle
   - MACD(12,26,9) with all components: MACD line, signal line, histogram
   - Squeeze Momentum Indicator (by LazyBear)

5. **Temporal Features** (2 features)
   - Hour of day as cyclical features:
     - `hour_sin = sin(2π * hour / 24)`
     - `hour_cos = cos(2π * hour / 24)`

#### Target Variables (y) - Total: 20 features

1. **Next 3 M3 Candles** (15 features)
   - For each of the next 3 M3 candles, 5 relative change features
   - Same calculation as input features

2. **Growth Pattern Parameters** (5 features)
   - OHLC changes of the last green candle before a red candle: 4 features
   - Length of consecutive green candles until reversal: 1 feature
   - If no reversal found within 50 candles: [0,0,0,0,0]

## Core Functions

### `build_training_dataset(parquet_path: str) -> Tuple[np.ndarray, np.ndarray]`
Main function that:
- Reads Parquet file with 1-minute data
- Converts to 3-minute (M3) and 1-hour (H1) candles
- Calculates technical indicators using pandas-ta
- Builds training windows with 1-candle shifts
- Validates data and removes NaN entries
- Returns (X, y) tuple of numpy arrays

### `generate_training_samples(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]`
Wrapper function used by the FastAPI application that:
- Takes a DataFrame as input
- Saves temporarily as Parquet
- Calls `build_training_dataset`
- Cleans up temporary files

## Technical Implementation Details

### Data Processing Flow
1. **Resampling**: 1-min → 3-min → 1-hour conversion using pandas resample
2. **Indicator Calculation**: Using pandas-ta library for RSI, MACD, Squeeze
3. **Window Generation**: Sliding windows of 60 M3 candles with 1-candle shifts
4. **Feature Assembly**: Combining all feature types into single vectors
5. **Target Generation**: Calculating future movements and growth patterns
6. **Validation**: Removing samples with NaN values

### Validation Steps
- Column presence validation
- Data type validation
- Sufficient data quantity checks
- NaN filtering post-processing
- Shape consistency verification

### Performance Considerations
- Efficient pandas operations for resampling
- Vectorized calculations for relative changes
- Memory-efficient processing of large datasets
- Proper handling of edge cases and boundaries

## Usage in FastAPI Application
The module integrates with the FastAPI application via the `/train` endpoint:
1. User uploads Parquet file via web interface
2. File is validated for format and content
3. `generate_training_samples` processes the data
4. Resulting (X, y) is passed to the neural network trainer
5. Model is saved and made available for inference

## Dependencies
- pandas (for data manipulation)
- numpy (for numerical operations)
- pandas-ta (for technical indicators)
- pyarrow (for Parquet support)
- Python standard library (math, datetime, etc.)

## Error Handling
- Invalid file format detection
- Insufficient data validation
- Missing column detection
- Boundary condition handling
- NaN value filtering