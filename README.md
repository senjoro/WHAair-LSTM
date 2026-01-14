# WHAair-LSTM framework

This project implements an LSTM (Long Short-Term Memory) neural network for time series forecasting using environmental data. The model is built in R with the Keras/TensorFlow framework.

## 📋 Project Overview

The model uses historical environmental variables (relative humidity, temperature, and a custom `WHAair` variable) to predict future values. The architecture includes lagged features to capture temporal dependencies in the data.

## 🏗️ Model Architecture

- **Input Layer**: 3D input with shape (3, 3) representing:
  - 3 time steps (lag periods)
  - 3 features per time step
- **LSTM Layers**:
  - First LSTM: 64 units with dropout (0.1) and recurrent dropout (0.1)
  - Second LSTM: 32 units
- **Dense Layers**:
  - Hidden layer: 16 units with ReLU activation
  - Output layer: 1 unit (linear regression output)
- **Training**:
  - Loss function: Mean Squared Error (MSE)
  - Optimizer: Adam
  - Callbacks: Learning rate scheduler and early stopping

## 📊 Data Preparation

The preprocessing pipeline includes:

1. **Feature Engineering**:
   - Creates lagged variables (lag1, lag2, lag3) for columns 1-3 and 5
   - Lag periods: 1, 2, and 3 time steps

2. **Data Structure**:
   - Training data: 3D array of shape (n_samples, 3, 3)
   - Features used: `rhu_avg`, `tem_avg`, `wha2` (with lag suffixes)
   - Target variable: Univariate time series

## 🚀 Usage

### Prerequisites

```r
# Required R packages
install.packages(c("dplyr", "keras", "ggplot2", "ggpointdensity", 
                   "tidyr", "patchwork", "tensorflow"))
#Loading dataset
read.table("lag_var_dataset.txt")
#Train the model using WHAairLSTM.R
