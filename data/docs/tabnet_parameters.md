# TabNet Integration with Ensemble Prediction Pipeline

The TabNet deep learning model is now integrated with our ensemble prediction pipeline.
This document describes the TabNet-specific parameters and configuration options.

## Overview

TabNet is a deep learning architecture for tabular data that provides:
- High performance with interpretability
- Feature selection capabilities with sparse attention
- Native handling of missing values
- Excellent performance on tabular data without extensive preprocessing

## Automatic Optimization with Optuna

All TabNet parameters are automatically tuned by Optuna during the hyperparameter optimization process.
The tuning system explores a wide range of values for each parameter to find optimal settings
for your specific dataset and prediction task.

## Available Parameters

When configuring a TabNet model in your ensemble, you can use the following parameters:

### Architecture Parameters

| Parameter | Type | Description | Default | Optuna Range |
|-----------|------|-------------|---------|-------------|
| `n_d` | int | Width of the decision prediction layer | 64 | 8-256 (log scale) |
| `n_a` | int | Width of the attention embedding for each mask step | 64 | 8-256 (log scale) |
| `n_steps` | int | Number of steps in the architecture | 5 | 1-15 |
| `gamma` | float | Scaling factor for attention updates | 1.5 | 0.5-3.0 |
| `lambda_sparse` | float | Sparsity regularization parameter | 0.001 | 1e-7 to 1e-1 (log scale) |

### Training Parameters

| Parameter | Type | Description | Default | Optuna Range |
|-----------|------|-------------|---------|-------------|
| `max_epochs` | int | Maximum number of epochs for training | 200 | 50-500 |
| `patience` | int | Early stopping patience (epochs with no improvement) | 15 | 5-50 |
| `batch_size` | int | Batch size for training | 1024 | [128, 256, 512, 1024, 2048, 4096] |
| `virtual_batch_size` | int | Size for Ghost Batch Normalization | 128 | 16-1024 (log scale) |
| `momentum` | float | Momentum for batch normalization | 0.02 | 0.005-0.5 |

### Optimizer Parameters

| Parameter | Type | Description | Default | Optuna Range |
|-----------|------|-------------|---------|-------------|
| `optimizer_lr` | float | Learning rate for optimizer | 0.02 | 1e-5 to 0.5 (log scale) |

## Interpreting TabNet Feature Importance

TabNet provides feature importance scores that indicate how much each feature contributed to the model's decision-making process:

1. **Attention-based importance**: Unlike tree-based models that use gain/split importance, TabNet's feature importance is derived from its sparse attention mechanism.

2. **Mask values**: TabNet creates masks at each decision step that select which features to focus on. The feature importance is the aggregation of these mask values.

3. **Interpretation**: Higher values indicate features that received more attention during prediction.

## Instance-Level Explanations

TabNet also provides per-instance explanations through the `explain()` method:

