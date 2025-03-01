# Time-Energy Model (TEM) reference implementation

This README explains how to use the simplified version of the Time-Energy Model (TEM) selective time series forecasting framework.
Last updated: 2025-02-28

## Overview

The TEM framework has been refactored to simplify usage by hardcoding most parameters while exposing only the most important ones for customization. This makes it easier to run experiments without having to specify dozens of parameters.

## Required setup

1. Make sure to have the required packages installed as defined in the requirements.txt file 
2. Download the dataset files and place them in the `./dataset` folder
 - Access to the datasets can be found in the README here https://github.com/thuml/Time-Series-Library?tab=readme-ov-file
 - Make sure to extract the dataset files and place .csv files in the `./dataset` folder (not the .zip files or folders)

## Required Parameters

When running `run_ebmExp.py`, you only need to specify the following parameters:

1. `--model`: The forecasting model architecture to use
   - Options: `TimesNet`, `Autoformer`, `Informer`, `FEDformer`, `PatchTST`

2. `--data_path`: The dataset file to use
   - Examples: `weather.csv`, `exchange_rate.csv`, `ETTh1.csv`, `ETTh2.csv`, `national_illness.csv`
   - The script will automatically determine the appropriate dataset type and configurations

3. `--features`: The type of forecasting task
   - `S`: Univariate forecasting (single variable input, single variable output)
   - `MS`: Multivariate to univariate forecasting (multiple variable input, single variable output)

4. `--output_parent_path`: Directory where experiment results will be saved

## Optional Parameters

5. `--inference_strategy`: The inference method to use for selective inference
   - `noise` (default): Uses Aggregated Energy inference
   - `optim`: Uses Energy Optimization inference

6. For Aggregated Energy inference:
   - `--noisy_std`: Standard deviation for noise (default: 0.1)

7. For Energy Optimization inference:
   - `--inference_steps`: Number of optimization steps (default: 25)
   - `--inference_optim_lr`: Learning rate for optimization (default: 0.01)

8. `--is_test_mode`: Enable test mode
   - `1` (default): Runs with fewer iterations for faster testing
   - `0`: Runs full experiments

## Example Usage

### Basic usage with noise-based inference (default):

```bash
python run_ebmExp.py \
  --model TimesNet \
  --data_path exchange_rate.csv \
  --features S \
  --output_parent_path ./output_results
```

### Using optimization-based inference:

```bash
python run_ebmExp.py \
  --model Autoformer \
  --data_path ETTh1.csv \
  --features MS \
  --inference_strategy optim \
  --inference_steps 50 \
  --inference_optim_lr 0.005 \
  --output_parent_path ./output_results
```

### Custom noise standard deviation:

```bash
python run_ebmExp.py \
  --model PatchTST \
  --data_path ETTh1.csv \
  --features MS \
  --inference_strategy noise \
  --noisy_std 0.2 \
  --output_parent_path ./output_results
```

## Default Values

Most parameters are hardcoded with sensible defaults. The key defaults include:

### General Parameters
- Sequence length: 96 
- Prediction length: 48 
- Batch size: 8 (except for TimesNet which uses 64)
- Training epochs: 30 
- Learning rate: 0.001 

### EBM Parameters
- EBM training epochs: 30
- EBM predictor size: 96
- EBM decoder size: 96
- EBM seed: 2024

### Other Parameters
- d_model: 512 (except for TimesNet which uses 16)
- n_heads: 8
- e_layers: 2
- d_layers: 1
- d_ff: 32 
- dropout: 0.05

For a complete list of default values, refer to the `get_default_args()` function in the `run_ebmExp.py` script.

## Model-Specific Configurations

Each model has specific configurations that are automatically applied when you select the model:

### TimesNet
- batch_size: 64 (overrides default of 8)
- d_model: 16 (overrides default of 512)

## Dataset-Specific Configurations

The script automatically detects the dataset type from the data path and applies appropriate configurations:

### ETT Datasets (ETTh1, ETTh2)
- data: Set to the corresponding ETT dataset name
- enc_in/dec_in: 7 for MS features, 1 for S features

### Exchange Rate Dataset
- data: custom
- enc_in/dec_in: 8, 1 for S features

### Traffic Dataset
- data: custom
- enc_in/dec_in: 862, 1 for S features

### Weather Dataset
- data: custom
- enc_in/dec_in: 21, 1 for S features

### National Illness Dataset
- data: custom
- enc_in/dec_in: 7, 1 for S features
