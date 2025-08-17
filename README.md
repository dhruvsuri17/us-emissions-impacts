# Regression Analysis Script

This script performs panel regression analysis on balancing authority data to analyze the relationship between renewable energy generation and emissions across different ISOs.

## Overview

The script processes data for seven balancing authorities:
- CAISO (CISO)
- ERCOT (ERCO)
- ISONE (ISNE)
- MISO (MISO)
- NYISO (NYIS)
- PJM (PJM)
- SWPP (SWPP)

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the script from the scripts directory:
```bash
cd scripts
python regression_analysis.py
```

## Output

The script generates:

1. **CSV Results**: Individual regression results for each balancing authority in `../results/regression/`
2. **LaTeX Tables**: Four main tables:
   - `main_regression.tex`: Generation, CO2 emissions, and CO2 emissions intensity
   - `emissions_regression.tex`: CO2, SO2, and NOx emissions
   - `intensity_regression.tex`: CO2, SO2, and NOx emissions intensity
   - `displacement_effectiveness.tex`: Displacement effectiveness for solar and wind

## Data Structure

The script expects the following data structure:
- `../data/CEMS_processed/`: Contains CEMS data for each BA
- `../data/processed/`: Contains generation data for each BA

## Methodology

The script performs panel regression with:
- Fixed effects for facility ID, month, and year
- Log-transformed dependent and independent variables
- Independent variables: solar generation, wind generation, residual demand, wind ramp, solar ramp

## Author

Dhruv Suri 