#!/usr/bin/env python3
"""
Generate Figure 4: Solar and Wind Generation and Emissions Intensity Coefficients
Author: Dhruv Suri

This script creates a 4-panel figure showing the regression coefficients for solar and wind
generation and their effects on emissions intensity across all balancing authorities.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

def extract_coefficient_and_se(value_str):
    """Extract coefficient and standard error from regression result string."""
    if pd.isna(value_str) or value_str == "":
        return 0.0, 0.0
    
    # Pattern: "coefficient (standard_error)***"
    match = re.match(r'(-?\d+\.\d+)\s*\((\d+\.\d+)\)(\**)', str(value_str))
    if match:
        coefficient = float(match.group(1))
        standard_error = float(match.group(2))
        return coefficient, standard_error
    else:
        return 0.0, 0.0

def load_regression_results(ba):
    """Load regression results for a specific balancing authority."""
    file_path = f"../results/regression/{ba}_regression_results.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0)
    else:
        print(f"Warning: {file_path} not found")
        return None

def main():
    # Data for each BA
    BAs = ['CAISO', 'PJM', 'ISONE', 'MISO', 'SWPP', 'ERCOT', 'NYISO']
    x = np.arange(len(BAs))  # label locations
    
    # Initialize arrays for coefficients and standard errors
    solar_coefficients_gen = []
    solar_se_gen = []
    wind_coefficients_gen = []
    wind_se_gen = []
    
    solar_coefficients_co2 = []
    solar_se_co2 = []
    wind_coefficients_co2 = []
    wind_se_co2 = []
    
    solar_coefficients_so2 = []
    solar_se_so2 = []
    wind_coefficients_so2 = []
    wind_se_so2 = []
    
    solar_coefficients_nox = []
    solar_se_nox = []
    wind_coefficients_nox = []
    wind_se_nox = []
    
    # Extract coefficients and standard errors from regression results
    for ba in BAs:
        df = load_regression_results(ba)
        if df is not None:
            # Generation coefficients
            solar_gen_coef, solar_gen_se = extract_coefficient_and_se(
                df.loc['solar_generation_mw', 'gross_load_mw'] if 'gross_load_mw' in df.columns else None
            )
            wind_gen_coef, wind_gen_se = extract_coefficient_and_se(
                df.loc['wind_generation_mw', 'gross_load_mw'] if 'gross_load_mw' in df.columns else None
            )
            
            # CO2 emissions intensity coefficients
            solar_co2_coef, solar_co2_se = extract_coefficient_and_se(
                df.loc['solar_generation_mw', 'co2_emissions_intensity'] if 'co2_emissions_intensity' in df.columns else None
            )
            wind_co2_coef, wind_co2_se = extract_coefficient_and_se(
                df.loc['wind_generation_mw', 'co2_emissions_intensity'] if 'co2_emissions_intensity' in df.columns else None
            )
            
            # SO2 emissions intensity coefficients
            solar_so2_coef, solar_so2_se = extract_coefficient_and_se(
                df.loc['solar_generation_mw', 'so2_emissions_intensity'] if 'so2_emissions_intensity' in df.columns else None
            )
            wind_so2_coef, wind_so2_se = extract_coefficient_and_se(
                df.loc['wind_generation_mw', 'so2_emissions_intensity'] if 'so2_emissions_intensity' in df.columns else None
            )
            
            # NOx emissions intensity coefficients
            solar_nox_coef, solar_nox_se = extract_coefficient_and_se(
                df.loc['solar_generation_mw', 'nox_emissions_intensity'] if 'nox_emissions_intensity' in df.columns else None
            )
            wind_nox_coef, wind_nox_se = extract_coefficient_and_se(
                df.loc['wind_generation_mw', 'nox_emissions_intensity'] if 'nox_emissions_intensity' in df.columns else None
            )
        else:
            # Default values if file not found
            solar_gen_coef, solar_gen_se = 0.0, 0.0
            wind_gen_coef, wind_gen_se = 0.0, 0.0
            solar_co2_coef, solar_co2_se = 0.0, 0.0
            wind_co2_coef, wind_co2_se = 0.0, 0.0
            solar_so2_coef, solar_so2_se = 0.0, 0.0
            wind_so2_coef, wind_so2_se = 0.0, 0.0
            solar_nox_coef, solar_nox_se = 0.0, 0.0
            wind_nox_coef, wind_nox_se = 0.0, 0.0
        
        # Append to arrays
        solar_coefficients_gen.append(solar_gen_coef)
        solar_se_gen.append(solar_gen_se)
        wind_coefficients_gen.append(wind_gen_coef)
        wind_se_gen.append(wind_gen_se)
        
        solar_coefficients_co2.append(solar_co2_coef)
        solar_se_co2.append(solar_co2_se)
        wind_coefficients_co2.append(wind_co2_coef)
        wind_se_co2.append(wind_co2_se)
        
        solar_coefficients_so2.append(solar_so2_coef)
        solar_se_so2.append(solar_so2_se)
        wind_coefficients_so2.append(wind_so2_coef)
        wind_se_so2.append(wind_so2_se)
        
        solar_coefficients_nox.append(solar_nox_coef)
        solar_se_nox.append(solar_nox_se)
        wind_coefficients_nox.append(wind_nox_coef)
        wind_se_nox.append(wind_nox_se)
    
    # Print extracted values for verification
    print("Extracted coefficients and standard errors:")
    print(f"Solar Generation: {solar_coefficients_gen}")
    print(f"Solar Generation SE: {solar_se_gen}")
    print(f"Wind Generation: {wind_coefficients_gen}")
    print(f"Wind Generation SE: {wind_se_gen}")
    
    # Width of bars
    width = 0.25  # the width of the bars
    
    # Set up the figure and axes for subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    
    # Subplot 1: Solar and Wind Generation Coefficients
    axs[0].bar(x - width/2, solar_coefficients_gen, yerr=solar_se_gen, width=width, label='Solar', capsize=5, color='#ee9a01', alpha=0.5)
    axs[0].bar(x + width/2, wind_coefficients_gen, yerr=wind_se_gen, width=width, label='Wind', capsize=5, color='#0273b2', alpha=0.5)
    axs[0].set_title('Generation')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels('')
    axs[0].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    
    # Subplot 2: Solar and Wind CO2 EI Coefficients
    axs[1].bar(x - width/2, solar_coefficients_co2, yerr=solar_se_co2, width=width, label='Solar', capsize=5, color='#ee9a01', alpha=0.5)
    axs[1].bar(x + width/2, wind_coefficients_co2, yerr=wind_se_co2, width=width, label='Wind', capsize=5, color='#0273b2', alpha=0.5)
    axs[1].set_title('$CO_2$ emissions intensity')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels('')
    axs[1].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    
    # Subplot 3: Solar and Wind SO2 EI Coefficients
    axs[2].bar(x - width/2, solar_coefficients_so2, yerr=solar_se_so2, width=width, label='Solar', capsize=5, color='#ee9a01', alpha=0.5)
    axs[2].bar(x + width/2, wind_coefficients_so2, yerr=wind_se_so2, width=width, label='Wind', capsize=5, color='#0273b2', alpha=0.5)
    axs[2].set_title('$SO_2$ emissions intensity')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels('')
    axs[2].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    
    # Subplot 4: Solar and Wind NOx EI Coefficients
    axs[3].bar(x - width/2, solar_coefficients_nox, yerr=solar_se_nox, width=width, label='Solar', capsize=5, color='#ee9a01', alpha=0.5)
    axs[3].bar(x + width/2, wind_coefficients_nox, yerr=wind_se_nox, width=width, label='Wind', capsize=5, color='#0273b2', alpha=0.5)
    axs[3].set_title('$NO_x$ emissions intensity')
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(BAs)
    axs[3].axhline(0, color='grey', linewidth=0.8, linestyle='--')
    
    # Remove top and right spine
    for ax in axs.flat:  # Use axs.flat to iterate over all axes in a 2D array
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    handles, labels = axs[0].get_legend_handles_labels()  # Get handles and labels from the first subplot
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.55, -0.05), ncol=2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = "../results/regression/figure4_coefficients.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure 4 saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main() 