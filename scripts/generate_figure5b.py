import pandas as pd
import matplotlib.pyplot as plt
import glob

# Define paths for each BA's CSV file (assuming files are named as `<BA>_regression.csv` in the current directory)
ba_files = glob.glob('*_emissions_regression.csv')
fuel_categories = ['Coal', 'Gas - Combined cycle', 'Gas - Combustion turbine']
bas = [file.split('_')[0] for file in ba_files]  # Extracting BA names from file names

# Define storage for data aggregation
data = []

# Process each BA's file
for file, ba in zip(ba_files, bas):
    df = pd.read_csv(file)
    
    # Filter for specific fuel categories
    df_filtered = df[df['Fuel Category'].isin(fuel_categories)]
    
    # Extract relevant data
    for fuel in fuel_categories:
        df_fuel = df_filtered[df_filtered['Fuel Category'] == fuel]
        
        # Calculate means and errors (standard errors in parentheses) for each category
        solar_coef = df_fuel['solar_generation_mw_coef'].astype(float).mean()
        solar_error = df_fuel['solar_generation_mw'].str.extract(r'\((.*?)\)').astype(float).mean().values[0]

        wind_coef = df_fuel['wind_generation_mw_coef'].astype(float).mean()
        wind_error = df_fuel['wind_generation_mw'].str.extract(r'\((.*?)\)').astype(float).mean().values[0]

        # Store results
        data.append({'BA': ba, 'Fuel Category': fuel, 
                     'solar_coef': solar_coef, 'solar_error': solar_error,
                     'wind_coef': wind_coef, 'wind_error': wind_error})

# Convert aggregated data into DataFrame
df_plot = pd.DataFrame(data)

import matplotlib.pyplot as plt
import numpy as np

# Define the desired order of BAs
ordered_bas = ['CAISO', 'ERCOT', 'MISO', 'NYISO', 'SWPP', 'PJM', 'ISONE']

# CO₂ emissions intensity coefficients and standard errors
solar_coefficients_co2 = [0.005, 0.002, 0.001, 0.008, -0.003, -0.013, 0.000]
solar_se_co2 = [0.005, 0.006, 0.008, 0.004, 0.003, 0.006, 0.000]

wind_coefficients_co2 = [-0.004, 0.004, 0.003, 0.014, 0.022, 0.014, -0.002]
wind_se_co2 = [0.002, 0.001, 0.002, 0.001, 0.002, 0.001, 0.002]

# Set up figure
fig, axs = plt.subplots(1, len(ordered_bas), figsize=(13, 3.5), sharey=False)

# Loop over each BA to plot data
for i, ba in enumerate(ordered_bas):
    ax = axs[i]
    ba_data = df_plot[df_plot['BA'] == ba]
    
    # Adjust x positions for better spacing
    x_positions = np.arange(len(fuel_categories))  # Original positions
    x_offsets = [-0.1, 0, 0.1]  # Offsets for solar and wind points
    solar_x = x_positions + x_offsets[0]  # Shift solar points left
    wind_x = x_positions + x_offsets[1]  # Center wind points
    
    # Plot solar coefficients with error bars and caps
    ax.errorbar(solar_x, ba_data['solar_coef'], yerr=ba_data['solar_error'], 
                fmt='o', color='#ee9a01', label='Solar Coef', capsize=5) 

    # Plot wind coefficients with error bars and caps
    ax.errorbar(wind_x, ba_data['wind_coef'], yerr=ba_data['wind_error'], 
                fmt='o', color='#0273b2', label='Wind Coef', capsize=5) 
    
    ax.set_title(ba)

    # Overlay CO₂ EI solar and wind coefficients as dashed lines with shaded regions
    ax.plot(x_positions, [solar_coefficients_co2[i]] * len(fuel_categories), 
            linestyle='--', color='#ee9a01', alpha=0.4)
    ax.fill_between(x_positions, 
                    [solar_coefficients_co2[i] - solar_se_co2[i]] * len(fuel_categories),
                    [solar_coefficients_co2[i] + solar_se_co2[i]] * len(fuel_categories),
                    color='#ee9a01', alpha=0.15)
    
    ax.plot(x_positions, [wind_coefficients_co2[i]] * len(fuel_categories), 
            linestyle='--', color='#0273b2', alpha=0.4)
    ax.fill_between(x_positions, 
                    [wind_coefficients_co2[i] - wind_se_co2[i]] * len(fuel_categories),
                    [wind_coefficients_co2[i] + wind_se_co2[i]] * len(fuel_categories),
                    color='#0273b2', alpha=0.15)
    
    ax.set_xticks(range(len(fuel_categories)))
    ax.set_xticklabels(['Coal', 'NGCC', 'NGCT'], rotation=0)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)  # Reference line at 0
    ax.set_ylim(-0.21, 0.21)  

    if ba == 'ISONE':
        ax.set_ylim(-0.4, 0.9) 
        ax.set_yticks([-0.5, 0, 0.5, 1])  # Show y-ticks and labels for ISONE
    elif ba == 'CAISO':
        ax.set_yticks([ -0.2, -0.1, 0, 0.1, 0.2])  # Show y-ticks and labels for CAISO
        ax.tick_params(axis='y', which='both', labelleft=True)  # Show y-ticks and labels
    else:
        ax.set_yticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])  # Show y-ticks without labels
        ax.tick_params(axis='y', which='both', labelleft=False)  # Hide labels

    # Remove top and right spines
    ax.spines['top'].set_visible(False)  # Remove top spine
    ax.spines['right'].set_visible(False)

# Add ylabel for the entire figure
fig.text(-0.01, 0.5, 'Coefficient', va='center', rotation='vertical', fontsize=12)

# Add subfigure label "b"
fig.text(-0.01, 0.98, 'b', fontsize=14, fontweight='bold', ha='left', va='top')

# Adjust layout to reduce spacing
plt.subplots_adjust(wspace=1)  # Adjust wspace to reduce the horizontal space between plots

plt.tight_layout()

# Save the figure to results folder
plt.savefig('../../results/figure5b_emissions.png', dpi=300, bbox_inches='tight')
print("Figure 5b saved to ../../results/figure5b_emissions.png")

plt.show() 