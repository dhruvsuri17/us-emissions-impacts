#!/usr/bin/env python3
"""
Generate Figure 1: CO2 Emissions by Fuel Category and Capacity Factor
Author: Dhruv Suri

This script creates a 7-panel figure showing CO2 emissions across different fuel categories
and capacity factors for all balancing authorities.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load plant metadata for each ISO
plant_metadata_CAISO = pd.read_csv('../data/plant_metadata/plant_metadata_2 - CAISO.csv')
plant_metadata_PJM = pd.read_csv('../data/plant_metadata/plant_metadata_2 - PJM.csv')
plant_metadata_ERCOT = pd.read_csv('../data/plant_metadata/plant_metadata_2 - ERCOT.csv')
plant_metadata_ISONE = pd.read_csv('../data/plant_metadata/plant_metadata_2 - ISONE.csv')
plant_metadata_NYISO = pd.read_csv('../data/plant_metadata/plant_metadata_2 - NYISO.csv')
plant_metadata_MISO = pd.read_csv('../data/plant_metadata/plant_metadata_2 - MISO.csv')
plant_metadata_SWPP = pd.read_csv('../data/plant_metadata/plant_metadata_2 - SWPP.csv')

# Function to process data for each ISO
def process_iso_data(emissions_data, plant_metadata):
    emissions_data = emissions_data.rename(columns={
        'Gross Load (MW)': 'gross_load_mw',
        'CO2 Mass (short tons)': 'co2_tons',
        'SO2 Mass (lbs)': 'so2_lbs',
        'NOx Mass (lbs)': 'nox_lbs',
        'Heat Input (mmBtu)': 'heat_input_mmbtu'
    })

    # Calculate CO2 emission intensity (co2_EI) and remove NaNs
    emissions_data["co2_EI"] = emissions_data["co2_tons"] / emissions_data["gross_load_mw"]
    emissions_data = emissions_data.dropna().replace([np.inf, -np.inf], np.nan).dropna()

    # Aggregate statistics for each Facility ID
    emissions_data_grouped = emissions_data.groupby('Facility ID')['co2_EI'].agg(
        ['mean', lambda x: np.percentile(x, 90), lambda x: np.percentile(x, 10)]
    ).reset_index()
    emissions_data_grouped.columns = ['Facility ID', 'mean_co2_EI', 'p95_co2_EI', 'p5_co2_EI']

    # Merge with metadata
    emissions_data_grouped = pd.merge(emissions_data_grouped, plant_metadata, on='Facility ID', how='left')

    # Filter for the year 2023 and sum the emissions and load
    emissions_data['datetime'] = pd.to_datetime(emissions_data['datetime'])
    data_2023 = emissions_data[emissions_data.datetime.dt.year == 2023]
    data_2023 = data_2023.groupby('Facility ID')[['co2_tons', 'gross_load_mw']].sum().reset_index()

    # Merge back with aggregated data
    data_2023 = pd.merge(data_2023, emissions_data_grouped, on='Facility ID', how='left')

    # Calculate the CO2 tons for different EI percentiles
    data_2023['mean_co2_tons'] = data_2023['mean_co2_EI'] * data_2023['gross_load_mw']
    data_2023['p95_co2_tons'] = data_2023['p95_co2_EI'] * data_2023['gross_load_mw']
    data_2023['p5_co2_tons'] = data_2023['p5_co2_EI'] * data_2023['gross_load_mw']

    # Categorize the fuels
    data_2023['Category'] = data_2023['Primary Fuel'].apply(lambda x: x if x in ['NG', 'SUB', 'BIT'] else 'Other')

    # Group by category and capacity factor, and include gross_load_mw
    return data_2023.groupby(['Category', 'Capacity Factor']).agg({
        'co2_tons': 'sum', 
        'p95_co2_tons': 'sum', 
        'p5_co2_tons': 'sum',
        'gross_load_mw': 'sum'  # Include total generation for analysis
    }).reset_index()

# Load aggregated data for all ISOs
ISO_data = {
    'CAISO': process_iso_data(pd.read_csv('../data/aggregated_data/aggregated_data_filtered_CAISO.csv'), plant_metadata_CAISO),
    'ERCOT': process_iso_data(pd.read_csv('../data/aggregated_data/aggregated_data_filtered_ERCOT.csv'), plant_metadata_ERCOT),
    'ISONE': process_iso_data(pd.read_csv('../data/aggregated_data/aggregated_data_filtered_ISONE.csv'), plant_metadata_ISONE),
    'MISO': process_iso_data(pd.read_csv('../data/aggregated_data/aggregated_data_filtered_MISO.csv'), plant_metadata_MISO),
    'NYISO': process_iso_data(pd.read_csv('../data/aggregated_data/aggregated_data_filtered_NYISO.csv'), plant_metadata_NYISO),
    'PJM': process_iso_data(pd.read_csv('../data/aggregated_data/aggregated_data_filtered_PJM.csv'), plant_metadata_PJM),
    'SWPP': process_iso_data(pd.read_csv('../data/aggregated_data/aggregated_data_filtered_SWPP.csv'), plant_metadata_SWPP),
}

# Define the color scheme for each category and capacity factor
categories = {
    'BIT_low': '#f4b942',   # BIT 0-0.3
    'BIT_mid': '#ee9a01',   # BIT 0.3-0.6
    'BIT_high': '#b86b00',  # BIT 0.6-1.0
    'SUB_low': '#d9d9d9',   # SUB 0-0.3
    'SUB_mid': '#999999',   # SUB 0.3-0.6
    'SUB_high': '#4d4d4d',  # SUB 0.6-1.0
    'NG_low': '#b7e4cc',    # NG 0-0.3
    'NG_mid': '#019e73',    # NG 0.3-0.6
    'NG_high': '#004c37',   # NG 0.6-1.0
    'Other_low': '#a1c4e9', # Other 0-0.3
    'Other_mid': '#0273b2', # Other 0.3-0.6
    'Other_high': '#014380' # Other 0.6-1.0
}

fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(11, 4.5), sharey=False)

for i, (iso_name, grouped_data) in enumerate(ISO_data.items()):
    ax = axes[i]
    bottom = [0, 0, 0]
    
    for _, row in grouped_data.iterrows():
        category_key = f"{row['Category']}_low" if row['Capacity Factor'] < 0.3 else \
                       f"{row['Category']}_mid" if row['Capacity Factor'] < 0.6 else \
                       f"{row['Category']}_high"
        
        values = [row['p5_co2_tons'], row['co2_tons'], row['p95_co2_tons']]
        ax.bar(['p5_co2_tons', 'co2_tons', 'p95_co2_tons'], values, bottom=bottom, color=categories[category_key], label=category_key if i == 0 else "")
        bottom = [sum(x) for x in zip(bottom, values)]
    
    ax.set_title(iso_name)
    ax.set_xticks([])

    bar_labels = ['P10', 'Obs', 'P90']

    ax.set_title(iso_name)
    ax.set_xticks([0, 1, 2])  # Positions for 'P10', 'Obs', 'P90'
    ax.set_xticklabels(bar_labels)

# Common y-axis label
axes[0].set_ylabel('CO$_2$ Emissions (Mtons)', va='center', rotation='vertical', labelpad=20, fontsize=13)

# Adjust y-axis to show values in millions (Mtons)
for ax in axes:
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x/1e6))))

# Add a common legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False)

y_limits = {
    'Total US': 2000 * 1e6,  # Set a reasonable limit for total US
    'CAISO': 100 * 1e6,
    'ERCOT': 250 * 1e6,
    'ISONE': 50 * 1e6,
    'MISO': 1000 * 1e6,
    'NYISO': 50 * 1e6,
    'PJM': 500 * 1e6,
    'SWPP': 200 * 1e6
}

axes[0].set_ylim(0, 100000000)
axes[1].set_ylim(0, 250000000)
axes[2].set_ylim(0, 50000000)
axes[3].set_ylim(0, 800000000)
axes[4].set_ylim(0, 50000000)
axes[5].set_ylim(0, 400000000)
axes[6].set_ylim(0, 160000000)

legend_entries = [
    ('BIT (0-0.3)', categories['BIT_low']),
    ('BIT (0.3-0.6)', categories['BIT_mid']),
    ('BIT (0.6-1.0)', categories['BIT_high']),
    ('SUB (0-0.3)', categories['SUB_low']),
    ('SUB (0.3-0.6)', categories['SUB_mid']),
    ('SUB (0.6-1.0)', categories['SUB_high']),
    ('NG (0-0.3)', categories['NG_low']),
    ('NG (0.3-0.6)', categories['NG_mid']),
    ('NG (0.6-1.0)', categories['NG_high']),
    ('Other (0-0.3)', categories['Other_low']),
    ('Other (0.3-0.6)', categories['Other_mid']),
    ('Other (0.6-1.0)', categories['Other_high'])
]

# Create custom legend
handles = [plt.Line2D([0], [0], marker='s', color=color, linestyle='', markersize=10) for _, color in legend_entries]
labels = [label for label, _ in legend_entries]

fig.legend(handles, labels, loc='lower center', title='Fuel Category and Capacity Factor',
            ncol=4, frameon=True, handletextpad=1, columnspacing=2, bbox_to_anchor=(0.5, -0.1), fontsize=12,
            title_fontsize=12)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(bottom=0.25)

# Save the figure
output_path = "../results/regression/figure1_emissions_by_fuel.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure 1 saved to: {output_path}")

plt.show() 