#!/usr/bin/env python3
"""
Generate Figure 3 with three subfigures:
(a) 2010 eGRID data: Share of annual generation vs emissions
(b) 2022 eGRID data: Share of annual generation vs emissions  
(c) Distributions of daily percentage changes for solar and wind generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde
import matplotlib.patches as mpatches
import os

def load_egrid_data(year):
    """Load eGRID data for specified year"""
    if year == 2010:
        file_path = 'data/egrid/eGRID2010_Data.xls'
        sheet_name = 'PLNT10'
        skiprows = 4  # Skip the first 4 rows to get to the actual data
        # Column mappings for 2010 data - use the short column names
        cols = ['ORISPL', 'ISORTO', 'PLCO2AN', 'PLNGENAN', 'PLCO2RTA', 'NAMEPCAP', 'PLFUELCT',
                'PLNOXAN', 'PLNOXRTA', 'PLSO2AN', 'PLSO2RTA']
    else:  # 2022
        file_path = 'data/egrid/egrid2022_data.xlsx'
        sheet_name = 'PLNT22'
        skiprows = 1
        # Column mappings for 2022 data
        cols = ['ORISPL', 'BACODE', 'PLCO2AN', 'PLNGENAN', 'PLCO2RTA', 'NAMEPCAP', 'PLFUELCT',
                'PLNOXAN', 'PLNOXRTA', 'PLSO2AN', 'PLSO2RTA']
    
    try:
        if year == 2010:
            df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
            df = df[cols]
            df.dropna(inplace=True)
            # Filter for fossil fuels
            fossil_fuels = ['COAL', 'GAS', 'OIL']
            df = df.loc[df.PLFUELCT.isin(fossil_fuels)]
            # Rename ISORTO to BACODE to match 2022 structure
            df = df.rename(columns={'ISORTO': 'BACODE'})
            # Map 2010 ISO/RTO values to 2022 BACODE equivalents
            iso_mapping = {
                'CAISO': 'CISO',
                'ERCOT': 'ERCO', 
                'ISONE': 'ISNE',
                'MISO': 'MISO',
                'NYISO': 'NYIS',
                'PJM': 'PJM',
                'SPP': 'SWPP'
            }
            df['BACODE'] = df['BACODE'].map(iso_mapping)
        else:
            df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows)
            df = df[cols]
            df.dropna(inplace=True)
            # Filter for fossil fuels
            fossil_fuels = ['COAL', 'GAS', 'OIL']
            df = df.loc[df.PLFUELCT.isin(fossil_fuels)]
        
        return df
    except Exception as e:
        print(f"Error loading {year} eGRID data: {e}")
        return None

def create_egrid_subfigure(df_egrid, year, ax, subfig_label):
    """Create eGRID subfigure showing generation vs emissions shares"""
    bas_of_interest = ['CISO', 'ERCO', 'ISNE', 'MISO', 'NYIS', 'PJM', 'SWPP']
    ba_names = {'CISO': 'CAISO', 'ERCO': 'ERCOT', 'ISNE': 'ISONE', 'MISO': 'MISO', 
                'NYIS': 'NYISO', 'PJM': 'PJM', 'SWPP': 'SWPP'}
    
    # Filter data for BAs of interest
    df = df_egrid[df_egrid['BACODE'].isin(bas_of_interest)].copy()
    
    # Calculate total generation and emissions for each BA
    ba_totals = df.groupby('BACODE')[['PLCO2AN', 'PLNOXAN', 'PLSO2AN', 'PLNGENAN']].sum().rename(
        columns={'PLCO2AN': 'BA_CO2', 'PLNOXAN': 'BA_NOX', 'PLSO2AN': 'BA_SO2', 'PLNGENAN': 'BA_GEN'})
    df = df.merge(ba_totals, on='BACODE')
    
    # Calculate shares for all pollutants
    df['gen_share'] = df['PLNGENAN'] / df['BA_GEN']
    df['co2_emissions_share'] = df['PLCO2AN'] / df['BA_CO2']
    df['nox_emissions_share'] = df['PLNOXAN'] / df['BA_NOX']
    df['so2_emissions_share'] = df['PLSO2AN'] / df['BA_SO2']
    
    # Convert RTAs to appropriate units
    if year == 2010:
        # 2010 data might have different units, adjust as needed
        df['PLCO2RTA'] = df['PLCO2RTA'] / 2000  # lb/MWh to ton/MWh
        df['PLSO2RTA'] = df['PLSO2RTA'] * 0.453592  # lb/MWh to kg/MWh
        df['PLNOXRTA'] = df['PLNOXRTA'] * 0.453592  # lb/MWh to kg/MWh
    else:
        df['PLCO2RTA'] = df['PLCO2RTA'] / 2000  # lb/MWh to ton/MWh
        df['PLSO2RTA'] = df['PLSO2RTA'] * 0.453592  # lb/MWh to kg/MWh
        df['PLNOXRTA'] = df['PLNOXRTA'] * 0.453592  # lb/MWh to kg/MWh
    
    # Set up subplots (3 rows for CO2, SO2, NOx)
    fig, axs = plt.subplots(3, 8, figsize=(16, 6), sharex='col', sharey='row')
    axs = axs.reshape(3, 8)
    
    # Define x-tick labels for consistent formatting
    xticks = [0, 0.025, 0.05, 0.075, 0.1, 0.125]
    xtick_labels = ['0', '', '0.05', '', '0.1', '']
    yticks = [0, 0.025, 0.05, 0.075, 0.1, 0.125]
    ytick_labels = ['0', '', '0.05', '', '0.1', '']
    
    # Set color ranges
    co2_vmin, co2_vmax = 0, 2
    so2_vmin, so2_vmax = 0, 2
    nox_vmin, nox_vmax = 0, 2
    
    for i, ba in enumerate(bas_of_interest):
        ba_data = df[df['BACODE'] == ba]
        
        # CO2 plot (top row)
        scatter_co2 = axs[0, i].scatter(
            ba_data['gen_share'], 
            ba_data['co2_emissions_share'], 
            c=ba_data['PLCO2RTA'],
            s=ba_data['NAMEPCAP'] / 20,
            cmap='Blues',
            alpha=0.7,
            vmin=co2_vmin,
            vmax=co2_vmax
        )
        
        # SO2 plot (middle row)
        scatter_so2 = axs[1, i].scatter(
            ba_data['gen_share'], 
            ba_data['so2_emissions_share'], 
            c=ba_data['PLSO2RTA'],
            s=ba_data['NAMEPCAP'] / 20,
            cmap='Oranges',
            alpha=0.7,
            vmin=so2_vmin,
            vmax=so2_vmax
        )
        
        # NOx plot (bottom row)
        scatter_nox = axs[2, i].scatter(
            ba_data['gen_share'], 
            ba_data['nox_emissions_share'], 
            c=ba_data['PLNOXRTA'],
            s=ba_data['NAMEPCAP'] / 20,
            cmap='Greys',
            alpha=0.7,
            vmin=nox_vmin,
            vmax=nox_vmax
        )
        
        # Set limits and add diagonal lines for all plots
        for row in range(3):
            axs[row, i].plot([0, 1], [0, 1], transform=axs[row, i].transAxes, color='gray', alpha=0.1)
            axs[row, i].set_xlim(0, 0.125)
            axs[row, i].set_ylim(0, 0.125)
            axs[row, i].set_xticks(xticks)
            axs[row, i].set_yticks(yticks)
            axs[row, i].set_yticklabels(ytick_labels)
            
            # Set x-tick labels only on the bottom row
            if row == 2:
                axs[row, i].set_xticklabels(xtick_labels)
            
            axs[row, i].spines['right'].set_visible(False)
            axs[row, i].spines['top'].set_visible(False)
        
        axs[0, i].set_title(ba_names[ba], fontsize=16)
    
    # Set labels
    axs[0, 0].set_ylabel('CO$_2$', fontsize=14)
    axs[1, 0].set_ylabel('SO$_2$', fontsize=14)
    axs[2, 0].set_ylabel('NO$_x$', fontsize=14)
    
    # Add master labels
    fig.text(0.03, 0.5, 'Share of annual emissions (%)', va='center', rotation='vertical', fontsize=16)
    fig.text(0.5, 0.015, 'Share of annual generation (%)', ha='center', fontsize=16)
    
    # Remove extra subplots
    fig.delaxes(axs[0, -1])
    fig.delaxes(axs[1, -1])
    fig.delaxes(axs[2, -1])
    
    # Add colorbars
    cax_co2 = inset_axes(axs[0, 6], width="5%", height="100%", loc='center left', 
                         bbox_to_anchor=(1.1, 0, 1, 1), bbox_transform=axs[0, 6].transAxes, borderpad=0)
    cbar_co2 = fig.colorbar(scatter_co2, cax=cax_co2, orientation='vertical')
    cbar_co2.set_label('CO$_2$ EI \n(ton/MWh)', fontsize=12)
    
    cax_so2 = inset_axes(axs[1, 6], width="5%", height="100%", loc='center left', 
                         bbox_to_anchor=(1.1, 0, 1, 1), bbox_transform=axs[1, 6].transAxes, borderpad=0)
    cbar_so2 = fig.colorbar(scatter_so2, cax=cax_so2, orientation='vertical')
    cbar_so2.set_label('SO$_2$ EI \n(kg/MWh)', fontsize=12)
    
    cax_nox = inset_axes(axs[2, 6], width="5%", height="100%", loc='center left', 
                         bbox_to_anchor=(1.1, 0, 1, 1), bbox_transform=axs[2, 6].transAxes, borderpad=0)
    cbar_nox = fig.colorbar(scatter_nox, cax=cax_nox, orientation='vertical')
    cbar_nox.set_label('NO$_x$ EI \n(kg/MWh)', fontsize=12)
    
    # Add subfigure label
    fig.text(0.03, 0.98, subfig_label, fontsize=16, fontweight='bold', ha='left', va='top')
    
    # Adjust layout
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    
    return fig

def create_distributions_subfigure():
    """Create subfigure c showing distributions of daily percentage changes"""
    # Define the list of BAs and file paths
    bas = ['CISO', 'ERCO', 'ISNE', 'MISO', 'PJM', 'SWPP', 'NYIS']
    titles = ['CAISO', 'ERCOT', 'ISONE', 'MISO', 'PJM', 'SWPP', 'NYISO']
    file_paths = {ba: f'data/processed/{ba}.csv' for ba in bas}
    
    # Set up the figure for 7 subplots in a single row, with extra space for the legend
    fig, axes = plt.subplots(1, 8, figsize=(18, 4), sharey=True, 
                             gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1, 1, 0.3]})
    
    # Loop through each BA to load data, calculate percentage change, and plot
    for i, (ba, title) in enumerate(zip(bas, titles)):
        # Load data
        df_gen = pd.read_csv(file_paths[ba])
        
        # Filter and rename relevant columns
        columns = ['Local time', 'NG: SUN', 'NG: WND', 'D', 'NG: COL', 'NG: NG', 'NG: OIL', 'NG: WAT', 'TI', 'solar_ext_mw', 'wind_ext_mw', 'demand_ext_mw']
        df_gen = df_gen[columns]
        df_gen = df_gen.rename(columns={
            'Local time': 'datetime',
            'D': 'demand_mw',
            'NG: SUN': 'solar_generation_mw',
            'NG: WND': 'wind_generation_mw',
            'NG: COL': 'coal_generation_mw',
            'NG: NG': 'natural_gas_generation_mw',
            'NG: OIL': 'oil_generation_mw',
            'TI': 'imports_mw',
            'NG: WAT': 'hydro_generation_mw',
        })
        
        # Convert datetime and calculate daily totals
        df_gen['Date'] = pd.to_datetime(df_gen['datetime']).dt.date
        df_gen_D = df_gen.groupby('Date').sum().reset_index()
        df_gen_D = df_gen_D[(df_gen_D['Date'] >= pd.to_datetime('2019-01-01').date()) & 
                           (df_gen_D['Date'] <= pd.to_datetime('2023-12-31').date())]
        
        # Calculate daily percentage change and filter values within range
        df_gen_D['solar_pct_change'] = df_gen_D['solar_generation_mw'].pct_change() * 100
        df_gen_D['wind_pct_change'] = df_gen_D['wind_generation_mw'].pct_change() * 100
        
        # For NYISO, only filter by wind percentage change since it doesn't have solar data
        if ba == 'NYIS':
            df_gen_D = df_gen_D[(df_gen_D['wind_pct_change'].abs() <= 200)]
        else:
            df_gen_D = df_gen_D[(df_gen_D['solar_pct_change'].abs() <= 200) & 
                               (df_gen_D['wind_pct_change'].abs() <= 200)]
        
        # Plot KDE for solar and wind percentage changes
        ax = axes[i]
        solar_data = df_gen_D['solar_pct_change'].dropna()
        wind_data = df_gen_D['wind_pct_change'].dropna()
        
        # For NYISO, only plot wind (skip solar)
        if ba == 'NYIS':
            # Only plot wind data
            if len(wind_data) > 1:  # Need at least 2 points for KDE
                wind_kde = gaussian_kde(wind_data)
                x_wind = np.linspace(wind_data.min(), wind_data.max(), 1000)
                y_wind = wind_kde(x_wind) * 100
                ax.plot(x_wind, y_wind, color='grey', label='Wind Generation % Change')
                ax.fill_between(x_wind, y_wind, color='grey', alpha=0.3)
        else:
            # Plot both solar and wind for other BAs
            if len(solar_data) > 1:  # Need at least 2 points for KDE
                solar_kde = gaussian_kde(solar_data)
                x_solar = np.linspace(solar_data.min(), solar_data.max(), 1000)
                y_solar = solar_kde(x_solar) * 100
                ax.plot(x_solar, y_solar, color='#ee9a01', label='Solar Generation % Change')
                ax.fill_between(x_solar, y_solar, color='#ee9a01', alpha=0.5)
            
            if len(wind_data) > 1:  # Need at least 2 points for KDE
                wind_kde = gaussian_kde(wind_data)
                x_wind = np.linspace(wind_data.min(), wind_data.max(), 1000)
                y_wind = wind_kde(x_wind) * 100
                ax.plot(x_wind, y_wind, color='grey', label='Wind Generation % Change')
                ax.fill_between(x_wind, y_wind, color='grey', alpha=0.3)
        
        # Customize plot appearance
        ax.set_title(title, fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Label y-axis only for the first subplot
        if i == 0:
            ax.set_ylabel('Share of Days (%)', fontsize=16)
    
    # Set the single x-axis label
    fig.text(0.5, 0.03, 'Marginal Change (%)', ha='center', fontsize=16)
    
    # Add custom legend in the eighth subplot area
    legend_ax = axes[-1]
    legend_ax.axis('off')
    
    # Define custom legend handles
    solar_patch = mpatches.Patch(color='#ee9a01', alpha=0.5, label='Solar')
    wind_patch = mpatches.Patch(color='grey', alpha=0.3, label='Wind')
    legend_ax.legend(handles=[solar_patch, wind_patch], loc='center', frameon=False, fontsize=14)
    
    # Adjust layout and display
    fig.tight_layout(rect=[0, 0.1, 1, 0.95])
    plt.subplots_adjust(wspace=0.3)
    
    # Add panel label 'c'
    fig.text(0.01, 0.98, 'c', fontsize=18, fontweight='bold', ha='left', va='top')
    
    return fig

def main():
    """Main function to generate Figure 3"""
    print("Generating Figure 3...")
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Subfigure a: 2010 eGRID data
    print("Creating subfigure a (2010 eGRID data)...")
    df_egrid_2010 = load_egrid_data(2010)
    if df_egrid_2010 is not None:
        fig_a = create_egrid_subfigure(df_egrid_2010, 2010, None, 'a')
        fig_a.savefig('results/figure3a_egrid2010.png', dpi=300, bbox_inches='tight')
        plt.close(fig_a)
        print("Subfigure a saved as figure3a_egrid2010.png")
    else:
        print("Failed to load 2010 eGRID data")
    
    # Subfigure b: 2022 eGRID data
    print("Creating subfigure b (2022 eGRID data)...")
    df_egrid_2022 = load_egrid_data(2022)
    if df_egrid_2022 is not None:
        fig_b = create_egrid_subfigure(df_egrid_2022, 2022, None, 'b')
        fig_b.savefig('results/figure3b_egrid2022.png', dpi=300, bbox_inches='tight')
        plt.close(fig_b)
        print("Subfigure b saved as figure3b_egrid2022.png")
    else:
        print("Failed to load 2022 eGRID data")
    
    # Subfigure c: Distributions
    print("Creating subfigure c (distributions)...")
    try:
        fig_c = create_distributions_subfigure()
        fig_c.savefig('results/figure3c_distributions.png', dpi=300, bbox_inches='tight')
        plt.close(fig_c)
        print("Subfigure c saved as figure3c_distributions.png")
    except Exception as e:
        print(f"Failed to create subfigure c: {e}")
    
    print("Figure 3 generation complete!")

if __name__ == "__main__":
    main() 