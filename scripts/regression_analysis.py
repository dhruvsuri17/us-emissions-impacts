#!/usr/bin/env python3
"""
Regression Analysis Script for US Emissions Impacts Study

This script performs panel regression analysis on balancing authority data to analyze
the relationship between renewable energy generation and emissions across different ISOs.

Author: Dhruv Suri
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import re
from pathlib import Path

# Define balancing authority mappings
BA_MAPPING = {
    'CAISO': 'CISO',
    'ERCOT': 'ERCO', 
    'ISONE': 'ISNE',
    'MISO': 'MISO',
    'NYISO': 'NYIS',
    'PJM': 'PJM',
    'SWPP': 'SWPP'
}

# Define variables for analysis
VARIABLES = ['residual_demand_mw', 'solar_generation_mw', 'wind_generation_mw', 
             'solar_ramp', 'wind_ramp', 'R-squared', 'Num of Obs']

DEPENDENT_VARS = ['gross_load_mw', 'co2_mass_shorttons', 'co2_emissions_intensity', 
                  'so2_mass_kg', 'so2_emissions_intensity', 'nox_mass_kg', 'nox_emissions_intensity']

# Variable labels for LaTeX tables
VARIABLE_LABELS = {
    'residual_demand_mw': 'Thermal Generation',
    'solar_generation_mw': 'Solar',
    'wind_generation_mw': 'Wind'
}

def significance_stars(p_value):
    """Add significance stars based on p-value."""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''

def process_ba_data(ba, ba_2):
    """Process data for a single balancing authority."""
    print(f"Processing {ba}...")
    
    # Define file paths
    df_path = f'../data/CEMS_processed/{ba}.csv'
    df_gen_path = f'../data/processed/{ba_2}.csv'
    
    # Check if files exist
    if not os.path.exists(df_path):
        print(f"Warning: {df_path} not found, skipping {ba}")
        return None
    if not os.path.exists(df_gen_path):
        print(f"Warning: {df_gen_path} not found, skipping {ba}")
        return None
    
    try:
        # Load data
        df = pd.read_csv(df_path)
        df_gen = pd.read_csv(df_gen_path)
        
        # Preprocess CEMS data
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.groupby(['Date', 'Facility ID']).sum().reset_index()
        
        # Select and rename generation columns
        columns = ['Local time', 'NG: SUN', 'NG: WND', 'D', 'NG: COL', 'NG: NG', 
                  'NG: OIL', 'NG: WAT', 'TI', 'solar_ext_mw', 'wind_ext_mw', 'demand_ext_mw']
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
        
        # Aggregate generation data by date
        df_gen['Date'] = pd.to_datetime(df_gen['datetime']).dt.date
        df_gen_D = df_gen.groupby('Date').sum().reset_index()
        df_gen_D = df_gen_D.drop(columns=['datetime'])
        
        # Filter between 2019 and 2023
        df_gen_D = df_gen_D[(df_gen_D['Date'] >= pd.to_datetime('2019-01-01').date()) & 
                           (df_gen_D['Date'] <= pd.to_datetime('2023-12-31').date())]
        
        # Merge datasets
        df_merged = pd.merge(df, df_gen_D, left_on='Date', right_on='Date', how='left')
        
        # Rename columns
        df_merged = df_merged.rename(columns={
            'Date': 'datetime',
            'Facility ID': 'id',
            'Operating Time': 'optime',
            'Gross Load (MW)': 'gross_load_mw',
            'CO2 Mass (short tons)': 'co2_mass_shorttons',
        })
        
        # Calculate additional variables
        df_merged['imports_abs_mw'] = df_merged['imports_mw'].abs()
        df_merged['netimports_mw'] = np.where(df_merged['imports_mw'] < 0, df_merged['imports_mw'] * -1, 0)
        df_merged['netexports_mw'] = np.where(df_merged['imports_mw'] > 0, df_merged['imports_mw'], 0)
        df_merged.dropna(inplace=True)
        
        # Calculate thermal generation
        df_thermal = df_merged[['datetime', 'gross_load_mw']].groupby('datetime').sum().reset_index()
        df_thermal = df_thermal.rename(columns={'gross_load_mw': 'thermal_generation_mw'})
        df_merged = pd.merge(df_merged, df_thermal, left_on='datetime', right_on='datetime', how='left')
        
        # Add time variables
        df_merged['month'] = pd.to_datetime(df_merged['datetime']).dt.month
        df_merged['year'] = pd.to_datetime(df_merged['datetime']).dt.year
        
        # Calculate emissions intensity
        df_merged['co2_emissions_intensity'] = df_merged['co2_mass_shorttons'] / df_merged['gross_load_mw']
        df_merged['co2_emissions_intensity'] = df_merged['co2_emissions_intensity'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Convert SO2 and NOx to kg
        df_merged['so2_mass_kg'] = df_merged['SO2 Mass (lbs)'] * 0.453592
        df_merged['nox_mass_kg'] = df_merged['NOx Mass (lbs)'] * 0.453592
        
        # Calculate SO2 and NOx emissions intensity
        df_merged['so2_emissions_intensity'] = df_merged['so2_mass_kg'] / df_merged['gross_load_mw']
        df_merged['nox_emissions_intensity'] = df_merged['nox_mass_kg'] / df_merged['gross_load_mw']
        df_merged['so2_emissions_intensity'] = df_merged['so2_emissions_intensity'].fillna(0).replace([np.inf, -np.inf], 0)
        df_merged['nox_emissions_intensity'] = df_merged['nox_emissions_intensity'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Calculate wind ramp
        df_wramp_mw = df_gen[['datetime', 'wind_generation_mw']].dropna().reset_index(drop=True)
        df_wramp_mw['datetime'] = pd.to_datetime(df_wramp_mw['datetime'])
        df_wramp_mw['datetime'] = df_wramp_mw['datetime'].dt.date
        df_wramp_mw['datetime'] = pd.to_datetime(df_wramp_mw['datetime'])
        df_wramp_mw.set_index('datetime', inplace=True)
        df_wramp_mw['wind_ramp'] = df_wramp_mw['wind_generation_mw'].diff().abs()
        df_wramp_mw['hour'] = df_wramp_mw.index.hour
        df_wramp_mw.loc[df_wramp_mw['hour'] == 1, 'wind_ramp'] = None
        daily_wind_ramp = df_wramp_mw.groupby('datetime')['wind_ramp'].sum()
        
        # Calculate solar ramp
        df_sramp_mw = df_gen[['datetime', 'solar_generation_mw']].dropna().reset_index(drop=True)
        df_sramp_mw['datetime'] = pd.to_datetime(df_sramp_mw['datetime'])
        df_sramp_mw['datetime'] = df_sramp_mw['datetime'].dt.date
        df_sramp_mw['datetime'] = pd.to_datetime(df_sramp_mw['datetime'])
        df_sramp_mw.set_index('datetime', inplace=True)
        df_sramp_mw['solar_ramp'] = df_sramp_mw['solar_generation_mw'].diff().abs()
        df_sramp_mw['hour'] = df_sramp_mw.index.hour
        df_sramp_mw.loc[df_sramp_mw['hour'] == 1, 'solar_ramp'] = None
        daily_solar_ramp = df_sramp_mw.groupby('datetime')['solar_ramp'].sum()
        
        # Merge ramp data
        df_merged['datetime'] = pd.to_datetime(df_merged['datetime'])
        df_merged = pd.merge(df_merged, daily_wind_ramp, left_on='datetime', right_on='datetime', how='left')
        df_merged = pd.merge(df_merged, daily_solar_ramp, left_on='datetime', right_on='datetime', how='left')
        
        # Calculate shares
        df_merged['solar_share'] = df_merged['solar_generation_mw'] / df_merged['demand_mw']
        df_merged['wind_share'] = df_merged['wind_generation_mw'] / df_merged['demand_mw']
        
        # Calculate residual demand
        df_merged['residual_demand_mw'] = df_merged['demand_mw'] - df_merged['hydro_generation_mw'] + df_merged['imports_mw']
        
        return df_merged
        
    except Exception as e:
        print(f"Error processing {ba}: {str(e)}")
        return None

def run_regression(df_merged, ba):
    """Run regression analysis for a balancing authority.
    
    Special handling for PJM, SWPP, and NYISO:
    - Filters by gross_load_mw > 0.1 MW to avoid zero load issues
    - Applies log(x + 0.001) transformation to independent variables to handle zeros
    
    Special handling for NYISO:
    - Adds 1 to all independent variables before log transformation to handle zeros
    """
    if df_merged is None:
        return None
    
    # Define base independent variables
    base_ind_vars = ['solar_generation_mw', 'wind_generation_mw', 'residual_demand_mw', 'wind_ramp', 'solar_ramp']
    
    # Initialize results dictionary
    all_results = {}
    
    for dependent_var in DEPENDENT_VARS:
        # Set ind_vars to base_ind_vars for this iteration
        ind_vars = base_ind_vars
        
        summary_dict = {var: [] for var in base_ind_vars}  # Use base_ind_vars for consistent structure
        summary_dict['R-squared'] = []
        summary_dict['Num of Obs'] = []
        
        # Create regression formula
        formula = f'{dependent_var} ~ {" + ".join(ind_vars)} + C(id) + C(month) + C(year)'
        
        # Special handling for PJM, SWPP, and NYISO due to zero load issues
        if ba in ['PJM', 'SWPP', 'NYISO']:
            # Filter by gross load > 0.1 MW to avoid zero load issues
            df_filtered = df_merged[df_merged['gross_load_mw'] > 0.1].copy()
            print(f"{ba}: Filtered to {len(df_filtered):,} rows with load > 0.1 MW")
        else:
            # Standard filtering for other BAs
            df_filtered = df_merged[df_merged[dependent_var] > 0].copy()
        
        if len(df_filtered) == 0:
            print(f"Warning: No valid data for {dependent_var} in {ba}")
            continue
        
        # Additional filtering for dependent variable > 0 (for all BAs)
        df_filtered = df_filtered[df_filtered[dependent_var] > 0].copy()
        
        if len(df_filtered) == 0:
            print(f"Warning: No valid data after dependent variable filtering for {dependent_var} in {ba}")
            continue
        
        # Special handling for NYISO: add 1 to all independent variables before log transformation
        if ba == 'NYISO':
            print(f"NYISO: Adding 1 to all independent variables before log transformation")
            df_filtered[ind_vars] = df_filtered[ind_vars] + 1
        
        # Log transformation
        df_filtered[dependent_var] = np.log(df_filtered[dependent_var])
        
        if ba == 'NYISO':
            # For NYISO: log transformation after adding 1
            df_filtered[ind_vars] = np.log(df_filtered[ind_vars])
            print(f"NYISO: Applied log transformation after adding 1")
        elif ba in ['PJM', 'SWPP']:
            # For PJM and SWPP: standard log transformation (no adding 1)
            df_filtered[ind_vars] = np.log(df_filtered[ind_vars])
            print(f"{ba}: Applied standard log transformation")
        else:
            # Standard log transformation for other BAs
            df_filtered[ind_vars] = np.log(df_filtered[ind_vars])
        
        # Handle infinite values
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df_filtered) == 0:
            print(f"Warning: No valid data after transformation for {dependent_var} in {ba}")
            continue
        
        try:
            # Fit model
            model = smf.ols(formula, data=df_filtered).fit()
            
            # Store results
            for var in base_ind_vars:  # Use base_ind_vars to maintain consistent structure
                if var in model.params.index:
                    coeff_value = model.params[var]
                    std_err_value = model.bse[var]
                    p_value = model.pvalues[var]
                    stars = significance_stars(p_value)
                    result = f"{coeff_value:.4f} ({std_err_value:.4f}){stars}"
                    summary_dict[var].append(result)
                else:
                    summary_dict[var].append("")  # Empty string for excluded variables
            
            summary_dict['R-squared'].append(f"{model.rsquared:.4f}")
            summary_dict['Num of Obs'].append(f"{int(model.nobs)}")
            
            # Log which variables were used
            if ba == 'NYISO':
                print(f"NYISO {dependent_var}: Used {len(ind_vars)} variables: {ind_vars}")
            
        except Exception as e:
            print(f"Error in regression for {dependent_var} in {ba}: {str(e)}")
            for var in base_ind_vars:
                summary_dict[var].append("")
            summary_dict['R-squared'].append("")
            summary_dict['Num of Obs'].append("")
        
        # Create DataFrame for this dependent variable
        summary_df = pd.DataFrame.from_dict(summary_dict, orient='index')
        summary_df.columns = [dependent_var]
        all_results[dependent_var] = summary_df
    
    # Combine results
    if all_results:
        final_summary_df = pd.concat(all_results.values(), axis=1)
        
        # Log summary for NYISO
        if ba == 'NYISO':
            print(f"\nNYISO Summary:")
            print(f"  - Final dataset size: {len(df_filtered) if 'df_filtered' in locals() else 'N/A'}")
            print(f"  - Variables used: {ind_vars if 'ind_vars' in locals() else 'N/A'}")
            print(f"  - Solar variables included: {'solar_generation_mw' in ind_vars and 'solar_ramp' in ind_vars}")
        
        return final_summary_df
    else:
        return None

def format_coefficient(value, var):
    """Format coefficient values for LaTeX tables."""
    if var == 'R-squared':
        try:
            return f"{float(value):.2f}"
        except:
            return value
    
    if var == 'Num of Obs':
        try:
            return f"{int(value):,}"
        except:
            return value
    
    if isinstance(value, str):
        match = re.match(r'(-?\d+\.\d+)\s*\((\d+\.\d+)\)(\**)', value)
        if match:
            coef, se, stars = match.groups()
            coef_val = float(coef)
            se_val = float(se)
            if coef_val > 1000 or se_val > 1000:
                coef_val = se_val = 0
            return f"${coef_val:.3f}^{{\\footnotesize {stars}}}$&({se_val:.3f})"
    elif isinstance(value, (int, float)):
        if value > 1000:
            return "0"
        return f"{value:.4f}"
    return value

def generate_main_table():
    """Generate main regression table LaTeX code."""
    dependent_vars = ['gross_load_mw', 'co2_mass_shorttons', 'co2_emissions_intensity']
    
    latex_code = r"""\begin{table}[htbp]
    \small
    \centering
    \caption{\textbf{Coefficient of the panel regression formulation with generation, $\text{CO}_2$ emissions, and $\text{CO}_2$ emissions intensity as the dependent variable}. Significance levels: {\footnotesize ***} $p < 0.01$, {\footnotesize **} $p < 0.05$, {\footnotesize *} $p < 0.1$.}
    \label{tab:panel-regression}
    """
    
    for idx, dep_var in enumerate(dependent_vars):
        latex_code += r"""
    \begin{subtable}[t]{\textwidth}
        \centering
        \caption{""" + (f"Generation" if idx == 0 else f"$\\text{{CO}}_2$ {'Emissions' if idx == 1 else 'Emissions Intensity'}") + r"""}
        \small
        \begin{tabular}{l>{\raggedleft\arraybackslash}p{1cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}}
            \toprule
            & """ + " & ".join([f"\\textbf{{{iso}}}" for iso in BA_MAPPING.keys()]) + r""" \\
            \midrule"""
        
        for var in VARIABLES:
            var_label = VARIABLE_LABELS.get(var, var.replace('_', ' ').title())
            latex_code += f"\n\\textbf{{{var_label}}}"
            
            coefficients = []
            standard_errors = []
            
            for ba in BA_MAPPING.keys():
                file_path = f"../results/regression/{ba}_regression_results.csv"
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=0)
                    value = df.loc[var, dep_var] if var in df.index and dep_var in df.columns else "-"
                    formatted = format_coefficient(value, var)
                    if '&' in formatted:
                        coef, se = formatted.split('&')
                        coefficients.append(coef)
                        standard_errors.append(se)
                    else:
                        coefficients.append(formatted)
                        standard_errors.append('')
                else:
                    coefficients.append('-')
                    standard_errors.append('')
            
            latex_code += " & " + " & ".join(coefficients) + r" \\"
            
            if any(standard_errors):
                latex_code += "\n            & " + " & ".join(standard_errors) + r" \\"
            
            if var == 'wind_ramp':
                latex_code += r"\n            \midrule"
        
        latex_code += r"""
            \bottomrule
        \end{tabular}
    \end{subtable}
    
    \vspace{0.3cm}
    """
    
    latex_code += r"\end{table}"
    return latex_code

def generate_emissions_table():
    """Generate emissions regression table LaTeX code."""
    dependent_vars = ['co2_mass_shorttons', 'so2_mass_kg', 'nox_mass_kg']
    
    latex_code = r"""\begin{table}[htbp]
    \small
    \centering
    \caption{\textbf{Coefficient of the panel regression formulation with $\text{CO}_2$, $\text{SO}_2$, and $\text{NO}_x$ emissions as the dependent variable}. Significance levels: {\footnotesize ***} $p < 0.001$, {\footnotesize **} $p < 0.01$, {\footnotesize *} $p < 0.05$.}
    \label{tab:emissions-regression}
    """
    
    for idx, dep_var in enumerate(dependent_vars):
        subtable_caption = ""
        if dep_var == 'co2_mass_shorttons':
            subtable_caption = "$\\text{CO}_2$ Emissions"
        elif dep_var == 'so2_mass_kg':
            subtable_caption = "$\\text{SO}_2$ Emissions"
        elif dep_var == 'nox_mass_kg':
            subtable_caption = "$\\text{NO}_x$ Emissions"
        
        latex_code += r"""
    \begin{subtable}[t]{\textwidth}
        \centering
        \caption{""" + subtable_caption + r"""}
        \small
        \begin{tabular}{l>{\raggedleft\arraybackslash}p{1cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}}
            \toprule
            & """ + " & ".join([f"\\textbf{{{iso}}}" for iso in BA_MAPPING.keys()]) + r""" \\
            \midrule"""
        
        for var in VARIABLES:
            var_label = VARIABLE_LABELS.get(var, var.replace('_', ' ').title())
            latex_code += f"\n\\textbf{{{var_label}}}"
            
            coefficients = []
            standard_errors = []
            
            for ba in BA_MAPPING.keys():
                file_path = f"../results/regression/{ba}_regression_results.csv"
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=0)
                    value = df.loc[var, dep_var] if var in df.index and dep_var in df.columns else "-"
                    formatted = format_coefficient(value, var)
                    if '&' in formatted:
                        coef, se = formatted.split('&')
                        coefficients.append(coef)
                        standard_errors.append(se)
                    else:
                        coefficients.append(formatted)
                        standard_errors.append('')
                else:
                    coefficients.append('-')
                    standard_errors.append('')
            
            latex_code += " & " + " & ".join(coefficients) + r" \\"
            
            if any(standard_errors):
                latex_code += "\n            & " + " & ".join(standard_errors) + r" \\"
            
            if var == 'wind_ramp':
                latex_code += r"\n            \midrule"
        
        latex_code += r"""
            \bottomrule
        \end{tabular}
    \end{subtable}
    
    \vspace{0.3cm}
    """
    
    latex_code += r"\end{table}"
    return latex_code

def generate_intensity_table():
    """Generate emissions intensity regression table LaTeX code."""
    dependent_vars = ['co2_emissions_intensity', 'so2_emissions_intensity', 'nox_emissions_intensity']
    
    latex_code = r"""\begin{table}[htbp]
    \small
    \centering
    \caption{\textbf{Coefficient of the panel regression formulation with $\text{CO}_2$, $\text{SO}_2$, and $\text{NO}_x$ emissions intensity as the dependent variable}. Significance levels: {\footnotesize ***} $p < 0.001$, {\footnotesize **} $p < 0.01$, {\footnotesize *} $p < 0.05$.}
    \label{tab:intensity-regression}
    """
    
    for idx, dep_var in enumerate(dependent_vars):
        subtable_caption = ""
        if dep_var == 'co2_emissions_intensity':
            subtable_caption = "$\\text{CO}_2$ Emissions Intensity"
        elif dep_var == 'so2_emissions_intensity':
            subtable_caption = "$\\text{SO}_2$ Emissions Intensity"
        elif dep_var == 'nox_emissions_intensity':
            subtable_caption = "$\\text{NO}_x$ Emissions Intensity"
        
        latex_code += r"""
    \begin{subtable}[t]{\textwidth}
        \centering
        \caption{""" + subtable_caption + r"""}
        \small
        \begin{tabular}{l>{\raggedleft\arraybackslash}p{1cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}>{\raggedleft\arraybackslash}p{1.5cm}}
            \toprule
            & """ + " & ".join([f"\\textbf{{{iso}}}" for iso in BA_MAPPING.keys()]) + r""" \\
            \midrule"""
        
        for var in VARIABLES:
            var_label = VARIABLE_LABELS.get(var, var.replace('_', ' ').title())
            latex_code += f"\n\\textbf{{{var_label}}}"
            
            coefficients = []
            standard_errors = []
            
            for ba in BA_MAPPING.keys():
                file_path = f"../results/regression/{ba}_regression_results.csv"
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path, index_col=0)
                    value = df.loc[var, dep_var] if var in df.index and dep_var in df.columns else "-"
                    formatted = format_coefficient(value, var)
                    if '&' in formatted:
                        coef, se = formatted.split('&')
                        coefficients.append(coef)
                        standard_errors.append(se)
                    else:
                        coefficients.append(formatted)
                        standard_errors.append('')
                else:
                    coefficients.append('-')
                    standard_errors.append('')
            
            latex_code += " & " + " & ".join(coefficients) + r" \\"
            
            if any(standard_errors):
                latex_code += "\n            & " + " & ".join(standard_errors) + r" \\"
            
            if var == 'wind_ramp':
                latex_code += r"\n            \midrule"
        
        latex_code += r"""
            \bottomrule
        \end{tabular}
    \end{subtable}
    
    \vspace{0.3cm}
    """
    
    latex_code += r"\end{table}"
    return latex_code

def extract_coefficient(value):
    """Extract coefficient value from formatted string."""
    if isinstance(value, str):
        match = re.match(r'(-?\d+\.\d+)', value)
        if match:
            return float(match.group(1))
    return np.nan

def calculate_displacement(alpha_prime, alpha):
    """Calculate displacement fraction."""
    if alpha_prime - alpha != 0:
        return alpha_prime / (alpha_prime - alpha)
    else:
        return np.nan

def generate_displacement_table():
    """Generate displacement effectiveness table LaTeX code."""
    # Initialize displacement dictionaries
    solar_displacement = {ba: {'CO2': np.nan, 'SO2': np.nan, 'NOX': np.nan} for ba in BA_MAPPING.keys()}
    wind_displacement = {ba: {'CO2': np.nan, 'SO2': np.nan, 'NOX': np.nan} for ba in BA_MAPPING.keys()}
    
    # Calculate displacement for each BA
    for ba in BA_MAPPING.keys():
        file_path = f"../results/regression/{ba}_regression_results.csv"
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            
            for pollutant in ['co2', 'so2', 'nox']:
                emissions_var = f'{pollutant}_mass_kg' if pollutant != 'co2' else f'{pollutant}_mass_shorttons'
                intensity_var = f'{pollutant}_emissions_intensity'
                
                if emissions_var in df.columns and intensity_var in df.columns:
                    solar_coef_emissions = extract_coefficient(df.loc['solar_generation_mw', emissions_var])
                    solar_coef_intensity = extract_coefficient(df.loc['solar_generation_mw', intensity_var])
                    
                    wind_coef_emissions = extract_coefficient(df.loc['wind_generation_mw', emissions_var])
                    wind_coef_intensity = extract_coefficient(df.loc['wind_generation_mw', intensity_var])
                    
                    solar_displacement[ba][pollutant.upper()] = calculate_displacement(solar_coef_emissions, solar_coef_intensity)
                    wind_displacement[ba][pollutant.upper()] = calculate_displacement(wind_coef_emissions, wind_coef_intensity)
    
    # Convert to DataFrames
    solar_displacement_df = pd.DataFrame(solar_displacement).T
    wind_displacement_df = pd.DataFrame(wind_displacement).T
    
    # Format values
    solar_displacement_df = solar_displacement_df.applymap(lambda x: f"{x:.6f}" if pd.notna(x) and x >= 0 else '0')
    wind_displacement_df = wind_displacement_df.applymap(lambda x: f"{x:.6f}" if pd.notna(x) and x >= 0 else '0')
    
    # Generate LaTeX code
    latex_code = r"""\begin{table}[htbp]
    \small
    \centering
    \caption{\textbf{Displacement Effectiveness for Solar and Wind across CO\textsubscript{2}, SO\textsubscript{2}, and NO\textsubscript{x} Emissions for ISOs}.}
    \label{tab:displacement-effectiveness}
    """
    
    # Solar Displacement Subtable
    latex_code += r"""
    \begin{subtable}[t]{\textwidth}
        \centering
        \caption{Solar Displacement Effectiveness}
        \small
        \begin{tabular}{l""" + "r" * len(solar_displacement_df.index) + r"""}
            \toprule
            & """ + " & ".join([f"\\textbf{{{ba}}}" for ba in solar_displacement_df.index]) + r""" \\
            \midrule"""
    
    for gas in solar_displacement_df.columns:
        gas_label = f"CO\\textsubscript{{2}}" if gas == 'CO2' else f"SO\\textsubscript{{2}}" if gas == 'SO2' else f"NO\\textsubscript{{x}}"
        latex_code += f"\n\\textbf{{{gas_label}}} & " + " & ".join([f"{float(value):.2f}" if isinstance(value, (int, float)) else value for value in solar_displacement_df[gas]]) + r" \\"
    
    latex_code += r"""
            \bottomrule
        \end{tabular}
    \end{subtable}
    
    \vspace{0.5cm}
    """
    
    # Wind Displacement Subtable
    latex_code += r"""
    \begin{subtable}[t]{\textwidth}
        \centering
        \caption{Wind Displacement Effectiveness}
        \small
        \begin{tabular}{l""" + "r" * len(wind_displacement_df.index) + r"""}
            \toprule
            & """ + " & ".join([f"\\textbf{{{ba}}}" for ba in wind_displacement_df.index]) + r""" \\
            \midrule"""
    
    for gas in wind_displacement_df.columns:
        gas_label = f"CO\\textsubscript{{2}}" if gas == 'CO2' else f"SO\\textsubscript{{2}}" if gas == 'SO2' else f"NO\\textsubscript{{x}}"
        latex_code += f"\n\\textbf{{{gas_label}}} & " + " & ".join([f"{float(value):.2f}" if isinstance(value, (int, float)) else value for value in wind_displacement_df[gas]]) + r" \\"
    
    latex_code += r"""
            \bottomrule
        \end{tabular}
    \end{subtable}
    
    \vspace{0.3cm}
    \end{table}"""
    
    return latex_code

def main():
    """Main execution function."""
    print("Starting regression analysis...")
    
    # Create results directory if it doesn't exist
    results_dir = Path("../results/regression")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each balancing authority
    for ba, ba_2 in BA_MAPPING.items():
        print(f"\nProcessing {ba} ({ba_2})...")
        
        # Process data
        df_merged = process_ba_data(ba, ba_2)
        
        if df_merged is not None:
            # Run regression
            results = run_regression(df_merged, ba)
            
            if results is not None:
                # Save results
                output_file = results_dir / f"{ba}_regression_results.csv"
                results.to_csv(output_file)
                print(f"Results saved to {output_file}")
            else:
                print(f"No results generated for {ba}")
        else:
            print(f"Failed to process data for {ba}")
    
    print("\nGenerating LaTeX tables...")
    
    # Generate and save LaTeX tables
    tables = {
        'main_regression': generate_main_table(),
        'emissions_regression': generate_emissions_table(),
        'intensity_regression': generate_intensity_table(),
        'displacement_effectiveness': generate_displacement_table()
    }
    
    for table_name, latex_code in tables.items():
        output_file = results_dir / f"{table_name}.tex"
        with open(output_file, 'w') as f:
            f.write(latex_code)
        print(f"Table saved to {output_file}")
    
    print("\nRegression analysis complete!")

if __name__ == "__main__":
    main() 