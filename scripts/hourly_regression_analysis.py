#!/usr/bin/env python3
"""
Hourly Regression Analysis Script
Performs regression analysis on hourly CEMS and generation data without daily aggregation.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import glob

def load_hourly_data(ba_name):
    """
    Load hourly CEMS and generation data for a specific BA.
    
    Parameters:
        ba_name (str): Balancing authority name (e.g., 'CAISO', 'PJM')
    
    Returns:
        tuple: (cems_df, gen_df) - CEMS and generation dataframes
    """
    # Load CEMS data
    cems_file = f'../data/CEMS_processed/{ba_name}.csv'
    cems_df = pd.read_csv(cems_file)
    
    # Load generation data
    gen_file = f'../data/processed/{ba_name}.csv'
    gen_df = pd.read_csv(gen_file)
    
    return cems_df, gen_df

def preprocess_hourly_data(cems_df, gen_df):
    """
    Preprocess hourly data for regression analysis.
    
    Parameters:
        cems_df (pd.DataFrame): CEMS hourly data
        gen_df (pd.DataFrame): Generation hourly data
    
    Returns:
        pd.DataFrame: Merged and preprocessed hourly data
    """
    # Process CEMS data
    cems_df['datetime'] = pd.to_datetime(cems_df['Date'])
    cems_df = cems_df.groupby(['datetime', 'Facility ID']).sum().reset_index()
    
    # Process generation data
    columns = ['Local time', 'NG: SUN', 'NG: WND', 'D', 'NG: COL', 'NG: NG', 'NG: OIL', 'NG: WAT', 'TI', 'solar_ext_mw', 'wind_ext_mw', 'demand_ext_mw']
    gen_df = gen_df[columns]
    
    gen_df = gen_df.rename(columns={
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
    
    gen_df['datetime'] = pd.to_datetime(gen_df['datetime'])
    
    # Filter between 2019 and 2023
    gen_df = gen_df[(gen_df['datetime'] >= '2019-01-01') & (gen_df['datetime'] <= '2023-12-31')]
    
    # Merge CEMS and generation data
    merged_df = pd.merge(cems_df, gen_df, left_on='datetime', right_on='datetime', how='left')
    
    # Rename columns for consistency
    merged_df = merged_df.rename(columns={
        'Facility ID': 'id',
        'Operating Time': 'optime',
        'Gross Load (MW)': 'gross_load_mw',
        'CO2 Mass (short tons)': 'co2_mass_shorttons',
    })
    
    # Calculate additional variables
    merged_df['imports_abs_mw'] = merged_df['imports_mw'].abs()
    merged_df['netimports_mw'] = np.where(merged_df['imports_mw'] < 0, merged_df['imports_mw'] * -1, 0)
    merged_df['netexports_mw'] = np.where(merged_df['imports_mw'] > 0, merged_df['imports_mw'], 0)
    
    # Calculate thermal generation
    df_thermal = merged_df[['datetime', 'gross_load_mw']].groupby('datetime').sum().reset_index()
    df_thermal = df_thermal.rename(columns={'gross_load_mw': 'thermal_generation_mw'})
    merged_df = pd.merge(merged_df, df_thermal, left_on='datetime', right_on='datetime', how='left')
    
    # Add time features
    merged_df['hour'] = merged_df['datetime'].dt.hour
    merged_df['month'] = merged_df['datetime'].dt.month
    merged_df['year'] = merged_df['datetime'].dt.year
    merged_df['day_of_week'] = merged_df['datetime'].dt.dayofweek
    
    # Calculate emissions intensity
    merged_df['co2_emissions_intensity'] = merged_df['co2_mass_shorttons'] / merged_df['gross_load_mw']
    merged_df['co2_emissions_intensity'] = merged_df['co2_emissions_intensity'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Convert SO2 and NOx to kg
    merged_df['so2_mass_kg'] = merged_df['SO2 Mass (lbs)'] * 0.453592
    merged_df['nox_mass_kg'] = merged_df['NOx Mass (lbs)'] * 0.453592
    
    # Calculate SO2 and NOx emissions intensity
    merged_df['so2_emissions_intensity'] = merged_df['so2_mass_kg'] / merged_df['gross_load_mw']
    merged_df['nox_emissions_intensity'] = merged_df['nox_mass_kg'] / merged_df['gross_load_mw']
    merged_df['so2_emissions_intensity'] = merged_df['so2_emissions_intensity'].fillna(0).replace([np.inf, -np.inf], 0)
    merged_df['nox_emissions_intensity'] = merged_df['nox_emissions_intensity'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Calculate hourly ramps (differences between consecutive hours)
    merged_df['solar_ramp'] = merged_df.groupby('id')['solar_generation_mw'].diff().abs()
    merged_df['wind_ramp'] = merged_df.groupby('id')['wind_generation_mw'].diff().abs()
    
    # Calculate shares
    merged_df['solar_share'] = merged_df['solar_generation_mw'] / merged_df['demand_mw']
    merged_df['wind_share'] = merged_df['wind_generation_mw'] / merged_df['demand_mw']
    
    # Calculate residual demand
    merged_df['residual_demand_mw'] = merged_df['demand_mw'] - merged_df['hydro_generation_mw'] + merged_df['imports_mw']
    
    # Drop NaN values
    merged_df = merged_df.dropna()
    
    return merged_df

def run_hourly_regression(df, ba_name):
    """
    Run hourly regression analysis.
    
    Parameters:
        df (pd.DataFrame): Preprocessed hourly data
        ba_name (str): Balancing authority name
    
    Returns:
        dict: Regression results for different dependent variables
    """
    # Define dependent variables
    dependent_vars = ['gross_load_mw', 'co2_mass_shorttons', 'co2_emissions_intensity', 
                     'so2_mass_kg', 'so2_emissions_intensity', 'nox_mass_kg', 'nox_emissions_intensity']
    
    # Define independent variables
    independent_vars = ['solar_generation_mw', 'wind_generation_mw', 'residual_demand_mw', 
                       'wind_ramp', 'solar_ramp']
    
    # Add time fixed effects
    time_effects = ['C(hour)', 'C(month)', 'C(year)', 'C(day_of_week)']
    
    results = {}
    
    for dep_var in dependent_vars:
        print(f"Running regression for {dep_var}...")
        
        # Filter data where dependent variable > 0
        df_filtered = df[df[dep_var] > 0].copy()
        
        if len(df_filtered) == 0:
            print(f"No data for {dep_var}")
            continue
        
        # Log transformation
        df_filtered[dep_var] = np.log(df_filtered[dep_var])
        df_filtered[independent_vars] = np.log(df_filtered[independent_vars].abs() + 1)  # Add 1 to avoid log(0)
        
        # Replace inf and -inf with NaN
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(df_filtered) == 0:
            print(f"No valid data after filtering for {dep_var}")
            continue
        
        # Build formula
        formula = f"{dep_var} ~ {' + '.join(independent_vars)} + {' + '.join(time_effects)} + C(id)"
        
        try:
            # Fit model
            model = smf.ols(formula, data=df_filtered).fit()
            
            # Store results
            results[dep_var] = {
                'model': model,
                'nobs': model.nobs,
                'rsquared': model.rsquared,
                'adj_rsquared': model.rsquared_adj,
                'f_statistic': model.fvalue,
                'f_pvalue': model.f_pvalue
            }
            
            print(f"  R²: {model.rsquared:.4f}, Observations: {model.nobs}")
            
        except Exception as e:
            print(f"  Error in regression for {dep_var}: {e}")
            results[dep_var] = None
    
    return results

def save_hourly_results(results, ba_name):
    """
    Save hourly regression results to CSV.
    
    Parameters:
        results (dict): Regression results
        ba_name (str): Balancing authority name
    """
    # Create results directory if it doesn't exist
    os.makedirs('../results/hourly_regression', exist_ok=True)
    
    # Extract coefficients and statistics
    summary_data = []
    
    for dep_var, result in results.items():
        if result is None:
            continue
            
        model = result['model']
        
        for var in model.params.index:
            if var not in ['Intercept'] and not var.startswith('C('):
                summary_data.append({
                    'Dependent_Variable': dep_var,
                    'Independent_Variable': var,
                    'Coefficient': model.params[var],
                    'Standard_Error': model.bse[var],
                    't_statistic': model.tvalues[var],
                    'p_value': model.pvalues[var],
                    'R_squared': result['rsquared'],
                    'Adjusted_R_squared': result['adj_rsquared'],
                    'Observations': result['nobs'],
                    'F_statistic': result['f_statistic'],
                    'F_p_value': result['f_pvalue']
                })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save to CSV
    output_file = f'../results/hourly_regression/{ba_name}_hourly_regression_results.csv'
    summary_df.to_csv(output_file, index=False)
    print(f"Hourly regression results saved to {output_file}")
    
    return summary_df

def main():
    """
    Main function to run hourly regression analysis for all BAs.
    """
    # List of balancing authorities
    bas = ['CAISO', 'ERCOT', 'MISO', 'NYISO', 'SWPP', 'PJM', 'ISONE']
    
    all_results = {}
    
    for ba in bas:
        print(f"\n{'='*50}")
        print(f"Processing {ba}")
        print(f"{'='*50}")
        
        try:
            # Load data
            cems_df, gen_df = load_hourly_data(ba)
            print(f"Loaded {len(cems_df)} CEMS records and {len(gen_df)} generation records")
            
            # Preprocess data
            merged_df = preprocess_hourly_data(cems_df, gen_df)
            print(f"Preprocessed data: {len(merged_df)} hourly records")
            
            # Run regression
            results = run_hourly_regression(merged_df, ba)
            
            # Save results
            summary_df = save_hourly_results(results, ba)
            
            all_results[ba] = summary_df
            
        except Exception as e:
            print(f"Error processing {ba}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Hourly regression analysis complete!")
    print(f"{'='*50}")
    
    # Print summary statistics
    for ba, results in all_results.items():
        if results is not None:
            print(f"\n{ba}: {len(results)} regression coefficients")
            print(f"  Average R²: {results['R_squared'].mean():.4f}")
            print(f"  Total observations: {results['Observations'].sum():,}")

if __name__ == "__main__":
    main() 