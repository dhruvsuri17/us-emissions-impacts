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

def get_file_mapping(ba):
    """
    Get the correct file paths for CEMS and generation data for a specific BA.
    
    Parameters:
        ba (str): Balancing authority name (e.g., 'CAISO', 'PJM')
    
    Returns:
        dict: Dictionary with 'cems' and 'gen' file paths
    """
    # Mapping of BA names to their actual file names
    ba_mapping = {
        'CAISO': 'CISO',
        'ERCOT': 'ERCO',
        'MISO': 'MISO',
        'NYISO': 'NYIS',
        'SWPP': 'SWPP',
        'PJM': 'PJM',
        'ISONE': 'ISNE'
    }
    
    cems_path = f'data/CEMS_processed/{ba}.csv'
    gen_path = f'data/processed/{ba_mapping.get(ba, ba)}.csv'
    
    return {'cems': cems_path, 'gen': gen_path}

def load_hourly_data(ba_name, year=None):
    """
    Load hourly CEMS and generation data for a specific BA.
    
    Parameters:
        ba_name (str): Balancing authority name (e.g., 'CAISO', 'PJM')
        year (int, optional): Specific year to filter (e.g., 2022)
    
    Returns:
        tuple: (cems_df, gen_df) - CEMS and generation dataframes
    """
    # Get correct file paths
    file_paths = get_file_mapping(ba_name)
    
    # Load CEMS data
    cems_df = pd.read_csv(file_paths['cems'])
    
    # Load generation data
    gen_df = pd.read_csv(file_paths['gen'])
    
    # Filter by year if specified
    if year is not None:
        cems_df['Date'] = pd.to_datetime(cems_df['Date'])
        cems_df = cems_df[cems_df['Date'].dt.year == year]
        
        gen_df['Local time'] = pd.to_datetime(gen_df['Local time'])
        gen_df = gen_df[gen_df['Local time'].dt.year == year]
        
        print(f"Filtered data for year {year}: CEMS {len(cems_df):,} rows, Generation {len(gen_df):,} rows")
    
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
    
    # Debug: Check Facility ID column structure
    print(f"Facility ID column type: {cems_df['Facility ID'].dtype}")
    print(f"Facility ID unique values (first 5): {cems_df['Facility ID'].unique()[:5]}")
    print(f"Facility ID shape: {cems_df['Facility ID'].shape}")
    
    # Ensure Facility ID is 1-dimensional and handle any pandas version issues
    if cems_df['Facility ID'].ndim > 1:
        print("⚠️  Facility ID is multi-dimensional, flattening...")
        cems_df['Facility ID'] = cems_df['Facility ID'].iloc[:, 0] if cems_df['Facility ID'].ndim == 2 else cems_df['Facility ID'].iloc[0]
    
    # Additional safety: ensure Facility ID is a simple Series
    if not isinstance(cems_df['Facility ID'], pd.Series):
        print("⚠️  Facility ID is not a Series, converting...")
        cems_df['Facility ID'] = pd.Series(cems_df['Facility ID'].values.flatten())
    
    # Final check: ensure it's 1D
    if cems_df['Facility ID'].ndim != 1:
        print("❌ Facility ID still not 1D after conversion, using first column")
        cems_df['Facility ID'] = cems_df['Facility ID'].iloc[:, 0] if cems_df['Facility ID'].ndim == 2 else cems_df['Facility ID'].iloc[0]
    
    # Nuclear option: completely reconstruct the Facility ID column
    print("🔧 Reconstructing Facility ID column to ensure compatibility...")
    facility_values = cems_df['Facility ID'].values
    if hasattr(facility_values, 'flatten'):
        facility_values = facility_values.flatten()
    
    # Create a completely new Series
    cems_df = cems_df.drop(columns=['Facility ID'])
    cems_df['Facility ID'] = pd.Series(facility_values, index=cems_df.index)
    
    print(f"✅ Facility ID reconstructed. New type: {type(cems_df['Facility ID'])}, shape: {cems_df['Facility ID'].shape}, ndim: {cems_df['Facility ID'].ndim}")
    
    # Select only numeric columns for summing (exclude datetime and string columns)
    numeric_columns = cems_df.select_dtypes(include=[np.number]).columns.tolist()
    # Also include the datetime and Facility ID columns for grouping
    group_columns = ['datetime', 'Facility ID']
    columns_to_sum = group_columns + numeric_columns
    
    # Ensure all columns exist
    available_columns = [col for col in columns_to_sum if col in cems_df.columns]
    print(f"Columns available for grouping: {available_columns}")
    
    cems_df = cems_df[available_columns].groupby(group_columns).sum().reset_index()
    
    # Rename CEMS columns FIRST (before merge)
    cems_rename_mapping = {
        'Facility ID': 'id',
        'Operating Time': 'optime',
        'Gross Load (MW)': 'gross_load_mw',
        'CO2 Mass (short tons)': 'co2_mass_shorttons',
    }
    
    # Filter to only existing columns
    existing_cems_rename = {k: v for k, v in cems_rename_mapping.items() if k in cems_df.columns}
    print(f"CEMS columns to rename: {existing_cems_rename}")
    
    cems_df = cems_df.rename(columns=existing_cems_rename)
    
    # Process generation data
    columns = ['Local time', 'NG: SUN', 'NG: WND', 'D', 'NG: COL', 'NG: NG', 'NG: WAT', 'TI', 'solar_ext_mw', 'wind_ext_mw', 'demand_ext_mw']
    
    # Check which columns exist
    available_columns = [col for col in columns if col in gen_df.columns]
    gen_df = gen_df[available_columns]
    
    # Rename columns
    column_mapping = {
        'Local time': 'datetime',
        'D': 'demand_mw',
        'NG: SUN': 'solar_generation_mw',
        'NG: WND': 'wind_generation_mw',
        'NG: COL': 'coal_generation_mw',
        'NG: NG': 'natural_gas_generation_mw',
        'TI': 'imports_mw',
        'NG: WAT': 'hydro_generation_mw',
    }
    
    # Only rename columns that exist
    existing_mapping = {k: v for k, v in column_mapping.items() if k in gen_df.columns}
    gen_df = gen_df.rename(columns=existing_mapping)
    
    # Add missing columns with default values to avoid nulls
    missing_columns = set(column_mapping.values()) - set(existing_mapping.values())
    for col in missing_columns:
        gen_df[col] = 0.0
        print(f"Added missing column {col} with default value 0.0")
    
    gen_df['datetime'] = pd.to_datetime(gen_df['datetime'])
    
    # Filter between 2019 and 2023
    gen_df = gen_df[(gen_df['datetime'] >= '2019-01-01') & (gen_df['datetime'] <= '2023-12-31')]
    
    # Merge CEMS and generation data
    merged_df = pd.merge(cems_df, gen_df, left_on='datetime', right_on='datetime', how='left')
    
    # CEMS columns are already renamed, no need to rename again
    
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
    Save hourly regression results to CSV in the same format as daily regression.
    
    Parameters:
        results (dict): Regression results
        ba_name (str): Balancing authority name
    """
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Define the variables we want to include in the summary table
    variables = ['residual_demand_mw', 'solar_generation_mw', 'wind_generation_mw', 'solar_ramp', 'wind_ramp', 'R-squared', 'Num of Obs']
    
    # Initialize results dictionary
    all_results = {}
    
    for dependent_var in results.keys():
        result = results[dependent_var]
        if result is None:
            continue
            
        # Initialize summary for this dependent variable
        summary_dict = {var: [] for var in variables}
        
        # Get coefficients and standard errors
        for var in ['residual_demand_mw', 'solar_generation_mw', 'wind_generation_mw', 'solar_ramp', 'wind_ramp']:
            log_var = f'{var}_log'
            if log_var in result.get('coefficients', {}):
                coef = result['coefficients'][log_var]
                std_err = result['standard_errors'][log_var]
                p_value = result['p_values'][log_var]
                
                # Add significance stars
                stars = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                
                # Format as "coefficient (std_error)***"
                summary_dict[var].append(f"{coef:.4f} ({std_err:.4f}){stars}")
            else:
                summary_dict[var].append("-")
        
        # Add R-squared and number of observations
        summary_dict['R-squared'].append(f"{result.get('rsquared', 0):.4f}")
        summary_dict['Num of Obs'].append(f"{int(result.get('nobs', 0))}")
        
        # Convert to DataFrame
        summary_df = pd.DataFrame.from_dict(summary_dict, orient='index')
        summary_df.columns = [dependent_var]
        
        # Store results
        all_results[dependent_var] = summary_df
    
    # Combine all results into a single DataFrame
    if all_results:
        final_summary_df = pd.concat(all_results.values(), axis=1)
        
        # Save to CSV
        output_file = f'results/{ba_name}_hourly_regression_results.csv'
        final_summary_df.to_csv(output_file)
        print(f"Hourly regression results saved to {output_file}")
        
        return final_summary_df
    else:
        print("No valid results to save")
        return None

def run_analysis_for_year(year=2022):
    """
    Run hourly regression analysis for a specific year.
    
    Parameters:
        year (int): Year to analyze (default: 2022)
    """
    print(f"=== Starting Hourly Regression Analysis for {year} ===")
    
    # List of balancing authorities
    bas = ['CAISO', 'ERCOT', 'MISO', 'NYISO', 'SWPP', 'PJM', 'ISONE']
    
    all_results = {}
    
    for ba in bas:
        print(f"\n{'='*50}")
        print(f"Processing {ba} for {year}")
        print(f"{'='*50}")
        
        try:
            # Load data for specific year
            cems_df, gen_df = load_hourly_data(ba, year)
            print(f"Loaded {len(cems_df)} CEMS records and {len(gen_df)} generation records")
            
            # Check if data is too large and apply memory optimization
            if len(cems_df) > 1000000:  # If more than 1M rows
                print(f"⚠️  Large dataset detected ({len(cems_df):,} rows)")
                print(f"   Using memory optimization for {ba}")
                # Sample 10% for very large datasets
                cems_df = cems_df.sample(frac=0.1, random_state=42)
                print(f"   Sampled to {len(cems_df):,} rows")
            
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
    print(f"Hourly regression analysis for {year} complete!")
    print(f"{'='*50}")
    
    # Print summary statistics
    for ba, results in all_results.items():
        if results is not None:
            print(f"\n{ba}: {len(results.columns)} dependent variables")
            print(f"  R² range: {results.loc['R-squared'].min():.4f} - {results.loc['R-squared'].max():.4f}")
            print(f"  Total observations: {results.loc['Num of Obs'].sum():,}")
    
    return all_results

def main():
    """
    Main function to run hourly regression analysis for all BAs.
    """
    # Run analysis for 2022 (most recent complete year)
    results_2022 = run_analysis_for_year(2022)
    
    # Optionally run for other years
    # results_2021 = run_analysis_for_year(2021)
    # results_2020 = run_analysis_for_year(2020)
    
    print(f"\n{'='*50}")
    print("Hourly regression analysis complete!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 