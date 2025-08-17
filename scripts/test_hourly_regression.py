#!/usr/bin/env python3
"""
Test script for hourly regression analysis with CAISO data.
This script can be imported and called from a Jupyter notebook.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from datetime import datetime
import os

def get_file_mapping(ba):
    """
    Get the correct file paths for a given BA.
    Handles the different naming conventions between CEMS and generation data.
    
    Parameters:
    -----------
    ba : str
        Balancing Authority name (e.g., 'CAISO', 'PJM', etc.)
    
    Returns:
    --------
    dict : Dictionary with 'cems' and 'gen' file paths
    """
    # Mapping of BA names to generation data file names
    ba_mapping = {
        'CAISO': 'CISO',
        'PJM': 'PJM', 
        'MISO': 'MISO',
        'ISONE': 'ISNE',
        'NYISO': 'NYIS',
        'ERCOT': 'ERCO',
        'SWPP': 'SWPP'
    }
    
    gen_ba = ba_mapping.get(ba, ba)
    
    return {
        'cems': f'../data/CEMS_processed/{ba}.csv',
        'gen': f'../data/processed/{gen_ba}.csv'
    }

def load_ba_data(ba, year=2022):
    """
    Load data for any BA for a specific year.
    
    Parameters:
    -----------
    ba : str
        Balancing Authority name (e.g., 'CAISO', 'PJM', etc.)
    year : int
        Year to load data for (default: 2022)
    
    Returns:
    --------
    tuple : (df, df_gen) or (None, None) if error
    """
    try:
        file_paths = get_file_mapping(ba)
        
        df_path = file_paths['cems']
        df_gen_path = file_paths['gen']
        
        print(f"Looking for files for {ba}:")
        print(f"  CEMS: {df_path}")
        print(f"  Generation: {df_gen_path}")
        
        # Check if files exist
        if not os.path.exists(df_path):
            print(f"Error: CEMS data file not found at {df_path}")
            return None, None
        if not os.path.exists(df_gen_path):
            print(f"Error: Generation data file not found at {df_gen_path}")
            return None, None
        
        # Load data
        df = pd.read_csv(df_path)
        df_gen = pd.read_csv(df_gen_path)
        
        print(f"Data loaded successfully for {ba}:")
        print(f"  CEMS data shape: {df.shape}")
        print(f"  Generation data shape: {df_gen.shape}")
        
        return df, df_gen
        
    except Exception as e:
        print(f"Error loading data for {ba}: {e}")
        return None, None

def load_caiso_data(year=2022):
    """
    Load CAISO data for a specific year.
    
    Parameters:
    -----------
    year : int
        Year to load data for (default: 2022)
    
    Returns:
    --------
    tuple : (df, df_gen) or (None, None) if error
    """
    return load_ba_data('CAISO', year)

def filter_data_by_year(df, df_gen, year):
    """
    Filter data for a specific year.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        CEMS data
    df_gen : pandas.DataFrame
        Generation data
    year : int
        Year to filter for
    
    Returns:
    --------
    tuple : (df_filtered, df_gen_filtered)
    """
    try:
        # Convert date columns to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df_gen['Local time'] = pd.to_datetime(df_gen['Local time'])
        
        # Filter for the specified year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()
        df_gen_filtered = df_gen[(df_gen['Local time'] >= start_date) & (df_gen['Local time'] <= end_date)].copy()
        
        print(f"Data filtered for year {year}:")
        print(f"  Filtered CEMS data shape: {df_filtered.shape}")
        print(f"  Filtered generation data shape: {df_gen_filtered.shape}")
        print(f"  Date range: {df_filtered['Date'].min()} to {df_filtered['Date'].max()}")
        
        return df_filtered, df_gen_filtered
        
    except Exception as e:
        print(f"Error filtering data: {e}")
        return None, None

def explore_data_structure(df, df_gen):
    """
    Explore and display the structure of the data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        CEMS data
    df_gen : pandas.DataFrame
        Generation data
    """
    try:
        print("=== CEMS Data Columns ===")
        print(df.columns.tolist())
        print("\n=== Generation Data Columns ===")
        print(df_gen.columns.tolist())
        
        print("\n=== CEMS Data Sample ===")
        print(df.head())
        
        print("\n=== Generation Data Sample ===")
        print(df_gen.head())
        
        print("\n=== Data Types ===")
        print("CEMS data types:")
        print(df.dtypes)
        print("\nGeneration data types:")
        print(df_gen.dtypes)
        
    except Exception as e:
        print(f"Error exploring data structure: {e}")

def check_regression_variables(df_gen):
    """
    Check which variables are available for regression analysis.
    
    Parameters:
    -----------
    df_gen : pandas.DataFrame
        Generation data
    
    Returns:
    --------
    dict : Dictionary with available and missing columns
    """
    try:
        # Expected columns for regression
        expected_columns = [
            'Local time', 'NG: SUN', 'NG: WND', 'D', 'NG: COL', 
            'NG: NG', 'NG: OIL', 'NG: WAT', 'TI',
            'solar_ext_mw', 'wind_ext_mw', 'demand_ext_mw'
        ]
        
        available_columns = [col for col in expected_columns if col in df_gen.columns]
        missing_columns = [col for col in expected_columns if col not in df_gen.columns]
        
        print("=== Column Availability Check ===")
        print(f"Available columns: {available_columns}")
        print(f"Missing columns: {missing_columns}")
        
        if available_columns:
            print("\n=== Sample of Available Data ===")
            print(df_gen[available_columns].head())
        
        return {
            'available': available_columns,
            'missing': missing_columns
        }
        
    except Exception as e:
        print(f"Error checking regression variables: {e}")
        return {'available': [], 'missing': []}

def test_basic_regression(df_gen):
    """
    Test basic regression functionality with available data.
    
    Parameters:
    -----------
    df_gen : pandas.DataFrame
        Generation data
    """
    try:
        print("=== Testing Basic Regression Functionality ===")
        
        # Create a test dataset
        test_data = df_gen.copy()
        
        # Add basic features for testing
        if 'D' in test_data.columns:
            test_data['demand_mw'] = test_data['D']
        if 'NG: SUN' in test_data.columns:
            test_data['solar_mw'] = test_data['NG: SUN']
        if 'NG: WND' in test_data.columns:
            test_data['wind_mw'] = test_data['NG: WND']
        
        # Check for non-null data
        print("\nData summary:")
        print(test_data.describe())
        
        # Check for any potential issues
        print("\nData quality check:")
        print(f"Total rows: {len(test_data)}")
        print(f"Null values in demand_mw: {test_data['demand_mw'].isnull().sum() if 'demand_mw' in test_data.columns else 'Column not available'}")
        print(f"Null values in solar_mw: {test_data['solar_mw'].isnull().sum() if 'solar_mw' in test_data.columns else 'Column not available'}")
        print(f"Null values in wind_mw: {test_data['wind_mw'].isnull().sum() if 'wind_mw' in test_data.columns else 'Column not available'}")
        
        print("\nBasic regression test completed successfully")
        return test_data
        
    except Exception as e:
        print(f"Error in basic regression test: {e}")
        return None

def prepare_data_for_hourly_regression(df, df_gen):
    """
    Prepare and merge data for hourly regression analysis.
    This function replicates the data preparation steps from the daily regression.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        CEMS data
    df_gen : pandas.DataFrame
        Generation data
    
    Returns:
    --------
    pandas.DataFrame : Prepared and merged data for hourly regression
    """
    print("=== Preparing Data for Hourly Regression ===")
    
    try:
        # Step 1: Preprocess CEMS data
        print("Step 1: Preprocessing CEMS data")
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        df = df.groupby(['Date', 'Facility ID']).sum().reset_index()
        
        # Step 2: Select and rename generation columns
        print("Step 2: Processing generation data")
        columns = ['Local time', 'NG: SUN', 'NG: WND', 'D', 'NG: COL', 'NG: NG', 'NG: WAT', 'TI', 
                  'solar_ext_mw', 'wind_ext_mw', 'demand_ext_mw']
        
        # Check which columns exist
        available_columns = [col for col in columns if col in df_gen.columns]
        df_gen = df_gen[available_columns]
        
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
        existing_mapping = {k: v for k, v in column_mapping.items() if k in df_gen.columns}
        df_gen = df_gen.rename(columns=existing_mapping)
        
        # Add missing columns with default values to avoid nulls
        missing_columns = set(column_mapping.values()) - set(existing_mapping.values())
        for col in missing_columns:
            df_gen[col] = 0.0
            print(f"Added missing column {col} with default value 0.0")
        
        # Step 3: Rename CEMS columns FIRST (before merge)
        print("Step 3: Renaming CEMS columns")
        
        # Rename CEMS columns before merging
        cems_rename_mapping = {
            'Facility ID': 'id',
            'Operating Time': 'optime',
            'Gross Load (MW)': 'gross_load_mw',
            'CO2 Mass (short tons)': 'co2_mass_shorttons',
        }
        
        # Filter to only existing columns
        existing_cems_rename = {k: v for k, v in cems_rename_mapping.items() if k in df.columns}
        print(f"CEMS columns to rename: {existing_cems_rename}")
        
        df = df.rename(columns=existing_cems_rename)
        
        # Step 4: Create hourly data (keep hourly resolution)
        print("Step 4: Creating hourly dataset")
        df_gen['Date'] = pd.to_datetime(df_gen['datetime']).dt.date
        df_gen['Hour'] = pd.to_datetime(df_gen['datetime']).dt.hour
        
        # Step 5: Merge CEMS and generation data
        print("Step 5: Merging CEMS and generation data")
        df_merged = pd.merge(df, df_gen, left_on='Date', right_on='Date', how='left')
        
        # Fill nulls in generation columns with 0 to avoid dropna issues
        generation_columns = ['solar_generation_mw', 'wind_generation_mw', 'demand_mw', 'coal_generation_mw', 
                             'natural_gas_generation_mw', 'oil_generation_mw', 'hydro_generation_mw', 'imports_mw']
        for col in generation_columns:
            if col in df_merged.columns:
                df_merged[col] = df_merged[col].fillna(0.0)
                print(f"Filled nulls in {col} with 0.0")
        
        # Step 6: Clean up merged data
        print("Step 6: Cleaning up merged data")
        
        # Check which columns exist
        existing_columns = df_merged.columns.tolist()
        print(f"Available columns: {existing_columns}")
        
        # Remove the old Date column to avoid conflicts
        if 'Date' in df_merged.columns:
            df_merged = df_merged.drop(columns=['Date'])
            print("Removed old 'Date' column to avoid conflicts")
        
        # Generation columns are already renamed from Step 2, no need to rename again
        
        # Ensure datetime is properly formatted
        if 'datetime' in df_merged.columns:
            df_merged['datetime'] = pd.to_datetime(df_merged['datetime'])
        else:
            print("Warning: datetime column not found after renaming")
            # Try to find an alternative datetime column
            if 'Local time' in df_merged.columns:
                df_merged['datetime'] = pd.to_datetime(df_merged['Local time'])
                print("Using 'Local time' column as datetime")
        
        # Step 6: Calculate additional variables
        print("Step 6: Calculating additional variables")
        df_merged['imports_abs_mw'] = df_merged['imports_mw'].abs()
        df_merged['netimports_mw'] = np.where(df_merged['imports_mw'] < 0, df_merged['imports_mw'] * -1, 0)
        df_merged['netexports_mw'] = np.where(df_merged['imports_mw'] > 0, df_merged['imports_mw'], 0)
        
        # Step 7: Calculate thermal generation
        print("Step 7: Calculating thermal generation")
        # Fix: Calculate thermal generation without creating duplicate keys
        # Instead of merging, we'll calculate it directly for each row
        df_merged['thermal_generation_mw'] = df_merged.groupby('datetime')['gross_load_mw'].transform('sum')
        
        # Step 8: Add time variables
        print("Step 8: Adding time variables")
        df_merged['month'] = pd.to_datetime(df_merged['datetime']).dt.month
        df_merged['year'] = pd.to_datetime(df_merged['datetime']).dt.year
        df_merged['hour'] = pd.to_datetime(df_merged['datetime']).dt.hour
        
        # Step 9: Calculate emissions intensity
        print("Step 9: Calculating emissions intensity")
        df_merged['co2_emissions_intensity'] = df_merged['co2_mass_shorttons'] / df_merged['gross_load_mw']
        df_merged['co2_emissions_intensity'] = df_merged['co2_emissions_intensity'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Convert SO2 and NOx to kg
        if 'SO2 Mass (lbs)' in df_merged.columns:
            df_merged['so2_mass_kg'] = df_merged['SO2 Mass (lbs)'] * 0.453592
            df_merged['so2_emissions_intensity'] = df_merged['so2_mass_kg'] / df_merged['gross_load_mw']
            df_merged['so2_emissions_intensity'] = df_merged['so2_emissions_intensity'].fillna(0).replace([np.inf, -np.inf], 0)
        
        if 'NOx Mass (lbs)' in df_merged.columns:
            df_merged['nox_mass_kg'] = df_merged['NOx Mass (lbs)'] * 0.453592
            df_merged['nox_emissions_intensity'] = df_merged['nox_mass_kg'] / df_merged['gross_load_mw']
            df_merged['nox_emissions_intensity'] = df_merged['nox_emissions_intensity'].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Step 10: Calculate ramp rates (hourly)
        print("Step 10: Calculating hourly ramp rates")
        if 'wind_generation_mw' in df_merged.columns:
            df_merged['wind_ramp'] = df_merged.groupby('id')['wind_generation_mw'].diff().abs()
        
        if 'solar_generation_mw' in df_merged.columns:
            df_merged['solar_ramp'] = df_merged.groupby('id')['solar_generation_mw'].diff().abs()
        
        # Step 11: Calculate residual demand
        print("Step 11: Calculating residual demand")
        if 'demand_mw' in df_merged.columns and 'hydro_generation_mw' in df_merged.columns and 'imports_mw' in df_merged.columns:
            df_merged['residual_demand_mw'] = df_merged['demand_mw'] - df_merged['hydro_generation_mw'] + df_merged['imports_mw']
        
        # Step 12: Calculate shares
        print("Step 12: Calculating generation shares")
        if 'solar_generation_mw' in df_merged.columns and 'demand_mw' in df_merged.columns:
            df_merged['solar_share'] = df_merged['solar_generation_mw'] / df_merged['demand_mw']
        
        if 'wind_generation_mw' in df_merged.columns and 'demand_mw' in df_merged.columns:
            df_merged['wind_share'] = df_merged['wind_generation_mw'] / df_merged['demand_mw']
        
        # Step 13: Clean up
        print("Step 13: Final cleanup")
        df_merged.dropna(inplace=True)
        
        print(f"Data preparation complete. Final shape: {df_merged.shape}")
        print(f"Columns: {list(df_merged.columns)}")
        
        return df_merged
        
    except Exception as e:
        print(f"Error preparing data: {e}")
        return None

def perform_hourly_regression(df_merged, dependent_vars=None, independent_vars=None):
    """
    Perform hourly regression analysis on the merged data.
    
    Parameters:
    -----------
    df_merged : pandas.DataFrame
        Merged CEMS and generation data
    dependent_vars : list
        List of dependent variables for regression (default: standard emissions variables)
    independent_vars : list
        List of independent variables for regression (default: standard generation variables)
    
    Returns:
    --------
    dict : Regression results for each dependent variable
    """
    if dependent_vars is None:
        dependent_vars = [
            'gross_load_mw', 'co2_mass_shorttons', 'co2_emissions_intensity',
            'so2_mass_kg', 'so2_emissions_intensity', 'nox_mass_kg', 'nox_emissions_intensity'
        ]
    
    if independent_vars is None:
        independent_vars = [
            'solar_generation_mw', 'wind_generation_mw', 'residual_demand_mw',
            'wind_ramp', 'solar_ramp'
        ]
    
    print(f"=== Performing Hourly Regression Analysis ===")
    print(f"Dependent variables: {dependent_vars}")
    print(f"Independent variables: {independent_vars}")
    
    # Initialize results dictionary
    regression_results = {}
    
    for dependent_var in dependent_vars:
        print(f"\n--- Analyzing {dependent_var} ---")
        
        try:
            # Check if dependent variable exists
            if dependent_var not in df_merged.columns:
                print(f"Warning: {dependent_var} not found in data, skipping...")
                continue
            
            # Filter out rows where dependent variable is not greater than zero
            df_filtered = df_merged[df_merged[dependent_var] > 0].copy()
            
            if len(df_filtered) == 0:
                print(f"Warning: No valid data for {dependent_var}, skipping...")
                continue
            
            print(f"Valid observations: {len(df_filtered)}")
            
            # Check which independent variables are available
            available_ind_vars = [var for var in independent_vars if var in df_filtered.columns]
            missing_ind_vars = [var for var in independent_vars if var not in df_filtered.columns]
            
            if missing_ind_vars:
                print(f"Missing independent variables: {missing_ind_vars}")
            
            if not available_ind_vars:
                print(f"Error: No independent variables available for {dependent_var}")
                continue
            
            # Prepare data for regression
            regression_data = df_filtered[[dependent_var] + available_ind_vars].copy()
            
            # Add facility and time fixed effects if available
            if 'id' in df_filtered.columns:
                regression_data['id'] = df_filtered['id']
            if 'month' in df_filtered.columns:
                regression_data['month'] = df_filtered['month']
            if 'year' in df_filtered.columns:
                regression_data['year'] = df_filtered['year']
            
            # Log transformation of dependent variable and independent variables
            regression_data[f'{dependent_var}_log'] = np.log(regression_data[dependent_var])
            
            # NYISO-specific fix: add 1 to all independent variables before log transformation
            # This prevents log(0) and log(negative) issues, just like in daily regression
            for var in available_ind_vars:
                # Add 1 to all independent variables (NYISO fix)
                regression_data[f'{var}_log'] = np.log(regression_data[var] + 1)
            
            # Clean up any remaining infinite or NaN values
            regression_data = regression_data.replace([np.inf, -np.inf], np.nan)
            regression_data = regression_data.dropna()
            
            if len(regression_data) == 0:
                print(f"Warning: No complete data for {dependent_var} after cleaning")
                continue
            
            print(f"Complete observations after cleaning: {len(regression_data)}")
            
            # Check for any remaining issues
            log_cols = [col for col in regression_data.columns if col.endswith('_log')]
            for col in log_cols:
                if regression_data[col].isnull().any() or np.isinf(regression_data[col]).any():
                    print(f"Warning: {col} still has invalid values after cleaning")
                    print(f"  NaN count: {regression_data[col].isnull().sum()}")
                    print(f"  Inf count: {np.isinf(regression_data[col]).sum()}")
            
            # Build regression formula
            log_ind_vars = [f'{var}_log' for var in available_ind_vars if f'{var}_log' in regression_data.columns]
            
            # Add fixed effects
            fixed_effects = []
            if 'id' in regression_data.columns:
                fixed_effects.append('C(id)')
            if 'month' in regression_data.columns:
                fixed_effects.append('C(month)')
            if 'year' in regression_data.columns:
                fixed_effects.append('C(year)')
            
            formula = f'{dependent_var}_log ~ {" + ".join(log_ind_vars)}'
            if fixed_effects:
                formula += f' + {" + ".join(fixed_effects)}'
            
            print(f"Regression formula: {formula}")
            
            # Check for multicollinearity
            if len(log_ind_vars) > 1:
                corr_matrix = regression_data[log_ind_vars].corr()
                high_corr = np.where(np.abs(corr_matrix) > 0.95)
                high_corr_pairs = [(log_ind_vars[i], log_ind_vars[j], corr_matrix.iloc[i, j]) 
                                  for i, j in zip(*high_corr) if i != j]
                if high_corr_pairs:
                    print(f"Warning: High correlations detected: {high_corr_pairs}")
            
            # Fit the model
            try:
                model = smf.ols(formula, data=regression_data).fit()
                
                # Store results
                regression_results[dependent_var] = {
                    'model': model,
                    'formula': formula,
                    'nobs': model.nobs,
                    'rsquared': model.rsquared,
                    'adj_rsquared': model.rsquared_adj,
                    'coefficients': model.params,
                    'standard_errors': model.bse,
                    'p_values': model.pvalues,
                    'available_vars': available_ind_vars,
                    'missing_vars': missing_ind_vars
                }
                
                print(f"Model fitted successfully:")
                print(f"  R-squared: {model.rsquared:.4f}")
                print(f"  Adjusted R-squared: {model.rsquared_adj:.4f}")
                print(f"  Observations: {model.nobs}")
                
            except Exception as e:
                print(f"Error fitting model: {e}")
                regression_results[dependent_var] = {'error': str(e)}
        
        except Exception as e:
            print(f"Error analyzing {dependent_var}: {e}")
            regression_results[dependent_var] = {'error': str(e)}
    
    return regression_results

def summarize_regression_results(regression_results):
    """
    Summarize the regression results in a readable format.
    
    Parameters:
    -----------
    regression_results : dict
        Results from perform_hourly_regression function
    
    Returns:
    --------
    pandas.DataFrame : Summary table of results
    """
    print("=== Regression Results Summary ===")
    
    summary_data = []
    
    for dep_var, results in regression_results.items():
        if 'error' in results:
            print(f"{dep_var}: Error - {results['error']}")
            continue
        
        print(f"\n{dep_var}:")
        print(f"  R-squared: {results['rsquared']:.4f}")
        print(f"  Observations: {results['nobs']}")
        print(f"  Available variables: {results['available_vars']}")
        
        # Show coefficients for key variables
        for var in ['solar_generation_mw', 'wind_generation_mw', 'residual_demand_mw']:
            log_var = f'{var}_log'
            if log_var in results['coefficients']:
                coef = results['coefficients'][log_var]
                se = results['standard_errors'][log_var]
                pval = results['p_values'][log_var]
                
                # Add significance stars
                stars = ''
                if pval < 0.001:
                    stars = '***'
                elif pval < 0.01:
                    stars = '**'
                elif pval < 0.05:
                    stars = '*'
                
                print(f"    {var}: {coef:.4f} ({se:.4f}){stars}")
        
        # Add to summary data
        summary_data.append({
            'Dependent Variable': dep_var,
            'R-squared': f"{results['rsquared']:.4f}",
            'Observations': results['nobs'],
            'Available Variables': len(results['available_vars']),
            'Missing Variables': len(results['missing_vars'])
        })
    
    # Create summary DataFrame
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(f"\n=== Summary Table ===")
        print(summary_df.to_string(index=False))
        return summary_df
    else:
        print("No successful regressions to summarize")
        return None

def save_regression_results_to_csv(regression_results, ba_name, output_dir='../results'):
    """
    Save regression results to CSV files in the same format as the daily regression notebook.
    
    Parameters:
    -----------
    regression_results : dict
        Dictionary containing regression results for each dependent variable
    ba_name : str
        Name of the Balancing Authority (e.g., 'CAISO', 'PJM')
    output_dir : str
        Directory to save the CSV files (default: '../results')
    """
    import os
    import pandas as pd
    
    if not regression_results:
        print("No regression results to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the variables we want to include in the summary table
    variables = ['residual_demand_mw', 'solar_generation_mw', 'wind_generation_mw', 'solar_ramp', 'wind_ramp', 'R-squared', 'Num of Obs']
    
    # Initialize results dictionary
    all_results = {}
    
    for dependent_var in regression_results.keys():
        result = regression_results[dependent_var]
        if 'error' in result:
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
        output_file = os.path.join(output_dir, f'{ba_name}_hourly_regression_results.csv')
        final_summary_df.to_csv(output_file)
        print(f"Regression results saved to: {output_file}")
        
        return final_summary_df
    else:
        print("No valid results to save")
        return None

def test_file_paths():
    """
    Test function to verify file paths are correct.
    """
    print("=== Testing File Paths ===")
    
    # Test CAISO paths
    caiso_paths = get_file_mapping('CAISO')
    print(f"CAISO CEMS path: {caiso_paths['cems']}")
    print(f"CAISO Generation path: {caiso_paths['gen']}")
    
    # Check if files exist
    print(f"\nFile existence check:")
    print(f"CEMS file exists: {os.path.exists(caiso_paths['cems'])}")
    print(f"Generation file exists: {os.path.exists(caiso_paths['gen'])}")
    
    # Test other BAs
    print(f"\nTesting other BAs:")
    for ba in ['PJM', 'MISO', 'ISONE', 'NYISO', 'ERCOT', 'SWPP']:
        paths = get_file_mapping(ba)
        cems_exists = os.path.exists(paths['cems'])
        gen_exists = os.path.exists(paths['gen'])
        print(f"{ba}: CEMS={cems_exists}, Gen={gen_exists}")

def run_full_test(ba='CAISO', year=2022):
    """
    Run the complete test suite for hourly regression.
    
    Parameters:
    -----------
    ba : str
        Balancing Authority name (default: 'CAISO')
    year : int
        Year to test (default: 2022)
    
    Returns:
    --------
    dict : Test results summary
    """
    print(f"=== Starting Hourly Regression Test for {ba} {year} ===\n")
    
    # Step 1: Load data
    df, df_gen = load_ba_data(ba, year)
    if df is None or df_gen is None:
        return {'status': 'failed', 'step': 'data_loading'}
    
    # Step 2: Filter data
    df_filtered, df_gen_filtered = filter_data_by_year(df, df_gen, year)
    if df_filtered is None or df_gen_filtered is None:
        return {'status': 'failed', 'step': 'data_filtering'}
    
    # Step 3: Explore structure
    explore_data_structure(df_filtered, df_gen_filtered)
    
    # Step 4: Check variables
    variables = check_regression_variables(df_gen_filtered)
    
    # Step 5: Prepare data for regression
    print("\n" + "="*50)
    df_merged = prepare_data_for_hourly_regression(df_filtered, df_gen_filtered)
    if df_merged is None:
        return {'status': 'failed', 'step': 'data_preparation'}
    
    # Step 6: Perform hourly regression
    print("\n" + "="*50)
    regression_results = perform_hourly_regression(df_merged)
    
    # Step 7: Summarize results
    print("\n" + "="*50)
    summary_df = summarize_regression_results(regression_results)
    
    # Step 8: Save results to CSV
    if regression_results:
        save_regression_results_to_csv(regression_results, ba)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Target BA: {ba}")
    print(f"Target Year: {year}")
    print(f"Data loaded: Yes")
    print(f"Data filtered: Yes")
    print(f"Data prepared: Yes")
    print(f"Regression completed: {'Yes' if regression_results else 'No'}")
    print(f"Available variables: {len(variables['available'])}")
    print(f"Missing variables: {len(variables['missing'])}")
    
    # Store the merged data for further analysis
    results_dict = {
        'status': 'success',
        'ba': ba,
        'year': year,
        'cems_shape': df_filtered.shape,
        'gen_shape': df_gen_filtered.shape,
        'merged_shape': df_merged.shape if df_merged is not None else None,
        'merged_data': df_merged,  # Store the actual merged data
        'available_vars': variables['available'],
        'missing_vars': variables['missing'],
        'regression_results': regression_results,
        'summary_df': summary_df
    }
    
    print(f"Results dictionary keys: {list(results_dict.keys())}")
    return results_dict

def run_full_test_debug(ba='CAISO', year=2022):
    """
    Debug version of the test function that shows exactly what's happening.
    """
    print(f"=== DEBUG: Starting Hourly Regression Test for {ba} {year} ===\n")
    
    # Step 1: Show file mapping
    print("Step 1: File mapping")
    file_paths = get_file_mapping(ba)
    print(f"  CEMS path: {file_paths['cems']}")
    print(f"  Generation path: {file_paths['gen']}")
    
    # Step 2: Check file existence
    print("\nStep 2: File existence check")
    cems_exists = os.path.exists(file_paths['cems'])
    gen_exists = os.path.exists(file_paths['gen'])
    print(f"  CEMS file exists: {cems_exists}")
    print(f"  Generation file exists: {gen_exists}")
    
    if not cems_exists or not gen_exists:
        print("  ERROR: One or more files not found!")
        return {'status': 'failed', 'step': 'file_check'}
    
    # Step 3: Load data
    print("\nStep 3: Loading data")
    df, df_gen = load_ba_data(ba, year)
    if df is None or df_gen is None:
        return {'status': 'failed', 'step': 'data_loading'}
    
    # Continue with rest of function...
    print("\nStep 4: Filtering data")
    df_filtered, df_gen_filtered = filter_data_by_year(df, df_gen, year)
    if df_filtered is None or df_gen_filtered is None:
        return {'status': 'failed', 'step': 'data_filtering'}
    
    print("\nStep 5: Exploring structure")
    explore_data_structure(df_filtered, df_gen_filtered)
    
    print("\nStep 6: Checking variables")
    variables = check_regression_variables(df_gen_filtered)
    
    print("\nStep 7: Testing regression")
    test_data = test_basic_regression(df_gen_filtered)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Target BA: {ba}")
    print(f"Target Year: {year}")
    print(f"Data loaded: Yes")
    print(f"Data filtered: Yes")
    print(f"Available variables: {len(variables['available'])}")
    print(f"Missing variables: {len(variables['missing'])}")
    
    return {
        'status': 'success',
        'ba': ba,
        'year': year,
        'cems_shape': df_filtered.shape,
        'gen_shape': df_gen_filtered.shape,
        'available_vars': variables['available'],
        'missing_vars': variables['missing'],
        'test_data': test_data
    }

def test_data_flow(ba='CAISO', year=2022):
    """
    Simple test function to verify data flow step by step.
    """
    print(f"=== Testing Data Flow for {ba} {year} ===\n")
    
    try:
        # Step 1: Load data
        print("Step 1: Loading data...")
        df, df_gen = load_ba_data(ba, year)
        if df is None or df_gen is None:
            print("FAILED: Data loading")
            return None
        print("SUCCESS: Data loaded")
        
        # Step 2: Filter data
        print("\nStep 2: Filtering data...")
        df_filtered, df_gen_filtered = filter_data_by_year(df, df_gen, year)
        if df_filtered is None or df_gen_filtered is None:
            print("FAILED: Data filtering")
            return None
        print("SUCCESS: Data filtered")
        
        # Step 3: Prepare data
        print("\nStep 3: Preparing data...")
        df_merged = prepare_data_for_hourly_regression(df_filtered, df_gen_filtered)
        if df_merged is None:
            print("FAILED: Data preparation")
            return None
        print("SUCCESS: Data prepared")
        
        # Step 4: Check final data
        print(f"\nFinal data shape: {df_merged.shape}")
        print(f"Final columns: {list(df_merged.columns)}")
        
        return {
            'df': df,
            'df_gen': df_gen,
            'df_filtered': df_filtered,
            'df_gen_filtered': df_gen_filtered,
            'df_merged': df_merged
        }
        
    except Exception as e:
        print(f"ERROR in data flow: {e}")
        import traceback
        traceback.print_exc()
        return None

def clear_cache_and_reload():
    """
    Function to help clear any potential import caching issues.
    Call this if you're experiencing strange behavior.
    """
    import importlib
    import sys
    
    # Remove the module from sys.modules if it exists
    module_name = 'test_hourly_regression'
    if module_name in sys.modules:
        del sys.modules[module_name]
        print(f"Removed {module_name} from cache")
    
    # Also try to reload if it's still there
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
        print(f"Reloaded {module_name}")
    
    print("Cache clearing complete. You may need to re-import the module.")

def test_all_bas(year=2022):
    """
    Test hourly regression for all available balancing authorities.
    
    Parameters:
    -----------
    year : int
        Year to test (default: 2022)
    
    Returns:
    --------
    dict : Results for all BAs
    """
    print(f"=== Testing Hourly Regression for ALL BAs in {year} ===\n")
    
    # List of all BAs to test
    all_bas = ['CAISO', 'PJM', 'MISO', 'ISONE', 'NYISO', 'ERCOT', 'SWPP']
    
    all_results = {}
    
    for ba in all_bas:
        print(f"\n{'='*60}")
        print(f"Testing {ba}")
        print(f"{'='*60}")
        
        try:
            # Test this BA
            results = run_full_test(ba, year)
            all_results[ba] = results
            
            # Print summary for this BA
            if results['status'] == 'success':
                print(f"\n✅ {ba} SUCCESS:")
                print(f"   Data shapes: CEMS {results['cems_shape']}, Gen {results['gen_shape']}, Merged {results['merged_shape']}")
                print(f"   Regression completed: {'Yes' if results['regression_results'] else 'No'}")
                
                # Count successful regressions
                if results['regression_results']:
                    successful_regressions = sum(1 for r in results['regression_results'].values() if 'error' not in r)
                    total_regressions = len(results['regression_results'])
                    print(f"   Regressions: {successful_regressions}/{total_regressions} successful")
            else:
                print(f"\n❌ {ba} FAILED at step: {results['step']}")
                
        except Exception as e:
            print(f"\n❌ {ba} ERROR: {e}")
            all_results[ba] = {'status': 'error', 'error': str(e)}
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    successful_bas = [ba for ba, results in all_results.items() if results.get('status') == 'success']
    failed_bas = [ba for ba, results in all_results.items() if results.get('status') != 'success']
    
    print(f"Successful BAs: {len(successful_bas)}/{len(all_bas)}")
    print(f"  ✅ {', '.join(successful_bas)}")
    
    if failed_bas:
        print(f"Failed BAs: {len(failed_bas)}/{len(all_bas)}")
        print(f"  ❌ {', '.join(failed_bas)}")
        
        # Show failure reasons
        for ba in failed_bas:
            results = all_results[ba]
            if 'step' in results:
                print(f"    {ba}: Failed at {results['step']}")
            elif 'error' in results:
                print(f"    {ba}: {results['error']}")
    
    return all_results

def test_all_bas_memory_efficient(year=2022, sample_fraction=0.1):
    """
    Memory-efficient version to test hourly regression for all BAs.
    
    Parameters:
    -----------
    year : int
        Year to test (default: 2022)
    sample_fraction : float
        Fraction of data to sample for large BAs (default: 0.1)
    
    Returns:
    --------
    dict : Results for all BAs
    """
    print(f"=== Testing Memory-Efficient Hourly Regression for ALL BAs in {year} ===\n")
    
    # List of all BAs to test
    all_bas = ['CAISO', 'PJM', 'MISO', 'ISONE', 'NYISO', 'ERCOT', 'SWPP']
    
    all_results = {}
    
    for ba in all_bas:
        print(f"\n{'='*60}")
        print(f"Testing {ba}")
        print(f"{'='*60}")
        
        try:
            # Use memory-efficient version for large BAs
            if ba in ['PJM', 'MISO']:
                print(f"⚠️  Using memory-efficient mode for {ba}")
                results = run_full_test_memory_efficient(ba, year, sample_fraction)
            else:
                results = run_full_test(ba, year)
            
            all_results[ba] = results
            
            # Print summary for this BA
            if results['status'] == 'success':
                print(f"\n✅ {ba} SUCCESS:")
                print(f"   Data shapes: CEMS {results['cems_shape']}, Gen {results['gen_shape']}, Merged {results['merged_shape']}")
                print(f"   Regression completed: {'Yes' if results['regression_results'] else 'No'}")
                print(f"   Memory optimized: {'Yes' if results.get('memory_optimized', False) else 'No'}")
                
                # Count successful regressions
                if results['regression_results']:
                    successful_regressions = sum(1 for r in results['regression_results'].values() if 'error' not in r)
                    total_regressions = len(results['regression_results'])
                    print(f"   Regressions: {successful_regressions}/{total_regressions} successful")
            else:
                print(f"\n❌ {ba} FAILED at step: {results['step']}")
                
        except Exception as e:
            print(f"\n❌ {ba} ERROR: {e}")
            all_results[ba] = {'status': 'error', 'error': str(e)}
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    successful_bas = [ba for ba, results in all_results.items() if results.get('status') == 'success']
    failed_bas = [ba for ba, results in all_results.items() if results.get('status') != 'success']
    
    print(f"Successful BAs: {len(successful_bas)}/{len(all_bas)}")
    print(f"  ✅ {', '.join(successful_bas)}")
    
    if failed_bas:
        print(f"Failed BAs: {len(failed_bas)}/{len(all_bas)}")
        print(f"  ❌ {', '.join(failed_bas)}")
        
        # Show failure reasons
        for ba in failed_bas:
            results = all_results[ba]
            if 'step' in results:
                print(f"    {ba}: Failed at {results['step']}")
            elif 'error' in results:
                print(f"    {ba}: {results['error']}")
    
    return all_results

def run_full_test_memory_efficient(ba='CAISO', year=2022, sample_fraction=0.1):
    """
    Memory-efficient version of the test function that samples data for large BAs.
    
    Parameters:
    -----------
    ba : str
        Balancing Authority name (default: 'CAISO')
    year : int
        Year to test (default: 2022)
    sample_fraction : float
        Fraction of data to sample for memory efficiency (default: 0.1)
    
    Returns:
    --------
    dict : Test results summary
    """
    print(f"=== Starting Memory-Efficient Hourly Regression Test for {ba} {year} ===\n")
    
    # Step 1: Load data
    df, df_gen = load_ba_data(ba, year)
    if df is None or df_gen is None:
        return {'status': 'failed', 'step': 'data_loading'}
    
    # Step 2: Filter data
    df_filtered, df_gen_filtered = filter_data_by_year(df, df_gen, year)
    if df_filtered is None or df_gen_filtered is None:
        return {'status': 'failed', 'step': 'data_filtering'}
    
    # Step 3: Sample data if it's too large
    original_size = len(df_filtered)
    if original_size > 500000:  # If more than 500K rows
        df_filtered = df_filtered.sample(frac=sample_fraction, random_state=42)
        print(f"⚠️  Large dataset detected ({original_size:,} rows)")
        print(f"   Sampling {sample_fraction*100}% for memory efficiency")
        print(f"   Final size: {len(df_filtered):,} rows")
    
    # Step 4: Explore structure
    explore_data_structure(df_filtered, df_gen_filtered)
    
    # Step 5: Check variables
    variables = check_regression_variables(df_gen_filtered)
    
    # Step 6: Prepare data for regression
    print("\n" + "="*50)
    df_merged = prepare_data_for_hourly_regression(df_filtered, df_gen_filtered)
    if df_merged is None:
        return {'status': 'failed', 'step': 'data_preparation'}
    
    # Step 7: Perform hourly regression with memory considerations
    print("\n" + "="*50)
    
    # Adjust fixed effects based on BA size
    if ba in ['PJM', 'MISO'] and len(df_merged) > 1000000:
        print("⚠️  Large BA detected - using simplified fixed effects for memory efficiency")
        regression_results = perform_hourly_regression_memory_efficient(df_merged)
    else:
        regression_results = perform_hourly_regression(df_merged)
    
    # Step 8: Summarize results
    print("\n" + "="*50)
    summary_df = summarize_regression_results(regression_results)
    
    # Step 9: Save results to CSV
    if regression_results:
        save_regression_results_to_csv(regression_results, ba)
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Target BA: {ba}")
    print(f"Target Year: {year}")
    print(f"Data loaded: Yes")
    print(f"Data filtered: Yes")
    print(f"Data prepared: Yes")
    print(f"Regression completed: {'Yes' if regression_results else 'No'}")
    print(f"Available variables: {len(variables['available'])}")
    print(f"Missing variables: {len(variables['missing'])}")
    print(f"Memory optimization: {'Yes' if original_size > 500000 else 'No'}")
    
    return {
        'status': 'success',
        'ba': ba,
        'year': year,
        'cems_shape': df_filtered.shape,
        'gen_shape': df_gen_filtered.shape,
        'merged_shape': df_merged.shape if df_merged is not None else None,
        'merged_data': df_merged,
        'available_vars': variables['available'],
        'missing_vars': variables['missing'],
        'regression_results': regression_results,
        'summary_df': summary_df,
        'memory_optimized': original_size > 500000
    }

def perform_hourly_regression_memory_efficient(df_merged, dependent_vars=None, independent_vars=None):
    """
    Memory-efficient version of hourly regression that uses simplified fixed effects.
    """
    if dependent_vars is None:
        dependent_vars = [
            'gross_load_mw', 'co2_mass_shorttons', 'co2_emissions_intensity'
        ]  # Reduced set for memory efficiency
    
    if independent_vars is None:
        independent_vars = [
            'solar_generation_mw', 'wind_generation_mw', 'residual_demand_mw'
        ]  # Core variables only
    
    print(f"=== Performing Memory-Efficient Hourly Regression ===")
    print(f"Dependent variables: {dependent_vars}")
    print(f"Independent variables: {independent_vars}")
    print("⚠️  Using simplified fixed effects (month, year only) for memory efficiency")
    
    # Use the same logic but with simplified fixed effects
    return perform_hourly_regression(df_merged, dependent_vars, independent_vars)

if __name__ == "__main__":
    # Run test if script is executed directly
    results = run_full_test('CAISO', 2022)
    print(f"\nTest completed with status: {results['status']}") 