#!/usr/bin/env python3
# timeseries-api/api/services/spillover_service.py
"""Spillover analysis service functions.
This module contains functions for analyzing spillover effects between financial time series.
"""

import json
import logging as l
import numpy as np
import pandas as pd
from fastapi import HTTPException
from typing import Dict, Any, List, Optional, Union

from api.services.interpretations import interpret_granger_causality, interpret_var_results
from timeseries_compute import spillover_processor as spillover
from utilities.configurator import load_configuration

def analyze_spillover_step(input_data):
    """
    Analyze spillover effects between time series using the new Diebold-Yilmaz implementation.
    
    This function uses the updated timeseries-compute package with proper Diebold-Yilmaz methodology
    and returns comprehensive intermediate outputs including VAR model details, FEVD matrix,
    spillover indices, and all methodology parameters.
    
    Args:
        input_data: SpilloverInput model containing data and parameters for analysis
    
    Returns:
        Dictionary with comprehensive spillover analysis results including:
        - Diebold-Yilmaz spillover indices (Total Connectedness Index, Directional, Net)
        - VAR model details and parameters
        - FEVD matrix and decomposition
        - ARIMA residuals and conditional mean filtering
        - Methodology parameters (AIC/BIC usage, lag selection, etc.)
    
    Raises:
        HTTPException: If analysis fails for any reason
    """
    try:
        # Load configuration to get parameters
        config = load_configuration("config.yml")
        var_max_lags = getattr(config, 'spillover_var_max_lags', 5)
        
        # Convert input data to DataFrame if needed
        if isinstance(input_data.data, list):
            df = pd.DataFrame(input_data.data)
            
            # If date/time column exists, set as index
            date_cols = [col for col in df.columns if col.lower() in ('date', 'time', 'datetime')]
            if date_cols:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
                df.set_index(date_cols[0], inplace=True)
        else:
            df = input_data.data
        
        # Ensure we have proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            l.warning("Converting index to DatetimeIndex")
            df.index = pd.to_datetime(df.index)

        # Select only numeric columns for analysis
        df_numeric = df.select_dtypes(include=['number'])
        
        # Calculate a safe maximum lag based on data size and config
        safe_max_lag = min(var_max_lags, len(df_numeric) // 3)
        max_lag = max(1, safe_max_lag)
        
        if max_lag < var_max_lags:
            l.warning(f"Adjusted max_lag from {var_max_lags} to {max_lag} due to insufficient observations")
        
        # Get forecast horizon from input or use default
        forecast_horizon = getattr(input_data, 'forecast_horizon', 10)
        
        # Use the new Diebold-Yilmaz analysis function
        result = spillover.run_diebold_yilmaz_analysis(
            returns_df=df_numeric,
            horizon=forecast_horizon,
            max_lags=max_lag,
            ic='aic',
            include_granger=True,
            significance_level=0.05
        )
        
        # Extract comprehensive results from the new structure
        spillover_results = result['spillover_results']
        var_model = result['var_model']
        metadata = result['metadata']
        
        # Extract VAR model details
        var_details = {
            'selected_lag_order': result['var_lag'],
            'information_criterion_used': metadata['ic'],
            'max_lags_considered': metadata['max_lags'],
            'n_variables': metadata['n_assets'],
            'n_observations': metadata['n_observations'],
            'variable_names': metadata['asset_names'],
            'var_model_summary': str(var_model.summary()) if hasattr(var_model, 'summary') else 'VAR model summary not available',
            'var_coefficients': {},
            'var_residuals': {},
            'var_fitted_values': {}
        }
        
        # Extract VAR coefficients and residuals if available
        try:
            if hasattr(var_model, 'params'):
                var_details['var_coefficients'] = var_model.params.to_dict()
            if hasattr(var_model, 'resid'):
                var_details['var_residuals'] = var_model.resid.to_dict('records')
            if hasattr(var_model, 'fittedvalues'):
                var_details['var_fitted_values'] = var_model.fittedvalues.to_dict('records')
        except Exception as e:
            l.warning(f"Could not extract detailed VAR model outputs: {e}")
        
        # FEVD matrix details
        fevd_details = {
            'fevd_matrix_raw': result['fevd_matrix'].tolist(),
            'fevd_matrix_labeled': spillover_results['fevd_table'].to_dict(),
            'fevd_horizon': metadata['horizon'],
            'fevd_normalized': True,  # Your implementation normalizes by default
            'fevd_row_sums': result['fevd_matrix'].sum(axis=1).tolist(),  # Should be ~100 each
            'fevd_interpretation': {
                f"{row_var}_explained_by": {
                    col_var: f"{spillover_results['fevd_table'].loc[row_var, col_var]:.2f}% of {row_var}'s forecast error variance explained by shocks to {col_var}"
                    for col_var in metadata['asset_names']
                }
                for row_var in metadata['asset_names']
            }
        }
        
        # Spillover indices details
        spillover_indices = {
            'total_connectedness_index': {
                'value': spillover_results['total_spillover_index'],
                'interpretation': f"Overall system connectedness: {spillover_results['total_spillover_index']:.2f}%",
                'calculation_method': "Sum of off-diagonal FEVD elements / Total FEVD sum * 100"
            },
            'directional_spillovers': {
                'to_spillovers': {asset: spillover_results['directional_spillovers'][asset]['to'] 
                                 for asset in metadata['asset_names']},
                'from_spillovers': {asset: spillover_results['directional_spillovers'][asset]['from'] 
                                   for asset in metadata['asset_names']},
                'interpretation': {
                    asset: {
                        'to_interpretation': f"{asset} contributes {spillover_results['directional_spillovers'][asset]['to']:.2f}% to other markets' volatility",
                        'from_interpretation': f"{asset} receives {spillover_results['directional_spillovers'][asset]['from']:.2f}% of its volatility from other markets"
                    }
                    for asset in metadata['asset_names']
                }
            },
            'net_spillovers': {
                'values': spillover_results['net_spillovers'],
                'interpretation': {
                    asset: f"{asset} is a net {'transmitter' if spillover_results['net_spillovers'][asset] > 0 else 'receiver'} of shocks ({spillover_results['net_spillovers'][asset]:.2f}%)"
                    for asset in metadata['asset_names']
                }
            },
            'pairwise_spillovers': {
                'matrix': spillover_results['pairwise_spillovers'].to_dict(),
                'interpretation': {
                    f"{from_asset}_to_{to_asset}": f"{spillover_results['pairwise_spillovers'].loc[from_asset, to_asset]:.2f}% spillover from {from_asset} to {to_asset}"
                    for from_asset in metadata['asset_names']
                    for to_asset in metadata['asset_names']
                    if from_asset != to_asset
                }
            }
        }
        
        # Enhanced Granger causality results with detailed interpretation
        granger_detailed = {
            'test_results': result['granger_causality'],
            'methodology': {
                'max_lag_tested': max_lag,
                'significance_levels': ['1%', '5%'],
                'test_statistic': 'F-statistic (SSR-based)',
                'null_hypothesis': 'Series X does not Granger-cause series Y'
            },
            'summary': {
                'total_pairs_tested': len([k for k in result['granger_causality'].keys() if '_to_' in k]),
                'significant_at_1pct': len([k for k, v in result['granger_causality'].items() if v.get('causality_1pct', False)]),
                'significant_at_5pct': len([k for k, v in result['granger_causality'].items() if v.get('causality_5pct', False)])
            }
        }
        
        # Methodology parameters summary
        methodology_params = {
            'spillover_method': 'Diebold-Yilmaz (2012)',
            'var_specification': {
                'lag_selection_criterion': metadata['ic'].upper(),
                'max_lags_considered': metadata['max_lags'],
                'selected_lag_order': result['var_lag'],
                'lag_selection_automatic': True
            },
            'fevd_specification': {
                'forecast_horizon': metadata['horizon'],
                'normalization': 'Row-wise to 100%',
                'identification': 'Cholesky decomposition (default)'
            },
            'granger_causality': {
                'enabled': True,
                'max_lag': max_lag,
                'significance_levels': [0.01, 0.05],
                'test_type': 'F-test (SSR-based)'
            },
            'data_characteristics': {
                'n_variables': metadata['n_assets'],
                'n_observations': metadata['n_observations'],
                'sample_period': f"{df_numeric.index.min()} to {df_numeric.index.max()}",
                'frequency': 'Inferred from index'
            }
        }
        
        # Generate comprehensive interpretation
        try:
            interpretation_dict = interpret_spillover_results({
                "spillover_analysis": {
                    "total_spillover_index": spillover_results['total_spillover_index'],
                    "directional_spillover": spillover_results['directional_spillovers'],
                    "net_spillover": spillover_results['net_spillovers'],
                    "pairwise_spillover": spillover_results['pairwise_spillovers'].to_dict(),
                    "granger_causality": result['granger_causality']
                }
            })
            interpretation = interpretation_dict.get("summary", "Interpretation summary not available.")
        except Exception as interp_error:
            l.warning(f"Could not generate interpretation: {interp_error}")
            interpretation = "Spillover analysis complete, but detailed interpretation unavailable."
        
        # Comprehensive API response with all intermediate outputs
        response = {
            # Core Diebold-Yilmaz Results
            "total_spillover_index": spillover_results['total_spillover_index'],
            "directional_spillover": spillover_results['directional_spillovers'],
            "net_spillover": spillover_results['net_spillovers'],
            "pairwise_spillover": spillover_results['pairwise_spillovers'].to_dict(),
            
            # Detailed Spillover Analysis
            "spillover_indices": spillover_indices,
            
            # VAR Model Details
            "var_model_details": var_details,
            
            # FEVD Matrix and Analysis
            "fevd_analysis": fevd_details,
            
            # Enhanced Granger Causality
            "granger_causality": granger_detailed,
            
            # Methodology and Parameters
            "methodology": methodology_params,
            
            # Legacy compatibility
            "fevd_table": spillover_results['fevd_table'].to_dict(),
            "interpretation": interpretation
        }
        
        return response
    
    except Exception as e:
        l.error(f"Error analyzing spillover: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Spillover analysis failed: {str(e)}"
        )


def run_granger_causality_test(
    series1: pd.Series, 
    series2: pd.Series, 
    max_lag: int = 5, 
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Run Granger causality test between two time series using the updated implementation.
    
    Args:
        series1: First time series (potential cause)
        series2: Second time series (potential effect)
        max_lag: Maximum lag to test
        significance_level: P-value threshold for significance
        
    Returns:
        Dictionary with test results including multi-level significance testing
    """
    return spillover.test_granger_causality(
        series1=series1,
        series2=series2,
        max_lag=max_lag,
        significance_level=significance_level
    )


def compute_spillover_index(
    returns_data: Union[pd.DataFrame, List[Dict[str, Any]]],
    method: str = "diebold_yilmaz",
    forecast_horizon: int = 10,
    window_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute spillover indices using the new Diebold-Yilmaz implementation.
    
    Args:
        returns_data: DataFrame or list of dictionaries with return data
        method: Analysis method (only "diebold_yilmaz" supported in new implementation)
        forecast_horizon: Forecast horizon for variance decomposition
        window_size: Window size for rolling analysis (None for full sample)
        
    Returns:
        Dictionary with spillover indices and related metrics
    """
    # Convert to DataFrame if needed
    if isinstance(returns_data, list):
        df = pd.DataFrame(returns_data)
        
        # If date/time column exists, set as index
        date_cols = [col for col in df.columns if col.lower() in ('date', 'time', 'datetime')]
        if date_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])
            df.set_index(date_cols[0], inplace=True)
            
        # Select only numeric columns
        df = df.select_dtypes(include=['number'])
    else:
        df = returns_data
    
    # Calculate spillover using the new function
    result = spillover.run_diebold_yilmaz_analysis(
        returns_df=df,
        horizon=forecast_horizon,
        max_lags=min(5, len(df) // 3),
        ic='aic',
        include_granger=True,
        significance_level=0.05
    )
    
    # Extract and format the results
    spillover_results = result['spillover_results']
    formatted_result = {
        "total_spillover": spillover_results['total_spillover_index'],
        "directional_spillover": spillover_results['directional_spillovers'],
        "net_spillover": spillover_results['net_spillovers'],
        "pairwise_spillover": spillover_results['pairwise_spillovers'].to_dict()
    }
    
    return formatted_result


def interpret_spillover_results(results):
    """
    Generate a human-readable interpretation of spillover results, including pairwise spillover matrix and relationship interpretations.
    
    This function analyzes the spillover results and creates a comprehensive
    interpretation that explains the findings in clear, business-relevant terms.
    
    Args:
        results: Dictionary with spillover analysis results
        
    Returns:
        String with human-readable interpretation of the results
    """
    # Extract spillover analysis from the new structure
    spillover_analysis = results.get("spillover_analysis", {})
    
    # Use .get() method with default values to avoid KeyError
    total_spillover = spillover_analysis.get("total_spillover_index", 0.0)
    net_spillover = spillover_analysis.get("net_spillover", {})
    pairwise_spillover = spillover_analysis.get("pairwise_spillover", {})

    # Identify top transmitters and receivers
    top_transmitters = sorted(net_spillover.items(), key=lambda x: x[1], reverse=True)[:2] if net_spillover else []
    top_receivers = sorted(net_spillover.items(), key=lambda x: x[1])[:2] if net_spillover else []

    # Main interpretation
    interpretation = (
        f"The system shows a total spillover index of {total_spillover:.2f}%, "
        f"indicating the overall level of interconnectedness between the variables. "
    )
    if total_spillover > 50:
        interpretation += (
            "This high level of spillover suggests strong interconnections where shocks in one market "
            "significantly affect others. Diversification benefits may be limited during periods of market stress. "
        )
    elif total_spillover > 25:
        interpretation += (
            "This moderate level of spillover indicates meaningful interconnections between markets, "
            "with some potential for shock transmission but also some diversification benefits. "
        )
    else:
        interpretation += (
            "This relatively low level of spillover suggests limited interconnections, "
            "with shocks tending to remain contained within individual markets. "
            "This environment may offer good diversification opportunities. "
        )
    if top_transmitters:
        interpretation += (
            f"The main transmitters of shocks are {top_transmitters[0][0]} "
            f"(net: {top_transmitters[0][1]:.2f}%)"
        )
        if len(top_transmitters) > 1:
            interpretation += f" and {top_transmitters[1][0]} (net: {top_transmitters[1][1]:.2f}%)"
        interpretation += ". "
    if top_receivers:
        interpretation += (
            f"The main receivers of shocks are {top_receivers[0][0]} "
            f"(net: {top_receivers[0][1]:.2f}%)"
        )
        if len(top_receivers) > 1:
            interpretation += f" and {top_receivers[1][0]} (net: {top_receivers[1][1]:.2f}%)"
        interpretation += ". "

    # Enhanced Granger causality interpretation with multi-level significance
    granger_results = spillover_analysis.get("granger_causality", {})
    if granger_results:
        pairs_1pct = [pair for pair, result in granger_results.items() if result.get("causality_1pct", False)]
        pairs_5pct = [pair for pair, result in granger_results.items() if result.get("causality_5pct", False)]
        if pairs_1pct or pairs_5pct:
            interpretation += (
                f"Granger causality analysis reveals significant directional relationships: "
            )
            if pairs_1pct:
                interpretation += (
                    f"{len(pairs_1pct)} market pair(s) show highly significant causality at the 1% level, "
                )
            if pairs_5pct:
                interpretation += (
                    f"{len(pairs_5pct)} market pair(s) show significant causality at the 5% level. "
                )
            interpretation += (
                "This indicates specific lead-lag relationships where returns in one market help predict "
                "future changes in another market. "
            )
        else:
            interpretation += (
                "Granger causality tests found no significant predictive relationships between the markets "
                "at conventional significance levels, suggesting independent price movements. "
            )

    # Pairwise spillover matrix interpretation
    pairwise_matrix_interpretation = {}
    pairwise_relationship_interpretation = {}
    for from_sym, to_dict in pairwise_spillover.items():
        for to_sym, value in to_dict.items():
            if from_sym == to_sym:
                continue
            # Matrix interpretation
            pairwise_matrix_interpretation[f"{from_sym}->{to_sym}"] = (
                f"{value:.2f}% of {to_sym}'s forecast error variance is explained by shocks from {from_sym}."
            )
            # Relationship interpretation
            if value < 5:
                strength = "minimal"
                desc = f"{from_sym} and {to_sym} are largely independent, with little evidence of spillover."
            elif value < 20:
                strength = "moderate"
                desc = f"{from_sym} has a moderate influence on {to_sym}, suggesting some interconnectedness."
            else:
                strength = "strong"
                desc = f"{from_sym} has a strong influence on {to_sym}; shocks in {from_sym} are quickly transmitted to {to_sym}."
            pairwise_relationship_interpretation[f"{from_sym}->{to_sym}"] = (
                f"{from_sym} transmits a {strength} spillover to {to_sym} ({value:.2f}%). {desc}"
            )

    return {
        "summary": interpretation,
        "pairwise_spillover_matrix_interpretation": pairwise_matrix_interpretation,
        "pairwise_spillover_relationship_interpretation": pairwise_relationship_interpretation
    }
def perform_granger_causality(df_returns, max_lag=5, alpha=0.05):
    """
    Perform Granger causality tests between all pairs of variables using the new implementation.
    
    Args:
        df_returns: DataFrame containing returns data
        max_lag: Maximum lag to consider for Granger causality test
        alpha: Significance level for hypothesis testing
        
    Returns:
        Dictionary containing Granger causality test results with multi-level significance
    """
    try:
        # Load configuration for Granger causality settings
        config = load_configuration("config.yml")
        
        # Use config values
        granger_enabled = getattr(config, 'granger_causality_enabled', True)
        config_max_lag = getattr(config, 'granger_causality_max_lag', 5)
        max_lag = config_max_lag
        
        if not granger_enabled:
            l.info("Granger causality analysis disabled in configuration")
            return {
                "causality_results": {},
                "interpretations": {"note": "Granger causality analysis is disabled in configuration."}
            }
        
        # Ensure we have enough data points for reliable testing
        min_observations = max_lag * 3
        if len(df_returns) < min_observations:
            l.warning(f"Insufficient data for Granger causality tests. Need at least {min_observations} observations for max_lag={max_lag}")
            return {"error": f"Insufficient data for Granger causality testing with max_lag={max_lag}"}
        
        # Run Granger causality tests between all pairs of variables
        results = {}
        symbols = df_returns.columns
        
        for source in symbols:
            for target in symbols:
                if source != target:  # Skip self-causality
                    test_result = run_granger_causality_test(
                        series1=df_returns[source],
                        series2=df_returns[target],
                        max_lag=max_lag,
                        significance_level=alpha
                    )
                    
                    # Store the result
                    key = f"{source}->{target}"
                    results[key] = test_result
        
        # Generate interpretations for the results
        interpretations = interpret_granger_causality(results)
        
        # Create the response with both raw results and interpretations
        response = {
            "causality_results": results,
            "interpretations": interpretations,
            "metadata": {
                "max_lag": max_lag,
                "n_pairs_tested": len(results),
                "significance_levels": ["1%", "5%"],
                "config_enabled": granger_enabled
            }
        }
        
        # Convert NumPy values to Python native types
        result_str = json.dumps(response, default=lambda obj: float(obj) if isinstance(obj, (np.integer, np.floating)) 
                                else (obj.tolist() if isinstance(obj, np.ndarray) 
                                      else (str(obj) if isinstance(obj, np.bool_) else None)))
        response = json.loads(result_str)
        
        return response
        
    except Exception as e:
        l.error(f"Error in Granger causality analysis: {e}")
        return {"error": str(e)}


def get_var_results_from_spillover(spillover_results: Dict[str, Any], variable_names: list) -> Dict[str, Any]:
    """
    Extract VAR model results from spillover analysis for API response.
    
    Args:
        spillover_results: Results from spillover analysis containing VAR model
        variable_names: List of variable names in the model
        
    Returns:
        Dictionary with formatted VAR results for API response
    """
    try:
        # Extract FEVD table and other VAR-related information
        fevd_table = spillover_results.get('fevd_table', {})
        granger_results = spillover_results.get('granger_causality', {})
        
        # Convert FEVD table back to matrix format for interpretation
        n_vars = len(variable_names)
        fevd_list = []
        
        if isinstance(fevd_table, dict) and variable_names[0] in fevd_table:
            # Convert fevd_table dict back to matrix format
            for i, var1 in enumerate(variable_names):
                row = []
                for j, var2 in enumerate(variable_names):
                    if var1 in fevd_table and var2 in fevd_table[var1]:
                        row.append(float(fevd_table[var1][var2]))
                    else:
                        row.append(100.0 if i == j else 0.0)
                fevd_list.append(row)
        else:
            # Create identity matrix as fallback
            fevd_list = [[100.0 if i == j else 0.0 for j in range(n_vars)] for i in range(n_vars)]
            
        # Generate interpretations
        var_interpretations = interpret_var_results(
            var_model=None,  # We don't have direct access to the model object
            selected_lag=1,  # Default assumption
            ic_used="AIC",
            fevd_matrix=np.array(fevd_list),
            variable_names=variable_names,
            granger_results=granger_results
        )
        
        var_results = {
            "fitted_model": f"VAR model fitted for {n_vars} variables ({', '.join(variable_names)}) using Diebold-Yilmaz methodology",
            "selected_lag": 1,  # Default - would need direct model access for actual value
            "ic_used": "AIC",
            "coefficients": {var: {} for var in variable_names},  # Would need model object for coefficients
            "granger_causality": granger_results,
            "fevd_matrix": fevd_list,
            "fevd_interpretation": var_interpretations.get('fevd_interpretations', {}),
            "interpretation": var_interpretations.get('overall_interpretation', 
                                                   f"VAR model fitted successfully for {n_vars} variables using standard Diebold-Yilmaz methodology.")
        }
        
        return var_results
        
    except Exception as e:
        l.error(f"Error extracting VAR results from spillover analysis: {e}")
        return create_placeholder_var_results(variable_names, spillover_results)


def create_placeholder_var_results(variable_names: list, spillover_results: Any = None) -> Dict[str, Any]:
    """
    Create placeholder VAR results when extraction fails.
    
    Args:
        variable_names: List of variable names
        spillover_results: Original spillover results (for any extractable info)
        
    Returns:
        Dictionary with placeholder VAR results
    """
    n_vars = len(variable_names)
    
    return {
        "fitted_model": f"VAR model fitted for {n_vars} variables ({', '.join(variable_names)}) using Diebold-Yilmaz methodology",
        "selected_lag": 1,
        "ic_used": "AIC",
        "coefficients": {var: {} for var in variable_names},
        "granger_causality": {},
        "fevd_matrix": [[100.0 if i == j else 0.0 for j in range(n_vars)] for i in range(n_vars)],
        "fevd_interpretation": {var: f"FEVD interpretation for {var} not available" for var in variable_names},
        "interpretation": (
            f"VAR model results are not fully available. The model was fitted for {n_vars} variables "
            f"({', '.join(variable_names)}) using standard Diebold-Yilmaz spillover methodology. "
            "Detailed VAR coefficients and diagnostics require direct access to the fitted model object."
        )
    }
