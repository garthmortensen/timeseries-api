#!/usr/bin/env python3
# timeseries-api/api/services/models_service.py
"""Statistical model service functions.
This module contains functions to run statistical models on time series data.
"""

import logging as l
from fastapi import HTTPException
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from timeseries_compute import stats_model

from typing import List, Tuple, Optional, Union, Dict, Any

def run_arima_step(df_stationary: pd.DataFrame, p: int, d: int, q: int,
                  forecast_steps: int) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    Run ARIMA model on stationary time series data with comprehensive intermediate outputs.
    
    This enhanced function returns detailed ARIMA modeling results including:
    - Model summaries and parameters
    - Forecasts and confidence intervals
    - Residuals for conditional mean filtering
    - Model diagnostics and fit statistics
    - Parameter significance tests
    
    Args:
        df_stationary: Stationary time series data
        p: Autoregressive order
        d: Differencing order
        q: Moving average order
        forecast_steps: Number of forecast steps
        
    Returns:
        Tuple containing:
        - Enhanced summaries with detailed model information
        - Comprehensive forecasts with confidence intervals
        - ARIMA residuals DataFrame for GARCH modeling
    """
    try:
        # Ensure Date is set as index before passing to the library
        if 'Date' in df_stationary.columns:
            df_stationary = df_stationary.set_index('Date')
            
        # Run ARIMA models with the explicit parameters
        arima_fits, arima_forecasts = stats_model.run_arima(
            df_stationary=df_stationary,
            p=p,
            d=d,
            q=q,
            forecast_steps=forecast_steps
        )
        
        # Extract comprehensive ARIMA results
        arima_residuals = pd.DataFrame(index=df_stationary.index)
        all_summaries = {}
        all_forecasts = {}
        
        for symbol in arima_fits.keys():
            fitted_model = arima_fits[symbol]
            
            # Extract residuals for conditional mean filtering
            arima_residuals[symbol] = fitted_model.resid
            
            # Enhanced model summary with detailed information
            model_details = {
                # Basic model information
                'model_specification': f"ARIMA({p},{d},{q})",
                'sample_size': len(fitted_model.resid),
                'log_likelihood': float(fitted_model.llf),
                'aic': float(fitted_model.aic),
                'bic': float(fitted_model.bic),
                'hqic': float(fitted_model.hqic),
                
                # Parameter estimates and significance
                'parameters': {},
                'parameter_pvalues': {},
                'parameter_significance': {},
                
                # Model diagnostics
                'residual_statistics': {},
                'fitted_values': fitted_model.fittedvalues.to_dict(),
                'residuals': fitted_model.resid.to_dict(),
                
                # Conditional mean filtering information
                'conditional_mean_filtering': {
                    'purpose': 'Remove predictable component from returns',
                    'filtered_series': fitted_model.resid.to_dict(),
                    'mean_component_removed': (df_stationary[symbol] - fitted_model.resid).to_dict(),
                    'residual_properties': {
                        'mean': float(fitted_model.resid.mean()),
                        'std': float(fitted_model.resid.std()),
                        'skewness': float(fitted_model.resid.skew()),
                        'kurtosis': float(fitted_model.resid.kurtosis())
                    }
                },
                
                # Enhanced forecasting information
                'forecasting': {
                    'method': 'Maximum Likelihood Estimation',
                    'forecast_steps': forecast_steps,
                    'point_forecasts': process_forecast_values(arima_forecasts[symbol]),
                },
                
                # Full model summary for reference
                'full_summary': str(fitted_model.summary())
            }
            
            # Extract parameter estimates and p-values
            try:
                if hasattr(fitted_model, 'params') and hasattr(fitted_model, 'pvalues'):
                    for param_name, param_value in fitted_model.params.items():
                        model_details['parameters'][param_name] = float(param_value)
                        if param_name in fitted_model.pvalues:
                            p_value = float(fitted_model.pvalues[param_name])
                            model_details['parameter_pvalues'][param_name] = p_value
                            # Determine significance level
                            if p_value < 0.01:
                                significance = "Significant at 1% level"
                            elif p_value < 0.05:
                                significance = "Significant at 5% level"
                            elif p_value < 0.10:
                                significance = "Significant at 10% level"
                            else:
                                significance = "Not significant"
                            model_details['parameter_significance'][param_name] = significance
            except Exception as e:
                l.warning(f"Could not extract parameter details for {symbol}: {e}")
            
            # Add confidence intervals for forecasts if available
            try:
                forecast_result = fitted_model.get_forecast(steps=forecast_steps)
                conf_int = forecast_result.conf_int()
                model_details['forecasting']['confidence_intervals'] = {
                    'lower_bound': conf_int.iloc[:, 0].tolist(),
                    'upper_bound': conf_int.iloc[:, 1].tolist(),
                    'confidence_level': '95%'
                }
                model_details['forecasting']['forecast_standard_errors'] = forecast_result.se.tolist()
            except Exception as e:
                l.warning(f"Could not extract forecast confidence intervals for {symbol}: {e}")
            
            # Calculate additional residual diagnostics
            try:
                residuals = fitted_model.resid
                model_details['residual_statistics'] = {
                    'ljung_box_test': 'Available in full summary',
                    'jarque_bera_test': 'Available in full summary', 
                    'mean': float(residuals.mean()),
                    'variance': float(residuals.var()),
                    'min': float(residuals.min()),
                    'max': float(residuals.max()),
                    'autocorrelation_lag1': float(residuals.autocorr(lag=1)) if len(residuals) > 1 else 0.0
                }
            except Exception as e:
                l.warning(f"Could not calculate residual statistics for {symbol}: {e}")
            
            # Store enhanced summary
            all_summaries[symbol] = model_details
            
            # Store enhanced forecasts
            forecast_values = arima_forecasts[symbol]
            all_forecasts[symbol] = {
                'point_forecasts': process_forecast_values(forecast_values),
                'forecast_steps': forecast_steps,
                'model_specification': f"ARIMA({p},{d},{q})",
                'forecast_method': 'Dynamic forecasting with MLE parameters'
            }
        
        return all_summaries, all_forecasts, arima_residuals
        
    except Exception as e:
        l.error(f"Error running ARIMA model: {e}")
        raise Exception(f"Error running ARIMA model: {str(e)}")

def run_garch_step(df_residuals: pd.DataFrame, p: int, q: int, dist: str,
                  forecast_steps: int) -> Tuple[Dict[str, str], Dict[str, List[float]], Optional[pd.DataFrame], Dict[str, Any]]:
    """Run GARCH model on ARIMA residuals.
    
    BEST PRACTICE: Academic research demonstrates that GARCH models effectively capture 
    volatility clustering in financial time series. The t-distribution option is particularly 
    valuable as financial returns typically exhibit fat tails that normal distributions 
    cannot adequately model. The configurable p,q parameters allow for model customization
    while defaulting to the parsimonious GARCH(1,1) specification that research shows 
    performs well in most applications.
    
    Returns:
        Tuple containing:
        - Dict[str, str]: GARCH model summaries
        - Dict[str, List[float]]: GARCH forecasts
        - Optional[pd.DataFrame]: Conditional volatilities
        - Dict[str, Any]: Fitted GARCH models (NEW - for multivariate GARCH)
    """
    try:
        # Ensure Date is set as index before passing to the library
        if 'Date' in df_residuals.columns:
            df_residuals = df_residuals.set_index('Date')
            
        # Run GARCH models with explicit parameters
        garch_fits, garch_forecasts = stats_model.run_garch(
            df_stationary=df_residuals,
            p=p,
            q=q,
            dist=dist,
            forecast_steps=forecast_steps
        )

        # Extract conditional volatilities
        cond_vol = pd.DataFrame(index=df_residuals.index)
        for column in df_residuals.columns:
            cond_vol[column] = np.sqrt(garch_fits[column].conditional_volatility)
        
        # Process results for all symbols
        all_summaries = {}
        all_forecasts = {}
        
        for symbol in garch_fits.keys():
            all_summaries[symbol] = str(garch_fits[symbol].summary())
            forecast_values = garch_forecasts[symbol]
            
            # Convert variance forecasts to volatility
            if hasattr(forecast_values, '__iter__'):
                all_forecasts[symbol] = [float(np.sqrt(x)) for x in forecast_values]
            else:
                all_forecasts[symbol] = [float(np.sqrt(forecast_values))]
        
        # Return fitted models as well for multivariate GARCH analysis
        return all_summaries, all_forecasts, cond_vol, garch_fits
        
    except Exception as e:
        l.error(f"Error running GARCH model: {e}")
        raise Exception(f"Error running GARCH model: {str(e)}")

def process_forecast_values(forecast_values: Union[float, np.ndarray, pd.Series, List]) -> List[float]:
    """Convert forecast values to a consistent list format."""
    if np.isscalar(forecast_values):
        return [float(forecast_values)]
    elif isinstance(forecast_values, pd.Series):
        return [float(x) for x in forecast_values.values]
    elif isinstance(forecast_values, np.ndarray):
        return [float(x) for x in forecast_values]
    elif isinstance(forecast_values, list):
        return [float(x) for x in forecast_values]
    else:
        try:
            # Handle other iterable types
            return [float(x) for x in forecast_values]
        except (TypeError, ValueError):
            l.warning(f"Could not convert forecast values of type {type(forecast_values)}: {forecast_values}")
            # Return as single value if it's numeric, otherwise empty list
            try:
                return [float(forecast_values)]
            except:
                return []
