#!/usr/bin/env python3
# timeseries-pipeline/api/services/interpretations.py

"""
Module for interpreting statistical results from time series analyses.
Provides human-readable explanations of ARIMA, GARCH, and other statistical models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union


def interpret_correlation(corr_value: float) -> str:
    """
    Interpret the strength and direction of a correlation coefficient.
    
    Args:
        corr_value (float): Correlation coefficient between -1 and 1
        
    Returns:
        str: Human-readable interpretation of the correlation
    """
    if abs(corr_value) < 0.2:
        strength = "very weak"
    elif abs(corr_value) < 0.4:
        strength = "weak"
    elif abs(corr_value) < 0.6:
        strength = "moderate"
    elif abs(corr_value) < 0.8:
        strength = "strong"
    else:
        strength = "very strong"
        
    if corr_value > 0:
        direction = "positive"
    else:
        direction = "negative"
        
    return f"This represents a {strength} {direction} correlation between the two markets."


def interpret_conditional_correlation(
    cc_corr: pd.DataFrame, 
    dcc_corr: pd.Series,
    uncond_corr: pd.DataFrame
) -> Dict[str, str]:
    """
    Interpret conditional correlation results from bivariate GARCH analysis.
    
    Args:
        cc_corr (pd.DataFrame): Constant conditional correlation matrix
        dcc_corr (pd.Series): Dynamic conditional correlation series
        uncond_corr (pd.DataFrame): Unconditional correlation matrix
        
    Returns:
        Dict[str, str]: Dictionary of interpretations for different correlation measures
    """
    # Extract the correlation value from the matrices (off-diagonal element)
    cc_value = cc_corr.iloc[0, 1]
    uncond_value = uncond_corr.iloc[0, 1]
    
    # Get statistics of dynamic correlation
    dcc_mean = dcc_corr.mean()
    dcc_min = dcc_corr.min()
    dcc_max = dcc_corr.max()
    dcc_range = dcc_max - dcc_min
    
    interpretations = {}
    
    # Interpret unconditional correlation
    uncond_interp = (
        f"The unconditional correlation between the markets is {uncond_value:.4f}. "
        f"{interpret_correlation(uncond_value)} This represents the average "
        f"relationship between the markets over the entire period, ignoring time-varying dynamics."
    )
    interpretations["unconditional_correlation"] = uncond_interp
    
    # Interpret constant conditional correlation
    cc_interp = (
        f"The constant conditional correlation is {cc_value:.4f}. "
        f"{interpret_correlation(cc_value)} This captures the relationship "
        f"between the markets after accounting for their individual volatility dynamics."
    )
    interpretations["constant_conditional_correlation"] = cc_interp
    
    # Interpret dynamic conditional correlation
    dcc_interp = (
        f"The dynamic conditional correlation varies over time, with a mean of {dcc_mean:.4f}, "
        f"ranging from {dcc_min:.4f} to {dcc_max:.4f}. "
    )
    
    # Add interpretation based on correlation stability
    if dcc_range > 0.3:
        dcc_interp += (
            f"The wide range ({dcc_range:.4f}) indicates significant instability in the "
            f"relationship between the markets, suggesting periods of both coupling and decoupling."
        )
    elif dcc_range > 0.15:
        dcc_interp += (
            f"The moderate range ({dcc_range:.4f}) indicates some variability in the "
            f"relationship between the markets over time."
        )
    else:
        dcc_interp += (
            f"The narrow range ({dcc_range:.4f}) indicates a relatively stable "
            f"relationship between the markets throughout the analysis period."
        )
    
    interpretations["dynamic_conditional_correlation"] = dcc_interp
    
    # Compare correlation types
    comparison = (
        f"Comparing the correlations: the unconditional correlation is {uncond_value:.4f}, "
        f"while the constant conditional correlation is {cc_value:.4f}, and the dynamic "
        f"conditional correlation averages {dcc_mean:.4f}. "
    )
    
    if abs(cc_value - uncond_value) > 0.1:
        comparison += (
            f"The significant difference between unconditional and conditional correlations "
            f"suggests that accounting for volatility dynamics is important for understanding "
            f"the true relationship between these markets."
        )
    else:
        comparison += (
            f"The similarity between unconditional and conditional correlations suggests "
            f"that the relationship between these markets is relatively consistent, "
            f"even after accounting for volatility dynamics."
        )
    
    interpretations["correlation_comparison"] = comparison
    
    return interpretations


def interpret_portfolio_risk(
    portfolio_volatility: float, 
    annualized_volatility: float,
    weights: np.ndarray,
    asset_names: List[str]
) -> str:
    """
    Interpret portfolio risk based on volatility measures.
    
    Args:
        portfolio_volatility (float): Daily portfolio volatility
        annualized_volatility (float): Annualized portfolio volatility
        weights (np.ndarray): Array of portfolio weights
        asset_names (List[str]): Names of assets in the portfolio
        
    Returns:
        str: Human-readable interpretation of portfolio risk
    """
    # Create a string representing portfolio allocation
    allocation_str = ", ".join([f"{asset_names[i]}: {weights[i]*100:.1f}%" for i in range(len(weights))])
    
    # Interpret the annualized volatility level
    if annualized_volatility < 0.10:
        risk_level = "low"
    elif annualized_volatility < 0.20:
        risk_level = "moderate"
    elif annualized_volatility < 0.30:
        risk_level = "high"
    else:
        risk_level = "very high"
    
    interpretation = (
        f"For a portfolio with allocation {allocation_str}, the daily volatility "
        f"is {portfolio_volatility:.6f} or {portfolio_volatility*100:.2f}%. When annualized, "
        f"this represents a volatility of {annualized_volatility:.6f} or {annualized_volatility*100:.2f}%, "
        f"which is considered {risk_level} risk. "
    )
    
    # Add context about what volatility means
    interpretation += (
        f"This means that, based on historical data, the portfolio's value may fluctuate "
        f"by approximately ±{annualized_volatility*100:.2f}% in a typical year (assuming a normal distribution). "
        f"In more concrete terms, an investment of $10,000 could gain or lose around "
        f"${annualized_volatility*10000:.2f} in a typical year due to market volatility."
    )
    
    return interpretation


def interpret_conditional_volatility(
    cond_vol_df: pd.DataFrame, 
    asset_names: List[str] = None
) -> Dict[str, str]:
    """
    Interpret conditional volatility results from GARCH analysis.
    
    Args:
        cond_vol_df (pd.DataFrame): DataFrame containing conditional volatility series
        asset_names (List[str], optional): Names to use for the assets. Defaults to column names.
        
    Returns:
        Dict[str, str]: Dictionary of interpretations for each asset and overall
    """
    if asset_names is None:
        asset_names = cond_vol_df.columns.tolist()
    
    interpretations = {}
    
    # Overall volatility comparison
    avg_vols = {col: cond_vol_df[col].mean() for col in cond_vol_df.columns}
    most_volatile = max(avg_vols, key=avg_vols.get)
    least_volatile = min(avg_vols, key=avg_vols.get)
    
    overall = (
        f"Comparing the average volatility: {most_volatile} shows higher volatility "
        f"({avg_vols[most_volatile]*100:.2f}% on average) compared to {least_volatile} "
        f"({avg_vols[least_volatile]*100:.2f}% on average). "
    )
    
    if avg_vols[most_volatile] > 2 * avg_vols[least_volatile]:
        overall += f"This indicates that {most_volatile} is significantly more risky than {least_volatile}."
    else:
        overall += f"The two markets show relatively similar risk profiles."
    
    interpretations["overall_comparison"] = overall
    
    # Interpret each asset's volatility characteristics
    for i, col in enumerate(cond_vol_df.columns):
        series = cond_vol_df[col]
        asset_name = asset_names[i] if i < len(asset_names) else col
        
        # Calculate key statistics
        mean_vol = series.mean()
        max_vol = series.max()
        min_vol = series.min()
        vol_range = max_vol - min_vol
        annualized_mean = mean_vol * np.sqrt(252)
        
        # Volatility clustering - check if high volatility periods persist
        autocorr = series.autocorr(lag=1)
        
        asset_interp = (
            f"{asset_name}'s volatility averages {mean_vol*100:.2f}% daily, or {annualized_mean*100:.2f}% "
            f"annualized. Volatility ranges from {min_vol*100:.2f}% to {max_vol*100:.2f}%, "
            f"indicating a {vol_range*100:.2f}% range of risk variation over the period. "
        )
        
        # Add interpretation about volatility clustering
        if autocorr > 0.7:
            asset_interp += (
                f"The high autocorrelation ({autocorr:.2f}) indicates strong volatility clustering, "
                f"where high volatility periods tend to be followed by more high volatility, "
                f"making risk forecasting more predictable."
            )
        elif autocorr > 0.3:
            asset_interp += (
                f"The moderate autocorrelation ({autocorr:.2f}) suggests some volatility clustering, "
                f"where volatile periods show some persistence."
            )
        else:
            asset_interp += (
                f"The low autocorrelation ({autocorr:.2f}) indicates weak volatility clustering, "
                f"suggesting more random transitions between high and low volatility periods."
            )
        
        interpretations[asset_name] = asset_interp
    
    return interpretations


def interpret_arima_results(
    arima_fits: Dict[str, Any], 
    arima_forecasts: Dict[str, Any]
) -> Dict[str, str]:
    """
    Interpret ARIMA model results and forecasts.
    
    Args:
        arima_fits (Dict[str, Any]): Dictionary of fitted ARIMA models
        arima_forecasts (Dict[str, Any]): Dictionary of ARIMA forecasts
        
    Returns:
        Dict[str, str]: Dictionary of interpretations for each series
    """
    interpretations = {}
    
    for column, model in arima_fits.items():
        # Extract parameters and their significance
        params = model.params
        pvalues = model.pvalues
        
        # Start with basic model description
        ar_terms = sum(1 for p in params.index if p.startswith('ar'))
        ma_terms = sum(1 for p in params.index if p.startswith('ma'))
        
        model_desc = (
            f"The ARIMA model for {column} includes {ar_terms} autoregressive terms and "
            f"{ma_terms} moving average terms. "
        )
        
        # Interpret parameter significance
        significant_params = [p for p, v in pvalues.items() if v < 0.05 and p != 'const']
        if significant_params:
            sig_terms = ", ".join(significant_params)
            model_desc += f"Significant terms include {sig_terms}, "
            
            if any(p.startswith('ar') for p in significant_params):
                model_desc += (
                    f"indicating that past values of the series have predictive power "
                    f"for future values. "
                )
            
            if any(p.startswith('ma') for p in significant_params):
                model_desc += (
                    f"The significant moving average terms suggest that past shocks "
                    f"continue to influence the current value. "
                )
        else:
            model_desc += (
                f"No terms are statistically significant at the 5% level, suggesting "
                f"limited predictability in this series. "
            )
        
        # Interpret forecast direction
        forecast_value = arima_forecasts[column]
        last_value = model.data.endog[-1]
        
        if isinstance(forecast_value, (pd.Series, np.ndarray)):
            first_forecast = forecast_value[0]
        else:
            first_forecast = forecast_value
            
        if first_forecast > last_value:
            forecast_desc = (
                f"The model forecasts an increase from the last observed value of {last_value:.4f} "
                f"to {first_forecast:.4f}, a change of {(first_forecast-last_value)*100:.2f}%. "
            )
        elif first_forecast < last_value:
            forecast_desc = (
                f"The model forecasts a decrease from the last observed value of {last_value:.4f} "
                f"to {first_forecast:.4f}, a change of {(last_value-first_forecast)*100:.2f}%. "
            )
        else:
            forecast_desc = (
                f"The model forecasts no change from the last observed value of {last_value:.4f}. "
            )
        
        # Add model fit information
        aic = model.aic
        forecast_desc += (
            f"The model has an AIC of {aic:.2f}, where lower values indicate better fit. "
        )
        
        interpretations[column] = model_desc + forecast_desc
    
    return interpretations


def interpret_stationarity_test(adf_results: Dict[str, Dict[str, float]], 
                               p_value_threshold: float = 0.05) -> Dict[str, str]:
    """
    Interpret Augmented Dickey-Fuller test results for stationarity.
    
    Args:
        adf_results (Dict[str, Dict[str, float]]): Dictionary of ADF test results
        p_value_threshold (float, optional): P-value threshold for significance. Defaults to 0.05.
        
    Returns:
        Dict[str, str]: Dictionary of interpretations for each series
    """
    interpretations = {}
    
    for series_name, result in adf_results.items():
        adf_stat = result["ADF Statistic"]
        p_value = result["p-value"]
        
        if p_value < p_value_threshold:
            interpretation = (
                f"The {series_name} series is stationary (p-value: {p_value:.4f}). "
                f"This means the statistical properties like mean and variance "
                f"remain constant over time, making it suitable for time series modeling. "
                f"The ADF test statistic of {adf_stat:.4f} is below the critical threshold, "
                f"allowing us to reject the null hypothesis of non-stationarity."
            )
        else:
            interpretation = (
                f"The {series_name} series is non-stationary (p-value: {p_value:.4f}). "
                f"This indicates the statistical properties change over time. "
                f"The ADF test statistic of {adf_stat:.4f} is not low enough to reject "
                f"the null hypothesis of non-stationarity. Consider differencing or "
                f"transformation before modeling to achieve stationarity."
            )
            
        interpretations[series_name] = interpretation
        
    return interpretations


def interpret_garch_results(
    garch_fits: Dict[str, Any], 
    garch_forecasts: Dict[str, Any]
) -> Dict[str, str]:
    """
    Interpret GARCH model results and volatility forecasts.
    
    Args:
        garch_fits (Dict[str, Any]): Dictionary of fitted GARCH models
        garch_forecasts (Dict[str, Any]): Dictionary of GARCH forecasts
        
    Returns:
        Dict[str, str]: Dictionary of interpretations for each series
    """
    interpretations = {}
    
    for column, model in garch_fits.items():
        # Extract parameters
        params = model.params
        
        # Get ARCH and GARCH terms
        omega = params.get('omega', params.get('const', None))
        alpha_terms = {k: v for k, v in params.items() if k.startswith('alpha')}
        beta_terms = {k: v for k, v in params.items() if k.startswith('beta')}
        
        alpha_sum = sum(alpha_terms.values())
        beta_sum = sum(beta_terms.values())
        persistence = alpha_sum + beta_sum
        
        # Start with model description
        model_desc = (
            f"The GARCH model for {column} has {len(alpha_terms)} ARCH terms (alpha) "
            f"and {len(beta_terms)} GARCH terms (beta). "
        )
        
        # Interpret persistence
        if persistence < 0.9:
            persistence_desc = (
                f"The volatility persistence is {persistence:.4f}, which is relatively low. "
                f"This suggests that shocks to volatility dissipate quickly, and the series "
                f"returns to its baseline volatility level fairly rapidly."
            )
        elif persistence < 0.99:
            persistence_desc = (
                f"The volatility persistence is {persistence:.4f}, which is moderate to high. "
                f"This indicates that volatility shocks persist for some time before dissipating."
            )
        else:
            persistence_desc = (
                f"The volatility persistence is {persistence:.4f}, which is very high and "
                f"close to 1. This indicates an 'integrated GARCH' pattern where volatility "
                f"shocks have very long-lasting effects and the series has a long memory for "
                f"volatility."
            )
        
        # Interpret relative importance of ARCH vs GARCH terms
        if alpha_sum > beta_sum:
            terms_desc = (
                f"The impact of recent shocks (ARCH, alpha={alpha_sum:.4f}) outweighs "
                f"the impact of past volatility (GARCH, beta={beta_sum:.4f}), "
                f"suggesting that the series reacts strongly to new information."
            )
        else:
            terms_desc = (
                f"The impact of past volatility (GARCH, beta={beta_sum:.4f}) outweighs "
                f"the impact of recent shocks (ARCH, alpha={alpha_sum:.4f}), "
                f"suggesting that volatility has a strong memory component and evolves gradually."
            )
        
        # Interpret forecast trend
        forecast = garch_forecasts[column]
        if hasattr(forecast, '__iter__'):
            first_forecast = forecast[0]
            last_forecast = forecast[-1]
            
            if last_forecast > first_forecast:
                forecast_trend = (
                    f"The volatility forecast shows an increasing trend from {first_forecast:.6f} "
                    f"to {last_forecast:.6f}, suggesting growing uncertainty ahead."
                )
            elif last_forecast < first_forecast:
                forecast_trend = (
                    f"The volatility forecast shows a decreasing trend from {first_forecast:.6f} "
                    f"to {last_forecast:.6f}, suggesting subsiding market turbulence."
                )
            else:
                forecast_trend = (
                    f"The volatility forecast remains stable around {first_forecast:.6f}, "
                    f"suggesting consistent risk levels in the near future."
                )
        else:
            forecast_trend = (
                f"The one-step ahead volatility forecast is {forecast:.6f}, "
                f"which represents the expected level of market fluctuation."
            )
        
        interpretations[column] = f"{model_desc} {persistence_desc} {terms_desc} {forecast_trend}"
    
    return interpretations