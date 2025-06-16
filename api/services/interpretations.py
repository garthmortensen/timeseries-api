#!/usr/bin/env python3
# timeseries-api/api/services/interpretations.py
"""
Interpretation module for statistical test results.
Contains functions to create human-readable interpretations of statistical test results.
"""

import logging as l
import numpy as np
from typing import Dict, Any


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
        try:
            adf_stat = result["ADF Statistic"]
            p_value = result["p-value"]
            critical_1 = result.get("Critical Values", {}).get("1%", None)
            critical_5 = result.get("Critical Values", {}).get("5%", None)
            
            # Start with detailed p-value explanation
            p_value_percent = p_value * 100
            
            if p_value < p_value_threshold:
                # Determine which critical value thresholds are passed
                passes_1_percent = critical_1 is not None and adf_stat < critical_1
                passes_5_percent = critical_5 is not None and adf_stat < critical_5
                
                # Stationary case - but be precise about confidence levels
                interpretation = (
                    f"The Augmented Dickey-Fuller (ADF) test on the {series_name} series gives a p-value of {p_value:.4f}. "
                    f"That number tells us there's only about a {p_value_percent:.2f}% chance of seeing a test statistic this extreme "
                    f"if the series really were non-stationary (i.e., wandering randomly or drifting). "
                )
                
                # Be precise about which significance levels we can reject at
                if passes_1_percent:
                    interpretation += (
                        f"Because that probability is so low—well below both 5% and 1% cutoffs—we can confidently reject "
                        f"the 'unit root' (non-stationary) hypothesis with high confidence (99% level).\n\n"
                    )
                elif passes_5_percent:
                    interpretation += (
                        f"Because that probability is below the 5% cutoff, we can reject the 'unit root' (non-stationary) hypothesis "
                        f"with moderate confidence (95% level), though not quite at the 1% level.\n\n"
                    )
                else:
                    # This shouldn't happen if p-value < 0.05 but critical values don't support it
                    interpretation += (
                        f"While the p-value suggests significance, the test statistic doesn't exceed the standard critical value thresholds, "
                        f"creating some ambiguity in the conclusion.\n\n"
                    )
                
                # Add critical value context
                interpretation += f"The test statistic itself, {adf_stat:.4f}, "
                
                if passes_1_percent:
                    interpretation += f"exceeds both the 5% critical value ({critical_5:.4f}) and the stricter 1% critical value ({critical_1:.4f})"
                elif passes_5_percent:
                    interpretation += f"exceeds the 5% critical value ({critical_5:.4f}) but falls short of the 1% critical value ({critical_1:.4f})"
                elif critical_5 is not None:
                    interpretation += f"doesn't reach the 5% critical value threshold of {critical_5:.4f}"
                
                interpretation += (
                    f". Put simply, the data's mean and variance aren't shifting around over time; "
                    f"they stick close to their long-term levels.\n\n"
                    
                    f"In practical terms for modeling, this stability means you don't have to first difference or otherwise "
                    f"transform the series to get rid of trends. It behaves like a mean-reverting process: shocks may sway it "
                    f"up or down, but it tends to pull back toward its average. That behavior lines up with the efficient market "
                    f"idea that prices jitter around a fair value without drifting in predictable ways—and makes the {series_name} "
                    f"a solid candidate for models that assume constant statistical properties over time."
                )
            else:
                # Non-stationary case
                interpretation = (
                    f"The Augmented Dickey-Fuller (ADF) test on the {series_name} series gives a p-value of {p_value:.4f}. "
                    f"That number tells us there's about a {p_value_percent:.2f}% chance of seeing a test statistic this extreme "
                    f"even if the series really were stationary. Because that probability is above common significance thresholds "
                    f"(like 5% or 1%), we cannot reject the 'unit root' (non-stationary) hypothesis with confidence.\n\n"
                    
                    f"The test statistic of {adf_stat:.4f} doesn't reach the critical threshold needed to declare stationarity"
                )
                
                # Add critical value context if available
                if critical_5 is not None:
                    interpretation += f" (it would need to be more negative than {critical_5:.4f} at 5% significance)"
                
                interpretation += (
                    f". This suggests the series exhibits changing statistical properties over time—perhaps a wandering mean, "
                    f"evolving variance, or persistent trends.\n\n"
                    
                    f"In practical modeling terms, this non-stationarity means you'll likely need to transform the data before "
                    f"applying standard time series models. Common approaches include taking first differences (converting prices "
                    f"to returns) or applying other transformations to remove trends and achieve stability. The current behavior "
                    f"suggests that shocks to the series tend to have permanent effects rather than temporary ones, creating "
                    f"persistent deviations from any long-term average. This is typical of many financial price series where "
                    f"market movements can establish new price levels that persist over time."
                )
                
            interpretations[series_name] = interpretation
        except KeyError as e:
            l.warning(f"Missing key in ADF results for {series_name}: {e}")
            interpretations[series_name] = f"Unable to interpret results for {series_name} due to missing data."
        except Exception as e:
            l.error(f"Error interpreting stationarity for {series_name}: {e}")
            interpretations[series_name] = f"Error interpreting results for {series_name}."
            
    return interpretations


def interpret_arima_results(model_summary: str, forecast: list) -> str:
    """
    Create a human-readable interpretation of ARIMA model results.
    
    Args:
        model_summary (str): Summary of the fitted ARIMA model
        forecast (list): List of forecasted values
        
    Returns:
        str: Human-readable interpretation of the ARIMA model results
    """
    try:
        # Extract simple trend from forecast
        if len(forecast) > 1:
            if forecast[-1] > forecast[0]:
                trend = "an increasing"
                plain_trend = "upward"
                implication = "suggesting future values are likely to be higher than current ones"
            elif forecast[-1] < forecast[0]:
                trend = "a decreasing"
                plain_trend = "downward"
                implication = "suggesting future values are likely to be lower than current ones"
            else:
                trend = "a stable"
                plain_trend = "flat"
                implication = "suggesting future values are likely to remain similar to current ones"
        else:
            trend = "an unknown"
            plain_trend = "unclear"
            implication = ""
            
        interpretation = (
            f"This model incorporates both autoregressive components (past values) and moving averages (past errors) "
            f"to generate forecasts, similar to technical analysis methods that consider recent price movements "
            f"and error corrections to predict future market behavior. "
            f"The ARIMA model has been fitted successfully. "
            f"The forecast shows {trend} trend over the forecast horizon. "
            f"In market analysis terms, the data is projected to follow a {plain_trend} trajectory {implication}."
        )
        
        return interpretation
    except Exception as e:
        l.error(f"Error interpreting ARIMA results: {e}")
        return "Unable to provide a detailed interpretation of the ARIMA model."


def interpret_garch_results(model_summary: str, forecast: list) -> str:
    """
    Create a human-readable interpretation of GARCH model results.
    
    Args:
        model_summary (str): Summary of the fitted GARCH model
        forecast (list): List of forecasted volatility values
        
    Returns:
        str: Human-readable interpretation of the GARCH model results
    """
    try:
        # Extract simple trend from forecast
        if len(forecast) > 1:
            if forecast[-1] > forecast[0]:
                trend = "an increasing"
                implication = "suggesting growing market uncertainty"
                plain_desc = "becoming more volatile with expanding price ranges"
                market_impact = "potentially requiring wider stop-loss orders and indicating heightened market risk"
                vix_context = "This pattern often coincides with rising VIX levels, reflecting increased investor fear and uncertainty in the broader market."
            elif forecast[-1] < forecast[0]:
                trend = "a decreasing"
                implication = "suggesting decreasing market uncertainty"
                plain_desc = "becoming more stable with narrower price ranges"
                market_impact = "indicating a calming market environment with reduced risk premiums"
                vix_context = "This trend typically aligns with declining VIX levels, suggesting increased investor confidence and market stability."
            else:
                trend = "a stable"
                implication = "suggesting stable market conditions"
                plain_desc = "maintaining its current volatility level"
                market_impact = "indicating consistent market risk levels in the forecast period"
                vix_context = "This stable pattern reflects steady market conditions, similar to when VIX remains within normal trading ranges."
        else:
            trend = "an unknown"
            implication = ""
            plain_desc = "showing unclear volatility patterns"
            market_impact = ""
            vix_context = ""
            
        interpretation = (
            f"The GARCH model has been fitted successfully, capturing the time-varying nature of volatility. "
            f"The volatility forecast shows {trend} trend {implication}. "
            f"In financial market terms, the asset prices are expected to be {plain_desc}, {market_impact}. "
            f"Volatility clustering—periods of high volatility followed by high volatility and low volatility periods followed by low volatility—is a key feature that GARCH models capture effectively. "
            f"{vix_context} "
            f"Understanding these volatility patterns is crucial for risk management, option pricing, and portfolio allocation decisions."
        )
        
        return interpretation
    except Exception as e:
        l.error(f"Error interpreting GARCH results: {e}")
        return "Unable to provide a detailed interpretation of the GARCH model."


def interpret_spillover_index(spillover_results: Dict[str, float],
                             threshold: float = 0.5) -> Dict[str, str]:
    """
    Interpret Diebold-Yilmaz spillover index results.
    
    Args:
        spillover_results (Dict[str, float]): Dictionary of spillover index results
        threshold (float, optional): Threshold for significant spillover. Defaults to 0.5.
        
    Returns:
        Dict[str, str]: Dictionary of interpretations for spillover metrics
    """
    interpretations = {}
    
    try:
        total_spillover = spillover_results.get("total", 0.0)
        
        if total_spillover > threshold:
            interpretation = (
                f"The total spillover index is {total_spillover:.4f}, indicating strong "
                f"interconnectedness between the time series. This suggests significant "
                f"transmission of shocks across markets or variables, making diversification "
                f"less effective during periods of high volatility. "
                f"In market terminology, this high interconnection indicates potential contagion effects "
                f"across assets or sectors. This resembles the 2008 financial crisis scenario where problems "
                f"in one sector (mortgage-backed securities) quickly spread throughout the financial system."
            )
        else:
            interpretation = (
                f"The total spillover index is {total_spillover:.4f}, indicating limited "
                f"interconnectedness between the time series. This suggests relatively "
                f"independent behavior of markets or variables, potentially allowing for "
                f"effective diversification strategies. "
                f"In portfolio management terms, this lower interconnection indicates better diversification potential. "
                f"The markets or assets appear to respond to their own specific factors rather than common drivers, "
                f"creating opportunities for risk reduction through asset allocation across these markets."
            )
            
        interpretations["Total Spillover"] = interpretation
        
        # Interpret directional spillovers if available
        for key, value in spillover_results.items():
            if key.startswith("from_") or key.startswith("to_"):
                direction = "receiving" if key.startswith("to_") else "transmitting"
                series_name = key.split("_")[1]
                
                if value > threshold:
                    dir_interpretation = (
                        f"The series {series_name} is {direction} substantial spillovers "
                        f"({value:.4f}), suggesting it is a {direction == 'transmitting' and 'source' or 'recipient'} "
                        f"of volatility in the system. "
                        f"In financial markets, this means {series_name} {direction == 'transmitting' and 'acts as a leading indicator that influences broader market movements' or 'functions as a lagging indicator that responds significantly to external market forces'}. "
                        f"This is characteristic of {direction == 'transmitting' and 'major indices or large-cap stocks that often drive sector movements' or 'highly responsive sectors like small-caps that tend to amplify broader market trends'}."
                    )
                else:
                    dir_interpretation = (
                        f"The series {series_name} is {direction} limited spillovers "
                        f"({value:.4f}), suggesting it is relatively isolated in terms of "
                        f"{direction == 'transmitting' and 'influencing' or 'being influenced by'} other series. "
                        f"In portfolio construction terms, {series_name} {direction == 'transmitting' and 'has minimal impact on other assets, making it less useful as a hedging instrument' or 'shows relative independence from market movements, potentially offering diversification benefits'}. "
                        f"This behavior is typical of {direction == 'transmitting' and 'niche market segments with specific drivers' or 'defensive assets that maintain stability regardless of broader market conditions'}."
                    )
                
                interpretations[key] = dir_interpretation
                
        return interpretations
    except Exception as e:
        l.error(f"Error interpreting spillover index: {e}")
        return {"error": "Unable to interpret spillover index results due to an error."}


def interpret_granger_causality(causality_results: Dict[str, Dict[str, float]],
                               p_value_threshold: float = 0.05) -> Dict[str, str]:
    """
    Interpret Granger causality test results between time series.
    
    Args:
        causality_results (Dict[str, Dict[str, float]]): Dictionary of Granger causality test results
        p_value_threshold (float, optional): P-value threshold for significance (kept for backward compatibility). Defaults to 0.05.
        
    Returns:
        Dict[str, str]: Dictionary of causality interpretations with multi-level significance
    """
    interpretations = {}
    
    try:
        for pair, result in causality_results.items():
            source, target = pair.split("->")
            
            # Extract multi-level significance results
            causality_1pct = result.get("causality_1pct", False)
            causality_5pct = result.get("causality_5pct", False)
            min_p_value = result.get("significance_summary", {}).get("min_p_value", 1.0)
            optimal_lag_1pct = result.get("optimal_lag_1pct")
            optimal_lag_5pct = result.get("optimal_lag_5pct")
            
            # Create interpretation based on multi-level significance
            if causality_1pct:
                interpretation = (
                    f"⭐ **Highly Significant Causality (1% level)**: {source} strongly Granger-causes {target} "
                    f"(optimal lag: {optimal_lag_1pct}, min p-value: {min_p_value:.4f}). "
                    f"This indicates a very robust predictive relationship where past values of {source} "
                    f"contain significant information for predicting future values of {target}. "
                    f"In financial analysis, this represents a strong leading indicator relationship that "
                    f"could be valuable for trading strategies and risk management."
                )
            elif causality_5pct:
                interpretation = (
                    f"✓ **Significant Causality (5% level)**: {source} Granger-causes {target} "
                    f"(optimal lag: {optimal_lag_5pct}, min p-value: {min_p_value:.4f}). "
                    f"This suggests that past values of {source} contain information that helps predict "
                    f"future values of {target}, beyond what is contained in past values of {target} alone. "
                    f"In market terms, movements in {source} statistically precede movements in {target}, "
                    f"creating a potential leading indicator relationship."
                )
            else:
                interpretation = (
                    f"✗ **No Significant Causality**: {source} does not Granger-cause {target} "
                    f"(min p-value: {min_p_value:.4f}). "
                    f"This suggests that past values of {source} do not contain significant "
                    f"additional information for predicting future values of {target}. "
                    f"In market efficiency terms, this supports the random walk hypothesis for these assets' relationship, "
                    f"suggesting that {target}'s price movements cannot be predicted using {source}'s historical data."
                )
                
            interpretations[pair] = interpretation
            
        return interpretations
    except Exception as e:
        l.error(f"Error interpreting Granger causality: {e}")
        return {"error": "Unable to interpret Granger causality results due to an error."}


def interpret_spillover_results(spillover_data: Dict[str, Any], 
                               significance_threshold: float = 0.1) -> Dict[str, str]:
    """
    Create human-readable interpretations of spillover analysis results.
    
    Args:
        spillover_data (Dict[str, Any]): Dictionary containing spillover analysis results
        significance_threshold (float, optional): Threshold for significant spillovers. Defaults to 0.1.
        
    Returns:
        Dict[str, str]: Dictionary of interpretations for spillover results
    """
    interpretations = {}
    
    try:
        # Total spillover interpretation
        if "total_spillover" in spillover_data:
            total = spillover_data["total_spillover"]
            
            if total > 0.5:
                strength = "strong"
                implication = "high level of interconnectedness"
            elif total > 0.3:
                strength = "moderate"
                implication = "meaningful level of interconnectedness"
            else:
                strength = "weak"
                implication = "low level of interconnectedness"
                
            interpretation = (
                f"The total spillover index is {total:.4f}, indicating a {strength} {implication} "
                f"between markets or variables in the system. "
                f"In simple terms, this means {'changes in one market or variable strongly affect other markets or variables' if total > 0.5 else 'changes in one market or variable have some effect on other markets or variables' if total > 0.3 else 'changes in one market or variable have limited effects on other markets or variables'}. "
                f"{'This suggests that diversification benefits might be limited during turbulent periods as shocks spread widely across the system.' if total > 0.5 else 'This suggests some diversification benefits might exist, but certain shocks could still spread across parts of the system.' if total > 0.3 else 'This suggests good diversification opportunities as shocks tend to remain contained within specific markets or variables.'}"
            )
            interpretations["Total Spillover"] = interpretation
        
        # Directional spillovers
        if "directional" in spillover_data:
            for direction, values in spillover_data["directional"].items():
                for entity, value in values.items():
                    if direction == "to":
                        if value > significance_threshold:
                            dir_interp = (
                                f"{entity} receives significant spillovers from other variables (value: {value:.4f}). "
                                f"This means {entity} is vulnerable to shocks originating elsewhere in the system. "
                                f"In market analysis terms, this asset or market segment demonstrates sensitivity to external factors "
                                f"and broader market movements. Such assets typically offer limited diversification benefits during market stress."
                            )
                        else:
                            dir_interp = (
                                f"{entity} receives minimal spillovers from other variables (value: {value:.4f}). "
                                f"This means {entity} is relatively insulated from shocks originating elsewhere. "
                                f"In portfolio construction terms, this market or asset demonstrates independence from external factors, "
                                f"making it potentially valuable for diversification. Its returns are more likely driven by idiosyncratic factors."
                            )
                    else:  # "from"
                        if value > significance_threshold:
                            dir_interp = (
                                f"{entity} transmits significant spillovers to other variables (value: {value:.4f}). "
                                f"This means shocks to {entity} spread widely through the system. "
                                f"In plain language, changes in this market or asset tend to affect other markets or assets noticeably. "
                                f"For example, when this market experiences volatility, other related markets often follow suit."
                            )
                        else:
                            dir_interp = (
                                f"{entity} transmits minimal spillovers to other variables (value: {value:.4f}). "
                                f"This means shocks to {entity} remain largely contained. "
                                f"In simple terms, changes in this market or asset have limited impact on other markets or assets. "
                                f"This suggests that volatility in this market is unlikely to trigger widespread market reactions."
                            )
                    
                    interpretations[f"{direction.capitalize()} {entity}"] = dir_interp
        
        # Net spillovers
        if "net" in spillover_data:
            for entity, value in spillover_data["net"].items():
                if abs(value) < significance_threshold:
                    net_interp = (
                        f"{entity} has a balanced spillover profile (net: {value:.4f}), "
                        f"meaning it transmits and receives spillovers in roughly equal measure. "
                        f"In simple terms, this market or asset influences other markets about as much as it is influenced by them. "
                        f"This balanced relationship suggests the asset plays both leading and following roles in the market system."
                    )
                elif value > 0:
                    net_interp = (
                        f"{entity} is a net transmitter of spillovers (net: {value:.4f}), "
                        f"meaning it influences other variables more than it is influenced by them. "
                        f"In plain English, this market or asset tends to lead market movements rather than follow them. "
                        f"Traders often watch such assets as potential leading indicators for broader market movements."
                    )
                else:
                    net_interp = (
                        f"{entity} is a net receiver of spillovers (net: {value:.4f}), "
                        f"meaning it is influenced by other variables more than it influences them. "
                        f"In simple terms, this market or asset tends to follow market movements rather than lead them. "
                        f"Traders often view such assets as lagging indicators that react to broader market trends."
                    )
                
                interpretations[f"Net {entity}"] = net_interp
                
        return interpretations
    except Exception as e:
        l.error(f"Error interpreting spillover results: {e}")
        return {"error": "Unable to interpret spillover results due to an error."}


def interpret_var_results(var_model: Any, selected_lag: int, ic_used: str, 
                         fevd_matrix: np.ndarray, variable_names: list,
                         granger_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create comprehensive human-readable interpretation of VAR model results.
    
    Args:
        var_model: Fitted VAR model from statsmodels
        selected_lag: Selected lag order
        ic_used: Information criterion used for selection
        fevd_matrix: FEVD matrix from spillover analysis
        variable_names: Names of variables in the model
        granger_results: Granger causality test results
        
    Returns:
        Dictionary with VAR interpretations including FEVD and overall model interpretation
    """
    try:
        # 1. Overall VAR Model Interpretation
        n_vars = len(variable_names)
        total_params = selected_lag * n_vars * n_vars + n_vars  # coefficients + intercepts
        
        overall_interpretation = (
            f"The Vector Autoregression (VAR) model has been successfully fitted with {selected_lag} lag(s) "
            f"using {ic_used.upper()} criterion for optimal lag selection. This multivariate model captures "
            f"the dynamic relationships between {n_vars} financial series ({', '.join(variable_names)}), "
            f"with a total of {total_params} parameters estimated. "
        )
        
        if selected_lag == 1:
            overall_interpretation += (
                "The single lag structure suggests that each variable's current value depends primarily on "
                "its own and other variables' values from the previous period, indicating relatively "
                "short-term interdependencies in the financial system."
            )
        elif selected_lag <= 3:
            overall_interpretation += (
                f"The {selected_lag}-lag structure indicates that past values from up to {selected_lag} periods "
                "influence current returns, suggesting moderate persistence in cross-market relationships "
                "and information transmission patterns."
            )
        else:
            overall_interpretation += (
                f"The {selected_lag}-lag structure reveals longer memory effects where market movements "
                "from up to {selected_lag} periods ago still influence current behavior, indicating "
                "persistent interdependencies and potentially complex market dynamics."
            )
        
        # 2. FEVD Interpretations for each variable
        fevd_interpretations = {}
        
        for i, var_name in enumerate(variable_names):
            # Own shock contribution (diagonal element)
            own_contribution = fevd_matrix[i, i]
            # External shock contributions (off-diagonal elements)
            external_contributions = [(variable_names[j], fevd_matrix[i, j]) 
                                    for j in range(n_vars) if j != i]
            external_contributions.sort(key=lambda x: x[1], reverse=True)
            
            interpretation = (
                f"For {var_name}, forecast errors are explained as follows: "
                f"{own_contribution:.1f}% comes from its own innovations (idiosyncratic shocks), "
            )
            
            if external_contributions:
                main_external = external_contributions[0]
                interpretation += (
                    f"while {100 - own_contribution:.1f}% comes from external sources. "
                    f"The largest external influence is from {main_external[0]} "
                    f"({main_external[1]:.1f}%), "
                )
                
                if len(external_contributions) > 1:
                    secondary_external = external_contributions[1]
                    interpretation += f"followed by {secondary_external[0]} ({secondary_external[1]:.1f}%). "
                
                # Interpretation based on own vs external contributions
                if own_contribution > 70:
                    interpretation += (
                        "This indicates that the asset is relatively autonomous, with most forecast "
                        "uncertainty stemming from asset-specific factors rather than spillovers from other markets."
                    )
                elif own_contribution > 50:
                    interpretation += (
                        "This shows a balanced influence where the asset has moderate independence but "
                        "is also meaningfully affected by external market forces."
                    )
                else:
                    interpretation += (
                        "This reveals high interconnectedness where external market forces dominate "
                        "forecast uncertainty, making the asset highly sensitive to spillover effects."
                    )
            
            fevd_interpretations[var_name] = interpretation
        
        # 3. Granger Causality Summary
        granger_summary = ""
        if granger_results:
            # Count significant relationships at different levels
            pairs_1pct = sum(1 for result in granger_results.values() 
                           if result.get("causality_1pct", False))
            pairs_5pct = sum(1 for result in granger_results.values() 
                           if result.get("causality_5pct", False))
            total_pairs = len(granger_results)
            
            granger_summary = (
                f"Granger causality analysis of {total_pairs} directional relationships reveals: "
            )
            
            if pairs_1pct > 0:
                granger_summary += f"{pairs_1pct} highly significant causal relationships (1% level), "
            if pairs_5pct > 0:
                granger_summary += f"{pairs_5pct} significant causal relationships (5% level). "
            
            if pairs_5pct == 0:
                granger_summary += (
                    "No significant predictive relationships were found, suggesting that past values "
                    "of these variables do not help predict each other beyond what each variable's "
                    "own history provides."
                )
            else:
                granger_summary += (
                    "These relationships indicate specific lead-lag patterns where some markets "
                    "systematically influence others' future movements."
                )
        
        return {
            'overall_interpretation': overall_interpretation + " " + granger_summary,
            'fevd_interpretations': fevd_interpretations,
            'granger_summary': granger_summary,
            'technical_summary': {
                'lag_order': selected_lag,
                'selection_criterion': ic_used.upper(),
                'n_variables': n_vars,
                'total_parameters': total_params,
                'variable_names': variable_names
            }
        }
        
    except Exception as e:
        l.error(f"Error interpreting VAR results: {e}")
        return {
            'overall_interpretation': "VAR model fitted successfully, but detailed interpretation unavailable due to an error.",
            'fevd_interpretations': {var: "Interpretation unavailable" for var in variable_names},
            'granger_summary': "Granger causality summary unavailable",
            'technical_summary': {
                'lag_order': selected_lag,
                'selection_criterion': ic_used.upper(),
                'n_variables': len(variable_names),
                'error': str(e)
            }
        }


