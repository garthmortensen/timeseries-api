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
            critical_values: Dict[str, Any] = result.get("Critical Values", {})
            critical_1 = None
            critical_5 = None
            critical_10 = None
            
            # Safe extraction of critical values
            if isinstance(critical_values, dict):
                critical_1 = critical_values.get("1%", None)
                critical_5 = critical_values.get("5%", None)
                critical_10 = critical_values.get("10%", None)
            
            p_value_percent = p_value * 100
            
            # Determine actual confidence level based on critical values
            # For ADF test, stationarity is indicated when ADF stat is MORE NEGATIVE than critical values
            confidence_level = "No significant evidence"
            evidence_strength = "No evidence"
            
            if critical_1 is not None and adf_stat < critical_1:
                confidence_level = "Very high confidence (99%+)"
                evidence_strength = "Overwhelming evidence"
            elif critical_5 is not None and adf_stat < critical_5:
                confidence_level = "High confidence (95%+)"
                evidence_strength = "Strong evidence"
            elif critical_10 is not None and adf_stat < critical_10:
                confidence_level = "Moderate confidence (90%+)"
                evidence_strength = "Moderate evidence"
            elif p_value < 0.05:
                # P-value suggests significance but critical values don't support it strongly
                confidence_level = "Borderline confidence (based on p-value)"
                evidence_strength = "Weak evidence"
            
            # Build interpretation with clean formatting
            interpretation = f"STATIONARITY TEST RESULTS FOR {series_name.upper()}\n"
            interpretation += "=" * 50 + "\n\n"
            
            # Explain what we're testing in simple terms
            interpretation += (
                "WHAT WE'RE TESTING:\n"
                "We're checking if the data behaves predictably over time (stationary) "
                "or if it wanders around unpredictably (non-stationary).\n\n"
                "The test assumes: 'This data is non-stationary (unpredictable).'\n"
                "We're trying to prove this assumption wrong.\n\n"
            )
            
            # Determine practical meaning and recommendation based on actual confidence level
            if "Very high confidence" in confidence_level:
                practical_meaning = (
                    "The data has stable, predictable patterns. The average level and "
                    "variability stay consistent over time. Perfect for most statistical models."
                )
                recommendation = "RECOMMENDATION: This data is ready to use - no transformation needed."
                p_interpretation = (
                    f"P-VALUE ({p_value:.4f}): If the data were truly unpredictable, there's "
                    f"less than a {p_value_percent:.1f}% chance you'd see results this clear "
                    f"just by random luck. Combined with critical value analysis, this provides "
                    f"overwhelming evidence of stationarity."
                )
                
            elif "High confidence" in confidence_level:
                practical_meaning = (
                    "The data shows stable, predictable patterns. The average and "
                    "variability are reasonably consistent. Good for most statistical models."
                )
                recommendation = "RECOMMENDATION: This data is likely ready to use without transformation."
                p_interpretation = (
                    f"P-VALUE ({p_value:.4f}): If the data were truly unpredictable, there's "
                    f"only a {p_value_percent:.1f}% chance you'd see results this clear by "
                    f"random chance. The critical value analysis confirms this at the 95% level."
                )
                
            elif "Moderate confidence" in confidence_level:
                practical_meaning = (
                    "The data shows some predictable patterns, but with uncertainty. "
                    "The behavior might not be perfectly consistent over time."
                )
                recommendation = "RECOMMENDATION: Check the data visually. It might need light transformation."
                p_interpretation = (
                    f"P-VALUE ({p_value:.4f}): The test shows some evidence of predictability "
                    f"(significant at 90% level based on critical values), but not at the "
                    f"standard 95% confidence level."
                )
                
            elif "Borderline confidence" in confidence_level:
                practical_meaning = (
                    "The data shows weak evidence of predictable patterns. The p-value suggests "
                    "significance, but the critical value analysis shows this is borderline."
                )
                recommendation = "RECOMMENDATION: Examine the data carefully - transformation may be needed."
                p_interpretation = (
                    f"P-VALUE ({p_value:.4f}): While the p-value suggests significance, "
                    f"the critical value analysis shows this is at the edge of statistical "
                    f"significance. This represents borderline evidence."
                )
                
            else:  # No significant evidence
                practical_meaning = (
                    "The data appears unpredictable - it wanders or drifts without "
                    "returning to a stable average. This makes standard statistical models struggle."
                )
                recommendation = "RECOMMENDATION: Transform this data (try 'differencing') before modeling."
                p_interpretation = (
                    f"P-VALUE ({p_value:.4f}): The test provides no significant evidence "
                    f"of predictability. Both p-value and critical value analysis suggest "
                    f"the data is non-stationary."
                )
            
            # Add the results summary with clear formatting
            interpretation += (
                f"THE BOTTOM LINE:\n"
                f"• Evidence strength: {evidence_strength} that the data is predictable\n"
                f"• Confidence level: {confidence_level}\n\n"
                f"WHAT THIS MEANS:\n"
                f"{practical_meaning}\n\n"
                f"{recommendation}\n\n"
                f"UNDERSTANDING THE STATISTICS:\n\n"
                f"{p_interpretation}\n\n"
            )
            
            # Add test statistic explanation
            if adf_stat < -3:
                stat_strength = "very strong"
            elif adf_stat < -2.5:
                stat_strength = "strong"
            elif adf_stat < -2:
                stat_strength = "moderate"
            else:
                stat_strength = "weak"
                
            interpretation += (
                f"TEST STATISTIC ({adf_stat:.4f}): Think of this as a 'predictability score.' "
                f"More negative = more predictable. This score shows {stat_strength} evidence "
                f"of predictability. The test compares this score against benchmarks to make "
                f"the final call.\n\n"
            )
            
            # Add critical value comparison if available
            if critical_5 is not None:
                interpretation += "BENCHMARK COMPARISONS:\n"
                if adf_stat < critical_5:
                    interpretation += f"• 5% Benchmark: PASS - This score ({adf_stat:.4f}) beats the benchmark ({critical_5:.4f})\n"
                else:
                    interpretation += f"• 5% Benchmark: BORDERLINE - This score ({adf_stat:.4f}) is close to the benchmark ({critical_5:.4f})\n"
                    
            if critical_1 is not None:
                if adf_stat < critical_1:
                    interpretation += f"• 1% Benchmark: PASS - Also beats the strict benchmark ({critical_1:.4f}) - excellent!\n"
                interpretation += "\n"
                    
            # Handle edge cases where p-value and critical values might disagree
            if p_value < p_value_threshold and critical_5 is not None and adf_stat >= critical_5:
                interpretation += (
                    f"TECHNICAL NOTE: The p-value suggests predictability but the test "
                    f"statistic is borderline with benchmarks. This happens in edge cases. "
                    f"The p-value is generally more reliable, so we lean toward predictable, "
                    f"but visual inspection of the data is recommended.\n"
                )
                
            interpretations[series_name] = interpretation
            
        except KeyError as e:
            l.warning(f"Missing key in ADF results for {series_name}: {e}")
            interpretations[series_name] = f"Unable to interpret results for {series_name} due to missing data."
        except Exception as e:
            l.error(f"Error interpreting stationarity for {series_name}: {e}")
            interpretations[series_name] = f"Error interpreting results for {series_name}."
            
    return interpretations


def interpret_arima_results(model_summary: str, forecast: list, residuals: list = None) -> str:
    """
    Create a human-readable interpretation of ARIMA model results.
    
    Args:
        model_summary (str): Summary of the fitted ARIMA model
        forecast (list): List of forecasted values
        residuals (list, optional): List of model residuals. Defaults to None.
        
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
            f"to generate forecasts, similar to how market analysis considers recent price movements "
            f"and error corrections to predict future behavior. "
            f"The ARIMA model has been fitted successfully. "
            f"The forecast shows {trend} trend over the forecast horizon, "
            f"with the data projected to follow a {plain_trend} trajectory {implication}. "
        )

        if residuals:
            interpretation += (
                f"\n\nResiduals Analysis: The model's residuals (the differences between the predicted and actual values) "
                f"are essential for diagnosing model fit. Ideally, they should resemble white noise, meaning they are "
                f"uncorrelated and have a constant variance. Visual inspection of the residuals plot can help identify "
                f"any remaining patterns (like autocorrelation or heteroscedasticity) that the model failed to capture. "
                f"If patterns are present, the model may need to be refined."
            )
        
        interpretation += (
            f"\n\nNote: ARIMA models assume volatility (price jumpiness) is constant over time, "
            f"which may not reflect real market conditions where volatility itself changes."
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
                market_context = "This pattern often coincides with rising market stress levels, reflecting increased uncertainty."
            elif forecast[-1] < forecast[0]:
                trend = "a decreasing"
                implication = "suggesting decreasing market uncertainty"
                plain_desc = "becoming more stable with narrower price ranges"
                market_context = "This trend typically aligns with calming market conditions, suggesting increased stability."
            else:
                trend = "a stable"
                implication = "suggesting stable market conditions"
                plain_desc = "maintaining its current volatility level"
                market_context = "This stable pattern reflects steady conditions in the underlying data."
        else:
            trend = "an unknown"
            implication = ""
            plain_desc = "showing unclear volatility patterns"
            market_context = ""
            
        interpretation = (
            f"The GARCH model has been fitted successfully, capturing the time-varying nature of volatility. "
            f"The volatility forecast shows {trend} trend {implication}. "
            f"In simple terms, the data is expected to be {plain_desc}. "
            f"Volatility clustering—periods of high volatility followed by high volatility and low volatility periods followed by low volatility—is a key feature that GARCH models capture effectively. "
            f"{market_context} "
            f"\n\nNote: GARCH models specifically model how volatility changes over time, unlike ARIMA models which assume constant volatility."
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
                f"This high interconnection indicates potential contagion effects "
                f"across assets or sectors, similar to how problems in one market "
                f"can spread throughout a financial system."
            )
        else:
            interpretation = (
                f"The total spillover index is {total_spillover:.4f}, indicating limited "
                f"interconnectedness between the time series. This suggests relatively "
                f"independent behavior of markets or variables, potentially allowing for "
                f"effective diversification strategies. "
                f"This lower interconnection indicates better diversification potential, "
                f"as markets appear to respond to their own specific factors rather than common drivers."
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
                        f"This means {series_name} {direction == 'transmitting' and 'acts as a leading indicator that influences broader market movements' or 'functions as a responsive asset that reacts significantly to external market forces'}. "
                        f"For diversification purposes, this {'limits its effectiveness as a risk reducer' if direction == 'receiving' else 'makes it a potential systemic risk source'}."
                    )
                else:
                    dir_interpretation = (
                        f"The series {series_name} is {direction} limited spillovers "
                        f"({value:.4f}), suggesting it is relatively isolated in terms of "
                        f"{direction == 'transmitting' and 'influencing' or 'being influenced by'} other series. "
                        f"From a diversification perspective, {series_name} {direction == 'transmitting' and 'has minimal impact on other assets' or 'shows relative independence from market movements, potentially offering diversification benefits'}."
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
                    f"**Highly Significant Causality (1% level)**: {source} strongly Granger-causes {target} "
                    f"(optimal lag: {optimal_lag_1pct}, min p-value: {min_p_value:.4f}). "
                    f"This indicates a very robust predictive relationship where past values of {source} "
                    f"contain significant information for predicting future values of {target}. "
                    f"In statistical terms, this represents a strong leading indicator relationship "
                    f"where changes in {source} systematically precede changes in {target}."
                )
            elif causality_5pct:
                interpretation = (
                    f"**Significant Causality (5% level)**: {source} Granger-causes {target} "
                    f"(optimal lag: {optimal_lag_5pct}, min p-value: {min_p_value:.4f}). "
                    f"This suggests that past values of {source} contain information that helps predict "
                    f"future values of {target}, beyond what is contained in past values of {target} alone. "
                    f"In statistical terms, movements in {source} systematically precede movements in {target}, "
                    f"creating a leading indicator relationship."
                )
            else:
                interpretation = (
                    f"**No Significant Causality**: {source} does not Granger-cause {target} "
                    f"(min p-value: {min_p_value:.4f}). "
                    f"This suggests that past values of {source} do not contain significant "
                    f"additional information for predicting future values of {target}. "
                    f"In statistical terms, this supports the hypothesis that these variables' relationship "
                    f"is essentially random, and {target}'s movements cannot be predicted using {source}'s historical data."
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
                f"between variables in the system. "
                f"In simple terms, this means {'changes in one variable strongly affect other variables' if total > 0.5 else 'changes in one variable have some effect on other variables' if total > 0.3 else 'changes in one variable have limited effects on other variables'}. "
                f"{'This suggests that shocks spread widely across the system during turbulent periods.' if total > 0.5 else 'This suggests some shock transmission exists, but certain disturbances could still spread across parts of the system.' if total > 0.3 else 'This suggests shocks tend to remain contained within specific variables.'}"
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
                                f"In statistical terms, this variable demonstrates sensitivity to external factors "
                                f"and broader system movements. Such variables typically show limited independence during periods of high spillover."
                            )
                        else:
                            dir_interp = (
                                f"{entity} receives minimal spillovers from other variables (value: {value:.4f}). "
                                f"This means {entity} is relatively insulated from shocks originating elsewhere. "
                                f"In statistical terms, this variable demonstrates independence from external factors, "
                                f"showing behavior that is more likely driven by idiosyncratic factors."
                            )
                    else:  # "from"
                        if value > significance_threshold:
                            dir_interp = (
                                f"{entity} transmits significant spillovers to other variables (value: {value:.4f}). "
                                f"This means shocks to {entity} spread widely through the system. "
                                f"In plain language, changes in this variable tend to affect other variables noticeably. "
                                f"When this variable experiences volatility, other related variables often follow suit."
                            )
                        else:
                            dir_interp = (
                                f"{entity} transmits minimal spillovers to other variables (value: {value:.4f}). "
                                f"This means shocks to {entity} remain largely contained. "
                                f"In simple terms, changes in this variable have limited impact on other variables. "
                                f"This suggests that volatility in this variable is unlikely to trigger widespread reactions in the system."
                            )
                    
                    interpretations[f"{direction.capitalize()} {entity}"] = dir_interp
        
        # Net spillovers
        if "net" in spillover_data:
            for entity, value in spillover_data["net"].items():
                if abs(value) < significance_threshold:
                    net_interp = (
                        f"{entity} has a balanced spillover profile (net: {value:.4f}), "
                        f"meaning it transmits and receives spillovers in roughly equal measure. "
                        f"In simple terms, this variable influences other variables about as much as it is influenced by them. "
                        f"This balanced relationship suggests the variable plays both leading and following roles in the system."
                    )
                elif value > 0:
                    net_interp = (
                        f"{entity} is a net transmitter of spillovers (net: {value:.4f}), "
                        f"meaning it influences other variables more than it is influenced by them. "
                        f"In plain English, this variable tends to lead changes in the system rather than follow them. "
                    )
                else:
                    net_interp = (
                        f"{entity} is a net receiver of spillovers (net: {value:.4f}), "
                        f"meaning it is influenced by other variables more than it influences them. "
                        f"In simple terms, this variable tends to follow changes in the system rather than lead them. "
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
            f"the dynamic relationships between {n_vars} time series ({', '.join(variable_names)}), "
            f"with a total of {total_params} parameters estimated. "
        )
        
        if selected_lag == 1:
            overall_interpretation += (
                "The single lag structure suggests that each variable's current value depends primarily on "
                "its own and other variables' values from the previous period, indicating relatively "
                "short-term interdependencies in the system."
            )
        elif selected_lag <= 3:
            overall_interpretation += (
                f"The {selected_lag}-lag structure indicates that past values from up to {selected_lag} periods "
                "influence current values, suggesting moderate persistence in cross-variable relationships "
                "and information transmission patterns."
            )
        else:
            overall_interpretation += (
                f"The {selected_lag}-lag structure reveals longer memory effects where movements "
                "from up to {selected_lag} periods ago still influence current behavior, indicating "
                "persistent interdependencies and potentially complex system dynamics."
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
                        "This indicates that the variable is relatively autonomous, with most forecast "
                        "uncertainty stemming from variable-specific factors rather than spillovers from other variables."
                    )
                elif own_contribution > 50:
                    interpretation += (
                        "This shows a balanced influence where the variable has moderate independence but "
                        "is also meaningfully affected by external factors."
                    )
                else:
                    interpretation += (
                        "This reveals high interconnectedness where external factors dominate "
                        "forecast uncertainty, making the variable highly sensitive to spillover effects."
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
                'total_parameters': total_params,
                'error': str(e)
            }
        }


def interpret_multivariate_garch_results(mgarch_results: Dict[str, Any], 
                                        variable_names: list,
                                        lambda_val: float = 0.95) -> Dict[str, str]:
    """
    Create comprehensive human-readable interpretations of multivariate GARCH analysis results.
    
    Args:
        mgarch_results (Dict[str, Any]): Dictionary containing multivariate GARCH results
        variable_names (list): Names of the variables analyzed
        lambda_val (float): EWMA decay parameter used in DCC analysis
        
    Returns:
        Dict[str, str]: Dictionary of interpretations for multivariate GARCH components
    """
    interpretations = {}
    
    try:
        # 1. CCC-GARCH Interpretation (Constant Conditional Correlation)
        if "cc_correlation" in mgarch_results and mgarch_results["cc_correlation"] is not None:
            cc_corr = mgarch_results["cc_correlation"]
            
            if hasattr(cc_corr, 'values'):
                # Extract correlation values for interpretation
                correlation_pairs = []
                for i in range(len(variable_names)):
                    for j in range(i+1, len(variable_names)):
                        if hasattr(cc_corr, 'iloc'):
                            corr_value = cc_corr.iloc[i, j]
                        else:
                            corr_value = cc_corr[i][j] if isinstance(cc_corr, list) else cc_corr
                        correlation_pairs.append((variable_names[i], variable_names[j], corr_value))
                
                # Generate interpretation based on correlation strength
                high_corr_pairs = [(pair[0], pair[1], pair[2]) for pair in correlation_pairs if abs(pair[2]) > 0.7]
                moderate_corr_pairs = [(pair[0], pair[1], pair[2]) for pair in correlation_pairs if 0.3 < abs(pair[2]) <= 0.7]
                low_corr_pairs = [(pair[0], pair[1], pair[2]) for pair in correlation_pairs if abs(pair[2]) <= 0.3]
                
                cc_interpretation = (
                    f"**Constant Conditional Correlation (CCC-GARCH) Analysis**: "
                    f"This model assumes that correlations between variables remain constant over time, "
                    f"while allowing individual volatilities to vary according to their own GARCH processes. "
                )
                
                if high_corr_pairs:
                    cc_interpretation += (
                        f"**High Correlations (>0.7)**: {', '.join([f'{pair[0]}-{pair[1]} ({pair[2]:.3f})' for pair in high_corr_pairs])}. "
                        f"These variable pairs move very similarly, indicating strong co-movement patterns. "
                    )
                
                if moderate_corr_pairs:
                    cc_interpretation += (
                        f"**Moderate Correlations (0.3-0.7)**: {', '.join([f'{pair[0]}-{pair[1]} ({pair[2]:.3f})' for pair in moderate_corr_pairs])}. "
                        f"These relationships suggest some co-movement but also periods of independent behavior. "
                    )
                
                if low_corr_pairs:
                    cc_interpretation += (
                        f"**Low Correlations (<0.3)**: {', '.join([f'{pair[0]}-{pair[1]} ({pair[2]:.3f})' for pair in low_corr_pairs])}. "
                        f"These variable pairs show largely independent behavior patterns. "
                    )
                
            else:
                cc_interpretation = "CCC-GARCH analysis completed, but correlation matrix format is not interpretable."
            
            interpretations["CCC_GARCH"] = cc_interpretation
        
        # 2. DCC-GARCH Interpretation (Dynamic Conditional Correlation)
        if "dcc_correlation" in mgarch_results and mgarch_results["dcc_correlation"] is not None:
            dcc_corr = mgarch_results["dcc_correlation"]
            
            # Analyze time-varying correlation patterns
            if hasattr(dcc_corr, '__len__') and len(dcc_corr) > 1:
                # Calculate correlation statistics
                if hasattr(dcc_corr, 'mean'):
                    mean_corr = dcc_corr.mean()
                    std_corr = dcc_corr.std()
                    min_corr = dcc_corr.min()
                    max_corr = dcc_corr.max()
                else:
                    # Handle list or array format
                    import numpy as np
                    dcc_array = np.array(dcc_corr)
                    mean_corr = np.mean(dcc_array)
                    std_corr = np.std(dcc_array)
                    min_corr = np.min(dcc_array)
                    max_corr = np.max(dcc_array)
                
                # Determine correlation regime
                if std_corr > 0.15:
                    volatility_regime = "highly variable"
                    regime_desc = "significant time-varying correlation patterns"
                elif std_corr > 0.08:
                    volatility_regime = "moderately variable"
                    regime_desc = "noticeable time-varying correlation patterns"
                else:
                    volatility_regime = "relatively stable"
                    regime_desc = "limited time-varying correlation patterns"
                
                dcc_interpretation = (
                    f"**Dynamic Conditional Correlation (DCC-GARCH) Analysis** (λ={lambda_val}): "
                    f"This advanced model captures time-varying correlations using EWMA with {(1-lambda_val)*100:.1f}% weight on recent observations. "
                    f"The correlation between variables shows {regime_desc}. "
                    f"**Statistical Summary**: Average correlation: {mean_corr:.3f}, "
                    f"Range: {min_corr:.3f} to {max_corr:.3f}, Volatility: {std_corr:.3f}. "
                    f"The {volatility_regime} correlation indicates that the strength of the relationship between variables changes over time. "
                )
                
                # Add specific insights based on correlation patterns
                if mean_corr > 0.6:
                    dcc_interpretation += (
                        "The high average correlation suggests these variables typically move together, "
                        "particularly during periods of stress when correlations tend to increase. "
                    )
                elif mean_corr > 0.3:
                    dcc_interpretation += (
                        "The moderate average correlation shows balanced co-movement patterns "
                        "with periods of both synchronized and independent behavior. "
                    )
                else:
                    dcc_interpretation += (
                        "The low average correlation suggests these variables generally move independently, "
                        "showing distinct behavioral patterns across different time periods. "
                    )
                
            else:
                dcc_interpretation = "DCC-GARCH analysis completed, but insufficient data for time-series correlation interpretation."
            
            interpretations["DCC_GARCH"] = dcc_interpretation
        
        # 3. Covariance Matrix Interpretation
        if "cc_covariance_matrix" in mgarch_results and mgarch_results["cc_covariance_matrix"] is not None:
            cov_matrix = mgarch_results["cc_covariance_matrix"]
            
            # Extract volatilities (diagonal elements) and covariances
            if hasattr(cov_matrix, 'shape') and len(cov_matrix.shape) == 2:
                volatilities = []
                covariances = []
                
                for i, var_name in enumerate(variable_names):
                    if hasattr(cov_matrix, 'iloc'):
                        vol = np.sqrt(cov_matrix.iloc[i, i])
                    else:
                        vol = np.sqrt(cov_matrix[i][i])
                    volatilities.append((var_name, vol))
                    
                    for j in range(i+1, len(variable_names)):
                        if hasattr(cov_matrix, 'iloc'):
                            cov_val = cov_matrix.iloc[i, j]
                        else:
                            cov_val = cov_matrix[i][j]
                        covariances.append((variable_names[i], variable_names[j], cov_val))
                
                cov_interpretation = (
                    f"**Covariance Matrix Analysis**: "
                    f"Individual variable volatilities: {', '.join([f'{vol[0]}: {vol[1]:.4f}' for vol in volatilities])}. "
                )
                
                # Interpret covariance relationships
                positive_cov = [cov for cov in covariances if cov[2] > 0]
                negative_cov = [cov for cov in covariances if cov[2] < 0]
                
                if positive_cov:
                    cov_interpretation += (
                        f"**Positive Covariances**: {', '.join([f'{cov[0]}-{cov[1]} ({cov[2]:.6f})' for cov in positive_cov])} "
                        f"indicate these variables tend to move in the same direction. "
                    )
                
                if negative_cov:
                    cov_interpretation += (
                        f"**Negative Covariances**: {', '.join([f'{cov[0]}-{cov[1]} ({cov[2]:.6f})' for cov in negative_cov])} "
                        f"indicate these variables tend to move in opposite directions. "
                    )
                
            else:
                cov_interpretation = "Covariance matrix analysis completed, but matrix format is not interpretable."
            
            interpretations["Covariance_Analysis"] = cov_interpretation
        
        # 4. Statistical Analysis Summary
        has_ccc = "cc_correlation" in mgarch_results and mgarch_results["cc_correlation"] is not None
        has_dcc = "dcc_correlation" in mgarch_results and mgarch_results["dcc_correlation"] is not None
        
        statistical_interpretation = (
            f"**Statistical Analysis Summary**: The multivariate GARCH analysis provides insights into "
            f"the joint behavior of {len(variable_names)} time series. "
        )
        
        if has_ccc and has_dcc:
            statistical_interpretation += (
                "both constant and dynamic correlation models provide insights into variable relationships. "
                "The constant correlation model shows long-term average relationships, "
                "while the dynamic model reveals how these relationships change over time. "
            )
        elif has_ccc:
            statistical_interpretation += (
                "the constant correlation model provides baseline understanding of variable relationships. "
                "These stable correlation estimates represent average historical relationships between variables. "
            )
        elif has_dcc:
            statistical_interpretation += (
                "the dynamic correlation model reveals time-varying relationships between variables. "
                "This shows how the strength of variable relationships changes across different time periods. "
            )
        
        statistical_interpretation += (
            "These correlation and covariance estimates provide a foundation for understanding "
            "how variables move together and can be used for further statistical modeling and analysis."
        )
        
        interpretations["Statistical_Analysis_Summary"] = statistical_interpretation
        
        # 5. Overall Multivariate GARCH Summary
        overall_interpretation = (
            f"**Comprehensive Multivariate GARCH Analysis Summary**: "
            f"The analysis successfully modeled the joint volatility and correlation dynamics of {len(variable_names)} time series. "
        )
        
        model_count = sum([
            1 if "cc_correlation" in mgarch_results and mgarch_results["cc_correlation"] is not None else 0,
            1 if "dcc_correlation" in mgarch_results and mgarch_results["dcc_correlation"] is not None else 0
        ])
        
        if model_count == 2:
            overall_interpretation += (
                "Both CCC-GARCH and DCC-GARCH models were fitted successfully, providing complementary perspectives on correlation dynamics. "
                "This dual approach offers robust insights for understanding both long-term average relationships and short-term correlation changes. "
            )
        elif model_count == 1:
            overall_interpretation += (
                "The multivariate GARCH model was fitted successfully, providing valuable insights into correlation and volatility dynamics. "
            )
        else:
            overall_interpretation += (
                "Multivariate GARCH analysis encountered issues, but partial results may still provide useful insights. "
            )
        
        overall_interpretation += (
            "**Key Benefits**: (1) Captures volatility clustering in individual time series, "
            "(2) Models correlation dynamics between variables, "
            "(3) Provides foundation for statistical modeling and analysis, "
            "(4) Enables sophisticated scenario analysis and forecasting."
        )
        
        interpretations["Overall_MGARCH_Summary"] = overall_interpretation
        
        return interpretations
        
    except Exception as e:
        l.error(f"Error interpreting multivariate GARCH results: {e}")
        return {
            "error": f"Unable to interpret multivariate GARCH results due to an error: {str(e)}",
            "CCC_GARCH": "CCC-GARCH interpretation unavailable",
            "DCC_GARCH": "DCC-GARCH interpretation unavailable",
            "Covariance_Analysis": "Covariance analysis unavailable",
            "Statistical_Analysis_Summary": "Statistical analysis summary unavailable",
            "Overall_MGARCH_Summary": "Overall MGARCH summary unavailable"
        }


def interpret_correlation_dynamics(correlation_series: Any, 
                                 variable_names: list,
                                 analysis_period: str = "full sample") -> str:
    """
    Interpret time-varying correlation patterns from DCC-GARCH analysis.
    
    Args:
        correlation_series: Time series of correlation values
        variable_names (list): Names of the variables being analyzed
        analysis_period (str): Description of the analysis period
        
    Returns:
        str: Human-readable interpretation of correlation dynamics
    """
    try:
        if hasattr(correlation_series, '__len__') and len(correlation_series) > 1:
            import numpy as np
            
            # Convert to numpy array for analysis
            if hasattr(correlation_series, 'values'):
                corr_array = correlation_series.values
            else:
                corr_array = np.array(correlation_series)
            
            # Calculate statistical measures
            mean_corr = np.mean(corr_array)
            median_corr = np.median(corr_array)
            std_corr = np.std(corr_array)
            min_corr = np.min(corr_array)
            max_corr = np.max(corr_array)
            
            # Identify correlation regimes
            high_corr_periods = np.sum(corr_array > (mean_corr + std_corr))
            low_corr_periods = np.sum(corr_array < (mean_corr - std_corr))
            total_periods = len(corr_array)
            
            interpretation = (
                f"**Correlation Dynamics Analysis** between {' and '.join(variable_names)} over {analysis_period}: "
                f"The correlation exhibits {'high' if std_corr > 0.15 else 'moderate' if std_corr > 0.08 else 'low'} variability "
                f"(standard deviation: {std_corr:.3f}). "
                f"**Statistical Profile**: Mean: {mean_corr:.3f}, Median: {median_corr:.3f}, "
                f"Range: [{min_corr:.3f}, {max_corr:.3f}]. "
            )
            
            # Regime analysis
            if high_corr_periods > 0 or low_corr_periods > 0:
                interpretation += (
                    f"**Regime Analysis**: "
                    f"{high_corr_periods}/{total_periods} periods ({high_corr_periods/total_periods*100:.1f}%) show unusually high correlation, "
                    f"{low_corr_periods}/{total_periods} periods ({low_corr_periods/total_periods*100:.1f}%) show unusually low correlation. "
                )
                
                if high_corr_periods > low_corr_periods:
                    interpretation += (
                        "The prevalence of high-correlation periods suggests these variables frequently move together, "
                        "particularly during periods of stress when correlations tend to increase. "
                    )
                elif low_corr_periods > high_corr_periods:
                    interpretation += (
                        "The prevalence of low-correlation periods suggests these variables often move independently, "
                        "providing opportunities for understanding distinct behavioral patterns. "
                    )
                else:
                    interpretation += (
                        "The balanced occurrence of high and low correlation periods suggests variable relationships "
                        "that change depending on conditions - active correlation monitoring reveals these patterns. "
                    )
            
            # Statistical implications
            if mean_corr > 0.7:
                interpretation += (
                    "**Statistical Implication**: High average correlation indicates strong co-movement - "
                    "these variables show similar response patterns to underlying factors."
                )
            elif mean_corr > 0.3:
                interpretation += (
                    "**Statistical Implication**: Moderate average correlation indicates balanced relationships - "
                    "variables show some co-movement while maintaining distinct characteristics."
                )
            else:
                interpretation += (
                    "**Statistical Implication**: Low average correlation indicates largely independent behavior - "
                    "these variables respond to different underlying factors and drivers."
                )
            
            return interpretation
            
        else:
            return f"Correlation dynamics analysis between {' and '.join(variable_names)}: Insufficient data for time-series analysis."
            
    except Exception as e:
        l.error(f"Error interpreting correlation dynamics: {e}")
        return f"Unable to interpret correlation dynamics between {' and '.join(variable_names)} due to an error: {str(e)}"


def interpret_portfolio_risk_metrics(cov_matrix: Any, 
                                   asset_names: list,
                                   equal_weights: bool = True) -> str:
    """
    Interpret portfolio risk metrics derived from multivariate GARCH covariance matrix.
    
    Args:
        cov_matrix: Covariance matrix from MGARCH analysis
        asset_names (list): Names of assets in the portfolio
        equal_weights (bool): Whether to use equal weights for portfolio analysis
        
    Returns:
        str: Human-readable interpretation of portfolio risk metrics
    """
    try:
        import numpy as np
        
        # Convert covariance matrix to numpy array
        if hasattr(cov_matrix, 'values'):
            cov_array = cov_matrix.values
        elif hasattr(cov_matrix, 'to_numpy'):
            cov_array = cov_matrix.to_numpy()
        else:
            cov_array = np.array(cov_matrix)
        
        n_assets = len(asset_names)
        
        # Calculate individual asset volatilities
        individual_vols = [np.sqrt(cov_array[i, i]) for i in range(n_assets)]
        
        # Calculate portfolio volatility (assuming equal weights)
        if equal_weights:
            weights = np.array([1.0/n_assets] * n_assets)
        else:
            weights = np.array([1.0/n_assets] * n_assets)  # Default to equal weights
        
        portfolio_variance = np.dot(weights, np.dot(cov_array, weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Calculate diversification ratio
        weighted_avg_vol = np.dot(weights, individual_vols)
        diversification_ratio = weighted_avg_vol / portfolio_vol
        
        # Calculate risk contributions
        marginal_risk = np.dot(cov_array, weights)
        risk_contributions = weights * marginal_risk / portfolio_variance
        
        interpretation = (
            f"**Portfolio Risk Analysis** ({n_assets} assets with {'equal' if equal_weights else 'optimized'} weights): "
            f"**Individual Volatilities**: {', '.join([f'{asset_names[i]}: {individual_vols[i]:.4f}' for i in range(n_assets)])}. "
            f"**Portfolio Volatility**: {portfolio_vol:.4f} "
            f"**Diversification Ratio**: {diversification_ratio:.3f} "
        )
        
        # Interpret diversification effectiveness
        if diversification_ratio > 1.5:
            div_assessment = "excellent diversification benefits"
            div_explanation = "The portfolio volatility is significantly lower than the weighted average of individual volatilities"
        elif diversification_ratio > 1.2:
            div_assessment = "good diversification benefits"
            div_explanation = "The portfolio volatility is noticeably lower than the weighted average of individual volatilities"
        elif diversification_ratio > 1.05:
            div_assessment = "moderate diversification benefits"
            div_explanation = "The portfolio volatility is slightly lower than the weighted average of individual volatilities"
        else:
            div_assessment = "limited diversification benefits"
            div_explanation = "The portfolio volatility is close to the weighted average of individual volatilities"
        
        interpretation += (
            f"indicates {div_assessment}. {div_explanation}. "
        )
        
        # Risk contribution analysis
        max_contributor_idx = np.argmax(risk_contributions)
        min_contributor_idx = np.argmin(risk_contributions)
        
        interpretation += (
            f"**Risk Contributions**: {asset_names[max_contributor_idx]} contributes most to portfolio risk "
            f"({risk_contributions[max_contributor_idx]*100:.1f}%), while {asset_names[min_contributor_idx]} "
            f"contributes least ({risk_contributions[min_contributor_idx]*100:.1f}%). "
        )
        
        # Investment recommendations
        if diversification_ratio > 1.3:
            recommendation = (
                "**Recommendation**: Maintain current diversification approach - the portfolio structure effectively reduces risk. "
                "Consider this covariance structure for optimal weight determination."
            )
        elif diversification_ratio > 1.1:
            recommendation = (
                "**Recommendation**: Diversification is working but could be improved. "
                "Consider adjusting weights to reduce concentration in the highest risk-contributing assets."
            )
        else:
            recommendation = (
                "**Recommendation**: Limited diversification suggests these assets are highly correlated. "
                "Consider adding uncorrelated assets or implementing correlation-based risk management strategies."
            )
        
        interpretation += recommendation
        
        return interpretation
        
    except Exception as e:
        l.error(f"Error interpreting portfolio risk metrics: {e}")
        return f"Unable to interpret portfolio risk metrics due to an error: {str(e)}"


