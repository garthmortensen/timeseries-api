#!/usr/bin/env python3
# timeseries-api/api/services/interpretations.py
"""
Interpretation module for statistical test results.
Contains functions to create human-readable interpretations of statistical test results.
"""

import logging as l
import numpy as np
from typing import Dict, Any, List, Union


def interpret_stationarity_test(adf_results: Dict[str, Dict[str, float]], 
                               p_value_threshold: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """
    Interpret Augmented Dickey-Fuller test results for stationarity.
    
    Args:
        adf_results (Dict[str, Dict[str, float]]): Dictionary of ADF test results
        p_value_threshold (float, optional): P-value threshold for significance. Defaults to 0.05.
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of structured interpretations for each series
    """
    interpretations = {}
    
    for series_name, result in adf_results.items():
        try:
            adf_stat = result.get("adf_statistic")
            p_value = result.get("p_value")
            critical_values = result.get("critical_values", {})
            
            # Check for None values
            if adf_stat is None or p_value is None:
                interpretations[series_name] = {
                    "error": f"Missing required values for {series_name}: adf_statistic={adf_stat}, p_value={p_value}"
                }
                continue
            
            # Safely calculate evidence strength and stationarity
            evidence_strength = "strong" if p_value < 0.01 else "moderate" if p_value < 0.05 else "weak"
            is_stationary = p_value < p_value_threshold
            
            # Get critical values for detailed analysis
            critical_1pct = critical_values.get("1%", -3.75)
            critical_5pct = critical_values.get("5%", -3.0)
            critical_10pct = critical_values.get("10%", -2.63)
            
            # Determine statistical significance level based on critical values
            if adf_stat < critical_1pct:
                significance_level = "1% (highly significant)"
                critical_comparison = f"exceeds the 1% critical value ({critical_1pct:.2f})"
            elif adf_stat < critical_5pct:
                significance_level = "5% (significant)"
                critical_comparison = f"exceeds the 5% critical value ({critical_5pct:.2f}) but not the 1% level"
            elif adf_stat < critical_10pct:
                significance_level = "10% (marginally significant)"
                critical_comparison = f"exceeds the 10% critical value ({critical_10pct:.2f}) but not higher significance levels"
            else:
                significance_level = "Not significant"
                critical_comparison = f"does not exceed any conventional critical values"
            
            # Create detailed bottom line explanation
            if p_value < 0.01:
                if adf_stat < critical_1pct:
                    bottom_line_detail = f"Strong evidence for stationarity. The p-value of {p_value:.6f} shows highly significant results, and the test statistic ({adf_stat:.4f}) {critical_comparison}."
                else:
                    bottom_line_detail = f"The p-value of {p_value:.6f} suggests high significance, but the test statistic ({adf_stat:.4f}) shows this is at the edge of statistical significance at the 1% level. This represents strong but borderline evidence."
            elif p_value < 0.05:
                if adf_stat < critical_5pct:
                    bottom_line_detail = f"Moderate evidence for stationarity. The p-value of {p_value:.6f} shows significant results, and the test statistic ({adf_stat:.4f}) {critical_comparison}."
                else:
                    bottom_line_detail = f"The p-value of {p_value:.6f} suggests significance, but the test statistic ({adf_stat:.4f}) shows this is at the edge of statistical significance. This represents borderline evidence."
            else:
                bottom_line_detail = f"Insufficient evidence for stationarity. The p-value of {p_value:.6f} exceeds the significance threshold, and the test statistic ({adf_stat:.4f}) {critical_comparison}."
            
            # Create test statistic interpretation
            if adf_stat < -4.0:
                predictability_desc = "very strong evidence of predictability"
            elif adf_stat < -3.0:
                predictability_desc = "strong evidence of predictability"
            elif adf_stat < -2.5:
                predictability_desc = "moderate evidence of predictability"
            else:
                predictability_desc = "weak evidence of predictability"
            
            test_statistic_explanation = (
                f"Think of the adf test statistic ({adf_stat:.4f}) as a 'predictability score.' "
                f"More negative = more predictable patterns in the data. This score shows {predictability_desc}. "
                f"The test compares this score against critical value benchmarks to make the final statistical call."
            )

            # Create detailed hypothesis testing explanation
            if is_stationary:
                hypothesis_decision = f"We REJECT the null hypothesis (H₀: unit root is present) at the {p_value_threshold*100}% significance level"
                hypothesis_explanation = (
                    f"Since p-value ({p_value:.6f}) < alpha ({p_value_threshold}), we have sufficient evidence to "
                    f"reject H₀ and accept the alternative hypothesis (H₁: no unit root, series is stationary)."
                )
            else:
                hypothesis_decision = f"We FAIL TO REJECT the null hypothesis (H₀: unit root is present) at the {p_value_threshold*100}% significance level"
                hypothesis_explanation = (
                    f"Since p-value ({p_value:.6f}) ≥ alpha ({p_value_threshold}), we do not have sufficient evidence to "
                    f"reject H₀. We cannot conclude that the series is stationary."
                )

            interpretations[series_name] = {
                "what_were_testing": "Testing whether the time series is stationary (constant mean, variance, and autocorrelation).",
                "purpose": "To determine if the series requires differencing for ARIMA modeling or other transformations.",
                "key_ideas": [
                    "Stationarity is crucial for many time series models.",
                    "ADF test checks for unit roots to assess stationarity.", 
                    "H₀: Unit root is present (series is non-stationary)",
                    "H₁: No unit root (series is stationary)",
                    "More negative test statistics indicate stronger evidence against unit roots.",
                    "UNIT ROOT EXPLAINED: A unit root means the series has a 'memory' - shocks have permanent effects and the series doesn't return to a long-term mean."
                ],
                "metrics": {
                    "adf_statistic": adf_stat,
                    "p_value": p_value,
                    "critical_values": critical_values,
                    "significance_level": significance_level,
                    "test_statistic_explanation": test_statistic_explanation,
                    "unit_root_explanation": (
                        f"WHAT IS A UNIT ROOT? Think of it as 'permanent memory' in your data. "
                        f"If a time series has a unit root, it means: (1) Shocks have LASTING effects - if the value jumps up, "
                        f"it tends to stay at that higher level rather than returning to the original trend. "
                        f"(2) The series 'wanders' without a fixed long-term average - like a random walk where each step "
                        f"depends entirely on the previous step plus some random change. "
                        f"(3) Variance grows over time - the uncertainty about future values increases the further you forecast. "
                        f"REAL-WORLD EXAMPLE: Stock prices often have unit roots - if Apple stock jumps from $150 to $160 on good news, "
                        f"it doesn't automatically drift back to $150. The new level becomes the new 'baseline' for future movements."
                    )
                },
                "results": {
                    "bottom_line": "Stationary" if is_stationary else "Non-stationary",
                    "bottom_line_detailed": bottom_line_detail,
                    "confidence_level": f"{(1 - p_value) * 100:.1f}%",
                    "evidence_strength": evidence_strength,
                    "hypothesis_decision": hypothesis_decision,
                    "hypothesis_explanation": hypothesis_explanation,
                    "statistical_interpretation": (
                        f"The ADF statistic of {adf_stat:.4f} compared against critical values shows "
                        f"{significance_level.lower()} evidence. {hypothesis_explanation}"
                    )
                },
                "implications": {
                    "practical_meaning": f"The series is {'stationary' if is_stationary else 'non-stationary'}. {'This means the statistical properties remain relatively constant over time, making it suitable for modeling without differencing.' if is_stationary else 'This indicates the statistical properties change over time and the series likely requires differencing to achieve stationarity.'}",
                    "recommendations": "Proceed with modeling as is." if is_stationary else "Apply differencing or transformations to stabilize the series before modeling.",
                    "limitations": "ADF test may not detect all forms of non-stationarity, particularly structural breaks or non-linear trends.",
                    "methodology_notes": (
                        "HYPOTHESIS TESTING FRAMEWORK: "
                        f"• H₀ (Null): Unit root is present → series is non-stationary. "
                        f"• H₁ (Alternative): No unit root → series is stationary. "
                        f"• Decision rule: If p-value < alpha ({p_value_threshold}), reject H₀ in favor of H₁. "
                        f"• Test statistic measures deviation from random walk behavior - more negative values provide stronger evidence against H₀."
                    ),
                    "unit_root_deep_dive": (
                        "UNIT ROOT TECHNICAL DETAILS: "
                        "A unit root exists when the autoregressive coefficient equals 1 in the equation: y(t) = rho*y(t-1) + ε(t). "
                        "When rho = 1 (unit root), the series is a random walk and non-stationary. "
                        "When |rho| < 1 (no unit root), the series is mean-reverting and stationary. "
                        "The ADF test essentially tests whether rho = 1 (unit root) vs rho < 1 (stationary). "
                        "FINANCIAL INTUITION: Unit roots are common in financial data because markets incorporate new information permanently. "
                        "When Tesla announces a breakthrough, the stock price adjusts to a new level and doesn't 'forget' this information."
                    )
                }
            }
        except Exception as e:
            interpretations[series_name] = {
                "error": f"Error interpreting results for {series_name}: {e}"
            }
    
    return interpretations


def interpret_arima_results(model_summary: str, forecast: list, residuals: list = None) -> Dict[str, Any]:
    """
    Create a structured interpretation of ARIMA model results.
    
    Args:
        model_summary (str): Summary of the fitted ARIMA model
        forecast (list): List of forecasted values
        residuals (list, optional): List of model residuals. Defaults to None.
        
    Returns:
        Dict[str, Any]: Structured interpretation of the ARIMA model results
    """
    try:
        # Basic trend analysis
        trend = "increasing" if forecast[-1] > forecast[0] else "decreasing" if forecast[-1] < forecast[0] else "stable"
        
        # Extract model order if available in summary
        p, d, q = 0, 0, 0
        if "ARIMA(" in model_summary and ")" in model_summary:
            try:
                order_part = model_summary.split("ARIMA(")[1].split(")")[0]
                p, d, q = map(int, order_part.split(","))
            except:
                pass
        
        # Calculate forecast statistics
        forecast_array = np.array(forecast)
        forecast_mean = float(np.mean(forecast_array))
        forecast_std = float(np.std(forecast_array))
        forecast_range = float(np.max(forecast_array) - np.min(forecast_array))
        
        # Determine forecast confidence based on volatility
        if abs(forecast_mean) < 1e-6:  # Avoid division by zero
            forecast_confidence = "Moderate"
            confidence_desc = "stable near-zero forecasts"
        else:
            coefficient_of_variation = forecast_std / abs(forecast_mean)
            if coefficient_of_variation < 0.02:
                forecast_confidence = "High"
                confidence_desc = "very stable"
            elif coefficient_of_variation < 0.05:
                forecast_confidence = "Moderate"
                confidence_desc = "moderately stable"
            else:
                forecast_confidence = "Low"
                confidence_desc = "highly variable"
        
        # Calculate accuracy metrics if residuals are available
        accuracy_metrics = {}
        model_quality = "Unknown"
        if residuals is not None:
            residuals = np.array(residuals)
            residuals = residuals[~np.isnan(residuals)]  # Remove NaN values
            
            if len(residuals) > 0:
                accuracy_metrics = {
                    "mean_error": float(np.mean(residuals)),
                    "mae": float(np.mean(np.abs(residuals))),
                    "rmse": float(np.sqrt(np.mean(residuals**2))),
                    "residual_std": float(np.std(residuals))
                }
                
                # Assess model quality based on residual statistics
                mean_abs_error = accuracy_metrics["mae"]
                if mean_abs_error < 0.01:
                    model_quality = "Excellent"
                elif mean_abs_error < 0.05:
                    model_quality = "Good"
                elif mean_abs_error < 0.10:
                    model_quality = "Fair"
                else:
                    model_quality = "Poor"
        
        # Create detailed model interpretation based on ARIMA components
        ar_interpretation = ""
        if p > 0:
            ar_interpretation = f"AR({p}) component captures {p} period(s) of autoregressive memory - the model uses the past {p} value(s) to predict future values."
        
        i_interpretation = ""
        if d > 0:
            i_interpretation = f"I({d}) component applies {d} level(s) of differencing to make the series stationary - removing trends and ensuring stable statistical properties."
        
        ma_interpretation = ""
        if q > 0:
            ma_interpretation = f"MA({q}) component models {q} period(s) of moving average effects - capturing short-term dependencies in forecast errors."
        
        # Create forecast interpretation based on trend and magnitude
        forecast_magnitude = abs((forecast[-1] - forecast[0]) / forecast[0]) if forecast[0] != 0 else 0
        if forecast_magnitude > 0.1:
            magnitude_desc = "substantial"
        elif forecast_magnitude > 0.05:
            magnitude_desc = "moderate"
        else:
            magnitude_desc = "minimal"
        
        # Bottom line detailed explanation
        bottom_line_detail = (
            f"The ARIMA({p},{d},{q}) model forecasts a {trend} trend with {magnitude_desc} change "
            f"({forecast_magnitude*100:.1f}% from start to end). Forecast confidence is {forecast_confidence.lower()} "
            f"due to {confidence_desc} predictions. "
        )
        
        if model_quality != "Unknown":
            bottom_line_detail += f"Model quality is assessed as {model_quality.lower()} based on residual analysis."
        
        # Model components explanation
        components_explanation = []
        if p > 0:
            components_explanation.append(f"AR({p}): Uses past {p} values as predictors")
        if d > 0:
            components_explanation.append(f"I({d}): {d} level(s) of differencing applied")
        if q > 0:
            components_explanation.append(f"MA({q}): Models {q} forecast error terms")
        
        model_explanation = (
            f"ARIMA({p},{d},{q}) MODEL BREAKDOWN: Think of this as a 'prediction recipe' with three ingredients. "
            f"{' + '.join(components_explanation)}. "
            f"The model essentially says: 'To predict tomorrow, look at the past {max(p,q,1)} values, "
            f"account for any trending behavior{' (via differencing)' if d > 0 else ''}, "
            f"and adjust for recent forecast errors{' using moving averages' if q > 0 else ''}.'"
        )

        return {
            "what_were_testing": "Fitting an ARIMA time series model to capture patterns and forecast future values.",
            "purpose": "To identify underlying patterns in time series data and make accurate forecasts based on historical behavior.",
            "key_ideas": [
                f"ARIMA({p},{d},{q}) combines autoregression (AR), differencing (I), and moving averages (MA).",
                "AR component: Uses past values to predict future values (memory effect).",
                "I component: Removes trends through differencing to achieve stationarity.",
                "MA component: Models dependencies in forecast errors (shock persistence).",
                "Model assumes that historical patterns will continue into the future."
            ],
            "metrics": {
                "model_order": f"ARIMA({p},{d},{q})",
                "forecast_trend": trend,
                "forecast_values": forecast,
                "forecast_statistics": {
                    "mean": forecast_mean,
                    "std": forecast_std,
                    "range": forecast_range,
                    "magnitude_change": forecast_magnitude
                },
                "accuracy_metrics": accuracy_metrics,
                "model_explanation": model_explanation,
                "components_breakdown": {
                    "ar_interpretation": ar_interpretation,
                    "i_interpretation": i_interpretation,
                    "ma_interpretation": ma_interpretation
                }
            },
            "results": {
                "bottom_line": f"Forecasts {trend} trend",
                "bottom_line_detailed": bottom_line_detail,
                "confidence_level": forecast_confidence,
                "evidence_strength": model_quality if model_quality != "Unknown" else "Moderate",
                "forecast_assessment": (
                    f"The model predicts {trend} movement with {confidence_desc} forecasts. "
                    f"The {magnitude_desc} change magnitude suggests "
                    f"{'significant directional movement' if forecast_magnitude > 0.05 else 'relatively stable behavior'} "
                    f"in the forecast period."
                ),
                "model_performance": (
                    f"Model fit quality: {model_quality}. " +
                    (f"Mean absolute error of {accuracy_metrics['mae']:.4f} indicates "
                     f"{'excellent' if model_quality == 'Excellent' else 'good' if model_quality == 'Good' else 'acceptable' if model_quality == 'Fair' else 'poor'} "
                     f"predictive accuracy." if accuracy_metrics else "Performance metrics unavailable without residuals.")
                )
            },
            "implications": {
                "practical_meaning": (
                    f"The time series is expected to follow a {trend} trajectory in the forecast period. "
                    f"{'This suggests continued upward momentum.' if trend == 'increasing' else 'This indicates declining values ahead.' if trend == 'decreasing' else 'This points to stable, mean-reverting behavior.'} "
                    f"Forecast reliability is {forecast_confidence.lower()} based on model diagnostics."
                ),
                "recommendations": (
                    f"{'Use forecasts for planning with {forecast_confidence.lower()} confidence.' if forecast_confidence in ['High', 'Moderate'] else 'Exercise caution with forecasts due to high uncertainty.'} "
                    f"Monitor actual values against predictions to validate model performance. "
                    f"{'Consider model retraining if significant deviations occur.' if model_quality in ['Fair', 'Poor'] else 'Model appears well-calibrated for current conditions.'}"
                ),
                "limitations": (
                    "ARIMA assumes that historical patterns will continue and may not account for: "
                    "(1) Structural breaks or regime changes, (2) External shocks or unprecedented events, "
                    "(3) Non-linear relationships, (4) Seasonal patterns (unless explicitly modeled). "
                    f"{'The {d}-level differencing may have removed important trend information.' if d > 1 else ''}"
                ),
                "methodology_notes": (
                    "ARIMA MODEL FRAMEWORK: "
                    f"• AR({p}): Autoregressive component using {p} lagged values as predictors. "
                    f"• I({d}): Integrated component with {d} level(s) of differencing for stationarity. "
                    f"• MA({q}): Moving average component modeling {q} lagged forecast errors. "
                    f"• Model selection typically uses information criteria (AIC/BIC) to balance fit and complexity. "
                    f"• Residual analysis validates model assumptions and identifies potential improvements."
                ),
                "forecasting_insights": (
                    f"PRACTICAL FORECASTING GUIDANCE: "
                    f"The {trend} forecast trend suggests {magnitude_desc} directional bias. "
                    f"Forecast horizon: {len(forecast)} periods ahead. "
                    f"Key assumption: Recent patterns ({max(p,q)} period memory) will persist. "
                    f"Confidence decreases with longer forecast horizons due to error accumulation. "
                    f"{'Consider ensemble methods for improved robustness.' if model_quality in ['Fair', 'Poor'] else 'Single model appears sufficient for current needs.'}"
                )
            }
        }
    except Exception as e:
        return {
            "what_were_testing": "Fitting an ARIMA time series model",
            "purpose": "To forecast future values based on historical patterns",
            "key_ideas": ["ARIMA modeling", "Time series forecasting", "Pattern recognition"],
            "metrics": {"error": str(e)},
            "results": {
                "bottom_line": "Error in model interpretation",
                "confidence_level": "N/A",
                "evidence_strength": "N/A"
            },
            "implications": {
                "practical_meaning": "Unable to interpret model results due to an error.",
                "recommendations": "Check the model specification and data for issues.",
                "limitations": "Error encountered during interpretation."
            }
        }


def interpret_garch_results(model_summary: str, forecast: list) -> Dict[str, Any]:
    """
    Create a structured interpretation of GARCH model results.
    
    Args:
        model_summary (str): Summary of the fitted GARCH model
        forecast (list): List of forecasted volatility values
        
    Returns:
        Dict[str, Any]: Structured interpretation of the GARCH model results
    """
    try:
        # Basic volatility trend analysis
        trend = "increasing" if forecast[-1] > forecast[0] else "decreasing" if forecast[-1] < forecast[0] else "stable"
        
        # Extract GARCH parameters with enhanced parsing
        alpha, beta, omega = 0, 0, 0
        garch_order = (1, 1)  # Default GARCH(1,1)
        
        try:
            # Parse model summary for parameters
            summary_lower = model_summary.lower()
            lines = model_summary.split('\n')
            
            for line in lines:
                line_lower = line.lower()
                if 'omega' in line_lower or 'const' in line_lower:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        try:
                            if part.replace('.', '').replace('-', '').isdigit():
                                omega = float(part)
                                break
                        except:
                            continue
                elif 'alpha' in line_lower and 'alpha[1]' in line_lower:
                    parts = line.split()
                    for part in parts:
                        try:
                            if part.replace('.', '').replace('-', '').isdigit():
                                alpha = float(part)
                                break
                        except:
                            continue
                elif 'beta' in line_lower and 'beta[1]' in line_lower:
                    parts = line.split()
                    for part in parts:
                        try:
                            if part.replace('.', '').replace('-', '').isdigit():
                                beta = float(part)
                                break
                        except:
                            continue
        except:
            pass
        
        # Calculate key GARCH metrics
        persistence = alpha + beta
        unconditional_var = omega / (1 - persistence) if persistence < 1 and persistence > 0 else None
        half_life = -np.log(2) / np.log(persistence) if persistence > 0 and persistence < 1 else None
        
        # Calculate forecast statistics
        forecast_array = np.array(forecast)
        forecast_mean = float(np.mean(forecast_array))
        forecast_std = float(np.std(forecast_array))
        forecast_change = ((forecast[-1] - forecast[0]) / forecast[0]) if forecast[0] != 0 else 0
        
        # Determine volatility characteristics
        if forecast[-1] > 0.03:
            volatility_level = "high"
        elif forecast[-1] > 0.015:
            volatility_level = "moderate"
        else:
            volatility_level = "low"
        
        # Assess volatility persistence
        if persistence > 0.99:
            persistence_desc = "very high (near integrated)"
            persistence_interpretation = "Volatility shocks have extremely long-lasting effects"
        elif persistence > 0.95:
            persistence_desc = "high" 
            persistence_interpretation = "Volatility shocks persist for extended periods"
        elif persistence > 0.85:
            persistence_desc = "moderate"
            persistence_interpretation = "Volatility shocks have moderate persistence"
        elif persistence > 0.5:
            persistence_desc = "low"
            persistence_interpretation = "Volatility shocks decay relatively quickly"
        else:
            persistence_desc = "very low"
            persistence_interpretation = "Volatility shocks have minimal persistence"
        
        # Assess forecast confidence based on model characteristics
        if persistence > 0.98:
            forecast_confidence = "Low"
            confidence_reason = "very high persistence makes long-term forecasts unreliable"
        elif persistence > 0.9:
            forecast_confidence = "Moderate"
            confidence_reason = "high persistence provides reasonable short-term forecast accuracy"
        else:
            forecast_confidence = "High"
            confidence_reason = "moderate persistence allows for reliable volatility forecasting"
        
        # Create detailed bottom line explanation
        magnitude_desc = "substantial" if abs(forecast_change) > 0.2 else "moderate" if abs(forecast_change) > 0.1 else "minimal"
        
        bottom_line_detail = (
            f"The GARCH({garch_order[0]},{garch_order[1]}) model forecasts {trend} volatility with {magnitude_desc} change "
            f"({forecast_change*100:.1f}% from start to end). Current volatility level is {volatility_level} "
            f"({forecast[-1]:.4f}). Persistence is {persistence_desc} (α+β = {persistence:.3f}), indicating "
            f"{persistence_interpretation.lower()}. Forecast confidence is {forecast_confidence.lower()} due to {confidence_reason}."
        )
        
        # GARCH components explanation
        garch_explanation = (
            f"GARCH({garch_order[0]},{garch_order[1]}) MODEL BREAKDOWN: Think of this as a 'volatility recipe' with three key ingredients. "
            f"• ω (omega = {omega:.6f}): The baseline volatility level - like a 'volatility floor'. "
            f"• α (alpha = {alpha:.3f}): How much recent shocks affect current volatility - the 'shock sensitivity'. "
            f"• β (beta = {beta:.3f}): How much past volatility affects current volatility - the 'volatility memory'. "
            f"The model says: 'Today's volatility = baseline + recent shock impact + yesterday's volatility influence.'"
        )
        
        # Volatility clustering explanation
        clustering_explanation = (
            f"VOLATILITY CLUSTERING CONCEPT: GARCH captures the famous 'volatility clustering' phenomenon where "
            f"'large changes tend to be followed by large changes, of either sign, and small changes tend to be followed by small changes.' "
            f"This means periods of high volatility cluster together, and periods of calm cluster together. "
            f"REAL-WORLD EXAMPLE: During market stress (like COVID-19), daily stock movements of 5-10% become common, "
            f"but during calm periods, daily movements of 0.5-1% are typical."
        )

        return {
            "what_were_testing": "Modeling time-varying volatility using GARCH to capture volatility clustering and forecast future volatility.",
            "purpose": "To understand how volatility evolves over time, quantify volatility persistence, and forecast future risk levels.",
            "key_ideas": [
                f"GARCH({garch_order[0]},{garch_order[1]}) models conditional volatility that changes over time.",
                "Volatility clustering: High volatility periods cluster together, low volatility periods cluster together.",
                "α parameter: Measures how sensitive volatility is to recent shocks (ARCH effect).",
                "β parameter: Measures how much past volatility influences current volatility (GARCH effect).",
                f"Persistence (α+β = {persistence:.3f}): Measures how long volatility shocks last.",
                "VOLATILITY CLUSTERING EXPLAINED: Financial markets show periods where large price movements cluster together, followed by periods of relative calm."
            ],
            "metrics": {
                "garch_order": f"GARCH({garch_order[0]},{garch_order[1]})",
                "parameters": {
                    "omega": omega,
                    "alpha": alpha,
                    "beta": beta,
                    "persistence": persistence
                },
                "volatility_trend": trend,
                "forecast_values": forecast,
                "forecast_statistics": {
                    "mean": forecast_mean,
                    "std": forecast_std,
                    "change": forecast_change,
                    "current_level": forecast[-1],
                    "volatility_level": volatility_level
                },
                "garch_explanation": garch_explanation,
                "clustering_explanation": clustering_explanation,
                "persistence_metrics": {
                    "unconditional_variance": unconditional_var,
                    "half_life": half_life,
                    "description": persistence_desc
                }
            },
            "results": {
                "bottom_line": f"Forecasts {trend} volatility",
                "bottom_line_detailed": bottom_line_detail,
                "confidence_level": forecast_confidence,
                "evidence_strength": f"Persistence is {persistence_desc}",
                "volatility_assessment": (
                    f"The model predicts {trend} volatility with {volatility_level} current levels. "
                    f"The {persistence_desc} persistence ({persistence:.3f}) suggests "
                    f"{'volatility shocks will have long-lasting effects' if persistence > 0.9 else 'volatility will mean-revert at a moderate pace' if persistence > 0.7 else 'volatility shocks will decay relatively quickly'}."
                ),
                "shock_impact_analysis": (
                    f"Shock sensitivity (α = {alpha:.3f}): {'High' if alpha > 0.15 else 'Moderate' if alpha > 0.05 else 'Low'} - "
                    f"recent market shocks have {'strong' if alpha > 0.15 else 'moderate' if alpha > 0.05 else 'limited'} impact on current volatility. "
                    f"Volatility memory (β = {beta:.3f}): {'High' if beta > 0.85 else 'Moderate' if beta > 0.7 else 'Low'} - "
                    f"past volatility has {'strong' if beta > 0.85 else 'moderate' if beta > 0.7 else 'limited'} influence on current volatility."
                )
            },
            "implications": {
                "practical_meaning": (
                    f"Expect {volatility_level} volatility levels in the near term, with a {trend} trend. "
                    f"The {persistence_desc} persistence means volatility changes will "
                    f"{'persist for extended periods' if persistence > 0.9 else 'have moderate lasting effects' if persistence > 0.7 else 'decay relatively quickly'}. "
                    f"This {'increases' if trend == 'increasing' else 'decreases' if trend == 'decreasing' else 'maintains'} the risk environment."
                ),
                "recommendations": (
                    f"Risk Management: {'Implement enhanced risk controls due to high/persistent volatility' if persistence > 0.9 and volatility_level == 'high' else 'Monitor volatility levels with standard risk frameworks' if persistence < 0.9 else 'Maintain current risk strategies with periodic reviews'}. "
                    f"Trading Strategy: {'Consider volatility-based position sizing' if persistence > 0.8 else 'Standard position sizing appropriate'}. "
                    f"Portfolio Impact: {'High volatility persistence suggests increased correlation during stress periods' if persistence > 0.9 else 'Moderate volatility dynamics support standard diversification approaches'}."
                ),
                "limitations": (
                    "GARCH model limitations: (1) Assumes volatility clustering follows specific mathematical patterns, "
                    "(2) May not capture structural breaks or regime changes, (3) Assumes constant parameters over time, "
                    "(4) Does not account for leverage effects (asymmetric volatility), (5) May struggle with extreme market events. "
                    f"{'High persistence near 1.0 may indicate model instability or near-unit root behavior.' if persistence > 0.98 else ''}"
                ),
                "methodology_notes": (
                    "GARCH MODEL FRAMEWORK: "
                    f"• σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1) - the core GARCH equation. "
                    f"• ω: Long-run variance component (baseline volatility). "
                    f"• α: ARCH coefficient measuring shock sensitivity (0 ≤ α ≤ 1). "
                    f"• β: GARCH coefficient measuring volatility persistence (0 ≤ β ≤ 1). "
                    f"• Persistence = α + β: Must be < 1 for stationarity. "
                    f"• Model captures volatility clustering without assuming constant variance."
                ),
                "volatility_insights": (
                    f"PRACTICAL VOLATILITY GUIDANCE: "
                    f"Current volatility regime: {volatility_level} ({forecast[-1]:.4f}). "
                    f"Shock decay rate: {'Very slow' if persistence > 0.95 else 'Slow' if persistence > 0.85 else 'Moderate' if persistence > 0.7 else 'Fast'}. "
                    f"Mean reversion: {'Weak' if persistence > 0.95 else 'Moderate' if persistence > 0.85 else 'Strong'}. "
                    f"Forecast horizon reliability: {forecast_confidence} for {len(forecast)} periods ahead. "
                    f"Risk management implication: {'Volatility shocks will persist - prepare for extended periods of elevated risk' if persistence > 0.9 else 'Volatility shows normal mean-reverting behavior - standard risk frameworks apply'}."
                )
            }
        }
    except Exception as e:
        return {
            "what_were_testing": "Modeling time-varying volatility using GARCH",
            "purpose": "To capture and forecast volatility clustering in time series",
            "key_ideas": ["Volatility clustering", "Time-varying risk", "GARCH modeling"],
            "metrics": {"error": str(e)},
            "results": {
                "bottom_line": "Error in model interpretation",
                "confidence_level": "N/A",
                "evidence_strength": "N/A"
            },
            "implications": {
                "practical_meaning": "Unable to interpret volatility patterns due to an error.",
                "recommendations": "Check the model specification and data for issues.",
                "limitations": "Error encountered during interpretation."
            }
        }


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


