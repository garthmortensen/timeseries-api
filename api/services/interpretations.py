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
            
            # Determine significance levels
            is_stationary = p_value < p_value_threshold
            confidence_level = (1 - p_value) * 100
            
            # Get critical values
            critical_1pct = critical_values.get("1%", -3.75)
            critical_5pct = critical_values.get("5%", -3.0)
            critical_10pct = critical_values.get("10%", -2.63)
            
            # Determine significance level based on critical values
            if adf_stat < critical_1pct:
                significance_level = "1% (highly significant)"
                evidence_strength = "Strong"
            elif adf_stat < critical_5pct:
                significance_level = "5% (significant)"
                evidence_strength = "Moderate"
            elif adf_stat < critical_10pct:
                significance_level = "10% (marginally significant)"
                evidence_strength = "Weak"
            else:
                significance_level = "Not significant"
                evidence_strength = "None"
            
            # Business impact assessment
            if is_stationary:
                business_impact = "Ready for ARIMA modeling"
                recommendation = "Proceed without differencing"
            else:
                business_impact = "Requires preprocessing before modeling"
                recommendation = "Apply differencing or other transformations"
            
            # Create justified conclusions
            stationarity_justification = (
                f"Given that the p-value ({p_value:.4f}) is {'less than' if is_stationary else 'greater than or equal to'} "
                f"the significance threshold ({p_value_threshold}), we {'reject' if is_stationary else 'fail to reject'} "
                f"the null hypothesis that the series has a unit root. Therefore, the series is "
                f"{'stationary' if is_stationary else 'non-stationary'}."
            )
            
            evidence_justification = (
                f"Given that the ADF test statistic ({adf_stat:.4f}) is {'more negative than' if adf_stat < critical_5pct else 'less negative than'} "
                f"the 5% critical value ({critical_5pct:.4f}), we have {evidence_strength.lower()} evidence against the presence of a unit root."
            )
            
            critical_value_justification = (
                f"Given that the test statistic ({adf_stat:.4f}) {'passes' if adf_stat < critical_5pct else 'fails'} "
                f"the 5% critical value threshold ({critical_5pct:.4f}), the test result is {significance_level}."
            )
            
            interpretations[series_name] = {
                "executive_summary": {
                    "bottom_line": "Stationary" if is_stationary else "Non-stationary",
                    "confidence": f"{confidence_level:.1f}%",
                    "business_impact": business_impact,
                    "recommendation": recommendation,
                    "justification": stationarity_justification
                },
                "key_findings": {
                    "adf_statistic": {
                        "value": adf_stat,
                        "interpretation": f"{evidence_strength} evidence against unit root",
                        "justification": evidence_justification
                    },
                    "p_value": {
                        "value": p_value,
                        "interpretation": f"{'Significant' if p_value < p_value_threshold else 'Not significant'} at {p_value_threshold*100}% level",
                        "justification": (
                            f"Given that the p-value ({p_value:.4f}) is {'less than' if p_value < p_value_threshold else 'greater than or equal to'} "
                            f"{p_value_threshold}, the result is {'statistically significant' if p_value < p_value_threshold else 'not statistically significant'} "
                            f"at the {p_value_threshold*100}% significance level."
                        )
                    },
                    "critical_values": {
                        "test_performance": significance_level,
                        "statistical_significance": f"Passes {significance_level} threshold",
                        "justification": critical_value_justification
                    }
                },
                "technical_details": {
                    "statistical_framework": {
                        "null_hypothesis": "Unit root is present (series is non-stationary)",
                        "alternative_hypothesis": "No unit root (series is stationary)",
                        "test_statistic": adf_stat,
                        "decision_rule": f"Reject H0 if p-value < {p_value_threshold}",
                        "decision_justification": (
                            f"Given that our decision rule is to reject H0 when p-value < {p_value_threshold}, "
                            f"and our observed p-value is {p_value:.4f}, we {'reject' if is_stationary else 'fail to reject'} "
                            f"the null hypothesis of a unit root."
                        )
                    },
                    "methodology_notes": {
                        "test_type": "Augmented Dickey-Fuller test",
                        "assumption": "Linear trend, constant variance",
                        "interpretation": "More negative test statistics indicate stronger evidence against unit roots"
                    },
                    "raw_statistics": {
                        "adf_statistic": adf_stat,
                        "p_value": p_value,
                        "critical_values": critical_values,
                        "test_result": "Reject H0" if is_stationary else "Fail to reject H0"
                    }
                },
                "background_context": {
                    "what_is_stationarity": (
                        "A stationary time series has constant mean, variance, and autocorrelation structure over time. "
                        "This means the statistical properties don't change as time progresses."
                    ),
                    "what_is_unit_root": (
                        "A unit root means the series has permanent memory - shocks have lasting effects and the series "
                        "doesn't return to a long-term mean. Like a random walk where each step depends entirely on "
                        "the previous step plus some random change."
                    ),
                    "why_it_matters": (
                        "Stationarity is crucial for time series modeling. Most statistical models assume constant "
                        "relationships over time. Non-stationary data can lead to spurious regression results and "
                        "unreliable forecasts."
                    ),
                    "test_limitations": (
                        "ADF test may not detect all forms of non-stationarity, particularly structural breaks, "
                        "non-linear trends, or seasonal patterns. Test power can be low against certain alternatives."
                    ),
                    "real_world_example": (
                        "Stock prices often have unit roots - if Apple stock jumps from $150 to $160 on good news, "
                        "it doesn't automatically drift back to $150. The new level becomes the new baseline for future movements."
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
    Interpret ARIMA model results using Pyramid Principle structure.
    
    Args:
        model_summary (str): Summary of the fitted ARIMA model
        forecast (list): List of forecasted values
        residuals (list, optional): List of model residuals. Defaults to None.
        
    Returns:
        Dict[str, Any]: Structured interpretation following Pyramid Principle
    """
    try:
        # Extract model parameters
        p, d, q = 0, 0, 0
        if "ARIMA(" in model_summary and ")" in model_summary:
            try:
                order_part = model_summary.split("ARIMA(")[1].split(")")[0]
                p, d, q = map(int, order_part.split(","))
            except:
                pass
        
        # Calculate forecast metrics
        forecast_array = np.array(forecast)
        forecast_mean = float(np.mean(forecast_array))
        forecast_std = float(np.std(forecast_array))
        
        # Determine trend direction
        if len(forecast) > 1:
            trend_direction = "increasing" if forecast[-1] > forecast[0] else "decreasing" if forecast[-1] < forecast[0] else "stable"
            magnitude_change = abs((forecast[-1] - forecast[0]) / forecast[0]) if forecast[0] != 0 else 0
        else:
            trend_direction = "stable"
            magnitude_change = 0
        
        # Model quality assessment
        model_quality = "Unknown"
        accuracy_metrics = {}
        if residuals is not None:
            residuals_array = np.array(residuals)
            residuals_array = residuals_array[~np.isnan(residuals_array)]
            
            if len(residuals_array) > 0:
                mae = float(np.mean(np.abs(residuals_array)))
                rmse = float(np.sqrt(np.mean(residuals_array**2)))
                accuracy_metrics = {
                    "mae": mae,
                    "rmse": rmse,
                    "mean_error": float(np.mean(residuals_array))
                }
                
                if mae < 0.01:
                    model_quality = "Excellent"
                elif mae < 0.05:
                    model_quality = "Good"
                elif mae < 0.10:
                    model_quality = "Fair"
                else:
                    model_quality = "Poor"
        
        # Business impact assessment
        if magnitude_change > 0.1:
            business_impact = f"Substantial {trend_direction} movement expected"
            recommendation = f"Plan for {magnitude_change*100:.1f}% change in forecast period"
        elif magnitude_change > 0.05:
            business_impact = f"Moderate {trend_direction} movement expected"
            recommendation = "Monitor closely for trend continuation"
        else:
            business_impact = "Stable behavior expected"
            recommendation = "Maintain current operational assumptions"
        
        # Forecast confidence
        cv = forecast_std / abs(forecast_mean) if abs(forecast_mean) > 1e-6 else 0
        if cv < 0.02:
            forecast_confidence = "High"
        elif cv < 0.05:
            forecast_confidence = "Moderate"
        else:
            forecast_confidence = "Low"

        # Create justified conclusions
        trend_justification = (
            f"Given that the forecast values change from {forecast[0]:.4f} to {forecast[-1]:.4f} "
            f"({'an increase' if forecast[-1] > forecast[0] else 'a decrease' if forecast[-1] < forecast[0] else 'no change'} "
            f"of {magnitude_change*100:.1f}%), the ARIMA model forecasts a {trend_direction} trend."
        )
        
        quality_justification = (
            f"Given that the Mean Absolute Error is {accuracy_metrics.get('mae', 'N/A')}, "
            f"the model quality is assessed as {model_quality}. "
            f"{'This indicates excellent predictive accuracy with very small forecast errors.' if model_quality == 'Excellent' else 'This indicates good predictive accuracy with acceptable forecast errors.' if model_quality == 'Good' else 'This indicates moderate predictive accuracy with noticeable forecast errors.' if model_quality == 'Fair' else 'This indicates poor predictive accuracy with large forecast errors.' if model_quality == 'Poor' else 'Model quality cannot be determined without residual data.'}"
        ) if accuracy_metrics else (
            "Given that no residual data is available, model quality cannot be quantitatively assessed. "
            "Quality evaluation requires comparing predicted values against actual values."
        )
        
        confidence_justification = (
            f"Given that the coefficient of variation (forecast standard deviation / forecast mean) is {cv:.4f}, "
            f"the forecast confidence is {forecast_confidence}. "
            f"{'This indicates low uncertainty in the forecast values.' if forecast_confidence == 'High' else 'This indicates moderate uncertainty in the forecast values.' if forecast_confidence == 'Moderate' else 'This indicates high uncertainty in the forecast values.'}"
        )
        
        business_impact_justification = (
            f"Given that the magnitude of change is {magnitude_change*100:.1f}% "
            f"({'substantial' if magnitude_change > 0.1 else 'moderate' if magnitude_change > 0.05 else 'minimal'}), "
            f"the business impact is {business_impact.lower()}. "
            f"{'Significant planning and resource allocation adjustments may be required.' if magnitude_change > 0.1 else 'Some operational adjustments may be needed.' if magnitude_change > 0.05 else 'Current operations can likely continue with minimal adjustments.'}"
        )

        return {
            "executive_summary": {
                "bottom_line": f"ARIMA({p},{d},{q}) forecasts {trend_direction} trend",
                "business_impact": business_impact,
                "recommendation": recommendation,
                "justification": trend_justification
            },
            "key_findings": {
                "forecast_trend": {
                    "direction": trend_direction,
                    "magnitude": f"{magnitude_change*100:.1f}% change",
                    "confidence": forecast_confidence,
                    "justification": trend_justification
                },
                "model_performance": {
                    "quality": model_quality,
                    "accuracy": f"MAE: {accuracy_metrics.get('mae', 'N/A')}" if accuracy_metrics else "No residuals provided",
                    "justification": quality_justification
                },
                "forecast_statistics": {
                    "mean_forecast": forecast_mean,
                    "forecast_range": f"{min(forecast):.4f} to {max(forecast):.4f}",
                    "volatility": forecast_std,
                    "justification": confidence_justification
                },
                "business_impact": {
                    "assessment": business_impact,
                    "justification": business_impact_justification
                }
            },
            "technical_details": {
                "model_specification": {
                    "order": f"ARIMA({p},{d},{q})",
                    "ar_component": f"Uses {p} lagged values" if p > 0 else "No autoregressive component",
                    "differencing": f"{d} level(s) of differencing applied" if d > 0 else "No differencing needed",
                    "ma_component": f"Models {q} forecast error terms" if q > 0 else "No moving average component",
                    "justification": (
                        f"Given the model specification ARIMA({p},{d},{q}): "
                        f"{'The AR(' + str(p) + ') component uses ' + str(p) + ' lagged values to capture autocorrelation patterns. ' if p > 0 else 'No autoregressive component suggests the series does not significantly depend on its own past values. '}"
                        f"{'The I(' + str(d) + ') component applies ' + str(d) + ' level(s) of differencing to achieve stationarity. ' if d > 0 else 'No differencing needed indicates the series is already stationary. '}"
                        f"{'The MA(' + str(q) + ') component models ' + str(q) + ' forecast error terms to capture short-term dependencies.' if q > 0 else 'No moving average component suggests forecast errors are independent.'}"
                    )
                },
                "forecast_mechanics": {
                    "prediction_method": "Linear combination of past values and errors",
                    "memory_length": f"{max(p,q)} periods" if max(p,q) > 0 else "1 period",
                    "stationarity": "Achieved through differencing" if d > 0 else "Series already stationary"
                },
                "accuracy_metrics": accuracy_metrics if accuracy_metrics else {"note": "Residuals not provided for accuracy assessment"}
            },
            "background_context": {
                "what_is_arima": (
                    "ARIMA combines three components: AutoRegressive (AR) - uses past values to predict future, "
                    "Integrated (I) - differences the data to remove trends, and Moving Average (MA) - uses past forecast errors. "
                    "It's like a sophisticated pattern recognition system that learns from historical behavior."
                ),
                "why_it_matters": (
                    "ARIMA is fundamental for time series forecasting because it captures the underlying patterns "
                    "in data - trends, seasonality, and dependencies. It's widely used in economics, finance, "
                    "and operations for planning and decision-making."
                ),
                "model_assumptions": (
                    "ARIMA assumes that historical patterns will continue into the future, that the relationship "
                    "between variables is linear, and that the residuals are normally distributed with constant variance. "
                    "It works best when the underlying process is relatively stable."
                ),
                "limitations": (
                    "ARIMA may struggle with structural breaks, non-linear relationships, or unprecedented events. "
                    "It assumes past patterns continue and may not adapt quickly to regime changes. "
                    "External factors not captured in the time series can affect forecast accuracy."
                ),
                "interpretation_guide": (
                    f"AR({p}): The model uses the last {p} values to make predictions - like saying 'tomorrow will be similar to recent days'. "
                    f"I({d}): The data was differenced {d} time(s) to remove trends - like focusing on changes rather than levels. "
                    f"MA({q}): The model corrects for the last {q} forecast errors - like learning from recent mistakes."
                )
            }
        }
    except Exception as e:
        return {
            "executive_summary": {
                "bottom_line": "Error in ARIMA interpretation",
                "business_impact": "Unable to assess forecast implications",
                "recommendation": "Review model inputs and specification",
                "justification": f"Given that an error occurred during interpretation ({str(e)}), conclusions cannot be drawn from the ARIMA results."
            },
            "key_findings": {
                "error": str(e),
                "justification": f"Given the error: {str(e)}, key findings cannot be determined."
            },
            "technical_details": {
                "error": f"Interpretation failed: {str(e)}"
            },
            "background_context": {
                "error": "Context unavailable due to processing error"
            }
        }


def interpret_garch_results(model_summary: str, forecast: list) -> Dict[str, Any]:
    """
    Interpret GARCH model results using Pyramid Principle structure.
    
    Args:
        model_summary (str): Summary of the fitted GARCH model
        forecast (list): List of forecasted volatility values
        
    Returns:
        Dict[str, Any]: Structured interpretation following Pyramid Principle
    """
    try:
        # Extract GARCH parameters with enhanced parsing
        alpha, beta, omega = 0, 0, 0
        garch_order = (1, 1)  # Default GARCH(1,1)
        
        try:
            # Parse model summary for parameters
            lines = model_summary.split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'omega' in line_lower or 'const' in line_lower:
                    parts = line.split()
                    for part in parts:
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
        
        # Determine trend direction
        if forecast[-1] > forecast[0]:
            trend_direction = "increasing"
        elif forecast[-1] < forecast[0]:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Assess volatility persistence
        if persistence > 0.99:
            persistence_desc = "very high"
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
        
        # Assess forecast confidence
        if persistence > 0.98:
            forecast_confidence = "Low"
        elif persistence > 0.9:
            forecast_confidence = "Moderate"
        else:
            forecast_confidence = "High"
        
        # Business impact assessment
        magnitude_desc = "substantial" if abs(forecast_change) > 0.2 else "moderate" if abs(forecast_change) > 0.1 else "minimal"
        
        if volatility_level == "high":
            business_impact = "Elevated risk environment requiring enhanced monitoring"
            recommendation = "Implement risk controls and position sizing adjustments"
        elif volatility_level == "moderate":
            business_impact = "Balanced risk environment with standard volatility"
            recommendation = "Maintain current risk management frameworks"
        else:
            business_impact = "Low risk environment with calm market conditions"
            recommendation = "Consider opportunities during stable periods"

        # Create justified conclusions
        trend_justification = (
            f"Given that the forecast volatility changes from {forecast[0]:.4f} to {forecast[-1]:.4f} "
            f"({'an increase' if forecast[-1] > forecast[0] else 'a decrease' if forecast[-1] < forecast[0] else 'no change'} "
            f"of {abs(forecast_change)*100:.1f}%), the GARCH model forecasts {trend_direction} volatility."
        )
        
        persistence_justification = (
            f"Given that the persistence parameter (α + β = {persistence:.3f}) is "
            f"{'greater than 0.99' if persistence > 0.99 else 'greater than 0.95' if persistence > 0.95 else 'greater than 0.85' if persistence > 0.85 else 'greater than 0.5' if persistence > 0.5 else 'less than or equal to 0.5'}, "
            f"the volatility persistence is {persistence_desc}. {persistence_interpretation}."
        )
        
        volatility_level_justification = (
            f"Given that the final forecast volatility ({forecast[-1]:.4f}) is "
            f"{'greater than 0.03' if forecast[-1] > 0.03 else 'greater than 0.015' if forecast[-1] > 0.015 else 'less than or equal to 0.015'}, "
            f"the volatility level is classified as {volatility_level}. This indicates "
            f"{'elevated risk conditions requiring enhanced monitoring' if volatility_level == 'high' else 'balanced risk conditions with standard volatility patterns' if volatility_level == 'moderate' else 'calm market conditions with low risk levels'}."
        )
        
        confidence_justification = (
            f"Given that the persistence parameter ({persistence:.3f}) is "
            f"{'greater than 0.98' if persistence > 0.98 else 'greater than 0.9' if persistence > 0.9 else 'less than or equal to 0.9'}, "
            f"the forecast confidence is {forecast_confidence}. "
            f"{'High persistence reduces forecast confidence as volatility shocks have very long-lasting effects' if persistence > 0.98 else 'Moderate persistence provides reasonable forecast confidence with some uncertainty' if persistence > 0.9 else 'Low persistence provides high forecast confidence as volatility returns to normal levels quickly'}."
        )
        
        business_impact_justification = (
            f"Given that the volatility level is {volatility_level} and the magnitude of change is {magnitude_desc} "
            f"({abs(forecast_change)*100:.1f}%), the business impact is {business_impact.lower()}. "
            f"{'This requires immediate risk management attention and position sizing adjustments' if volatility_level == 'high' else 'This suggests maintaining current risk management frameworks while monitoring for changes' if volatility_level == 'moderate' else 'This presents opportunities to consider increased positions during stable periods'}."
        )

        return {
            "executive_summary": {
                "bottom_line": f"GARCH({garch_order[0]},{garch_order[1]}) forecasts {trend_direction} volatility at {volatility_level} levels",
                "business_impact": business_impact,
                "recommendation": recommendation,
                "justification": trend_justification
            },
            "key_findings": {
                "volatility_trend": {
                    "direction": trend_direction,
                    "magnitude": f"{abs(forecast_change)*100:.1f}% change",
                    "current_level": f"{forecast[-1]:.4f} ({volatility_level})",
                    "justification": trend_justification
                },
                "persistence_metrics": {
                    "persistence": persistence,
                    "description": f"{persistence_desc} persistence",
                    "interpretation": persistence_interpretation,
                    "justification": persistence_justification
                },
                "forecast_confidence": {
                    "level": forecast_confidence,
                    "reliability": f"Based on {persistence_desc} persistence pattern",
                    "justification": confidence_justification
                },
                "volatility_level": {
                    "classification": volatility_level,
                    "justification": volatility_level_justification
                },
                "business_impact": {
                    "assessment": business_impact,
                    "justification": business_impact_justification
                }
            },
            "technical_details": {
                "model_specification": {
                    "order": f"GARCH({garch_order[0]},{garch_order[1]})",
                    "omega": f"{omega:.6f} (baseline volatility)",
                    "alpha": f"{alpha:.3f} (shock sensitivity)",
                    "beta": f"{beta:.3f} (volatility memory)",
                    "justification": (
                        f"Given the model parameters: omega (ω = {omega:.6f}) represents the baseline volatility floor, "
                        f"alpha (α = {alpha:.3f}) measures sensitivity to recent shocks, and beta (β = {beta:.3f}) "
                        f"captures volatility memory. The model equation σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1) shows that "
                        f"{'recent shocks dominate' if alpha > beta else 'past volatility dominates' if beta > alpha else 'shocks and memory are balanced'} "
                        f"in determining future volatility."
                    )
                },
                "volatility_mechanics": {
                    "equation": "σ²(t) = ω + α·ε²(t-1) + β·σ²(t-1)",
                    "persistence": f"α + β = {persistence:.3f}",
                    "half_life": f"{half_life:.1f} periods" if half_life else "N/A",
                    "unconditional_variance": f"{unconditional_var:.6f}" if unconditional_var else "N/A",
                    "justification": (
                        f"Given the volatility equation components: "
                        f"{'The high persistence (α + β = ' + f'{persistence:.3f}' + ') indicates volatility clustering effects persist for extended periods' if persistence > 0.9 else 'The moderate persistence indicates balanced volatility dynamics' if persistence > 0.5 else 'The low persistence indicates rapid return to baseline volatility'}. "
                        f"{'The half-life of ' + f'{half_life:.1f}' + ' periods shows how long it takes for volatility shocks to decay by half' if half_life else 'Half-life cannot be calculated due to parameter constraints'}."
                    )
                },
                "forecast_statistics": {
                    "mean_forecast": forecast_mean,
                    "forecast_volatility": forecast_std,
                    "forecast_range": f"{min(forecast):.4f} to {max(forecast):.4f}",
                    "total_change": f"{forecast_change*100:.1f}%"
                }
            },
            "background_context": {
                "what_is_garch": (
                    "GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models time-varying volatility. "
                    "Unlike traditional models that assume constant volatility, GARCH recognizes that volatility clusters - "
                    "periods of high volatility are followed by high volatility, and calm periods by calm periods. "
                    "It's like a volatility thermostat that adjusts based on recent market stress."
                ),
                "why_it_matters": (
                    "Volatility forecasting is crucial for risk management, option pricing, and portfolio optimization. "
                    "GARCH helps quantify how risk evolves over time, enabling better decision-making about position sizing, "
                    "hedging strategies, and capital allocation. It's widely used in finance because it captures the "
                    "reality that market risk is not constant."
                ),
                "volatility_clustering": (
                    "The famous quote 'large changes tend to be followed by large changes, of either sign, and small changes "
                    "tend to be followed by small changes' captures the essence of volatility clustering. "
                    "During market stress (like COVID-19), daily stock movements of 5-10% become common, "
                    "but during calm periods, daily movements of 0.5-1% are typical."
                ),
                "model_components": (
                    f"Omega (ω = {omega:.6f}): The baseline volatility level - think of it as the 'volatility floor'. "
                    f"Alpha (α = {alpha:.3f}): How sensitive volatility is to recent shocks - the 'shock amplifier'. "
                    f"Beta (β = {beta:.3f}): How much past volatility influences current volatility - the 'memory factor'. "
                    f"The model says: Today's volatility = baseline + recent shock impact + yesterday's volatility influence."
                ),
                "persistence_implications": (
                    f"Persistence (α + β = {persistence:.3f}) measures how long volatility shocks last. "
                    f"High persistence means volatility shocks have long-lasting effects - like ripples in a pond that take "
                    f"time to settle. Low persistence means volatility quickly returns to normal levels after shocks. "
                    f"Values near 1.0 indicate very persistent volatility, while values near 0.5 indicate quick mean reversion."
                ),
                "limitations": (
                    "GARCH assumes volatility clustering follows specific patterns and may not capture: "
                    "(1) Structural breaks or regime changes, (2) Asymmetric volatility (leverage effects), "
                    "(3) Extreme market events outside historical patterns, (4) Non-linear volatility dynamics. "
                    "It works best in relatively stable market regimes with consistent volatility patterns."
                )
            }
        }
    except Exception as e:
        return {
            "executive_summary": {
                "bottom_line": "Error in GARCH interpretation",
                "business_impact": "Unable to assess volatility implications",
                "recommendation": "Review model inputs and specification",
                "justification": f"Given that an error occurred during interpretation ({str(e)}), conclusions cannot be drawn from the GARCH results."
            },
            "key_findings": {
                "error": str(e),
                "justification": f"Given the error: {str(e)}, key findings cannot be determined."
            },
            "technical_details": {
                "error": f"Interpretation failed: {str(e)}"
            },
            "background_context": {
                "error": "Context unavailable due to processing error"
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
        total_spillover = spillover_results.get("total_spillover_index", spillover_results.get("total", 0.0))
        
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


