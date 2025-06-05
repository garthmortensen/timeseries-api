<!--
# === FILE META OPENING ===
# file: ./timeseries-api/pipeline.md
# role: documentation
# desc: detailed methodology and flowchart documentation for the spillover analysis pipeline
# === FILE META CLOSING ===
-->

# Spillover Methodology

## 1. Convert Prices to Log Returns

Raw price processes are problematic because of non-stationarity. Log returns are better bc:

- Percentage point changes are comparable across different price levels.
- Series is more symmetric and closer to normal distribution.
- Makes volatility more consistent across the series.

## 2. Test for Stationarity (ADF Test)

ADF tests if your data's constant over time. ARIMA and GARCH assume stationarity. A stationary series has:

- Constant mean
- Constant variance
- Autocorrelation

## 3. Scale Data for GARCH Modeling

Scaling standardizes data to prevent numerical issues. For GARCH, it helps the optimization algorithms converge better and produces more stable volatility estimates.

## 4. ARIMA Modeling

- Operations: ARIMA combines three components:
    1. Autoregressive (AR): Models relationship between current and past values.
    2. Integrated (I): Accounts for differencing to make data stationary.
    3. Moving Average (MA): Models the relationship between current and past value error terms.

- Inputs: Stationary time series data
- Outputs:
    - Conditional mean forecasts (expected future values)
    - Model residuals (unexplained movements)
    - Parameter estimates (showing which past lags are most important)

## 5. GARCH Modeling

- Operations: GARCH models volatility clustering by:
    - Models how past volatility affects current volatility (GARCH terms)
    - Models how past shocks affect current volatility (ARCH terms)
    - Accounting for fat-tailed distributions (common in financial data)

- Inputs: ARIMA model residuals
- Outputs:
    - Volatility forecasts (predicted future uncertainty)
    - Volatility parameters (showing persistence of volatility)

## 6. Spillover Analysis

Operations: Analyzes how movements in one time series affect others by:
    - Measures how shocks to one variable explain forecast error variance in others
    - Captures both direct and indirect transmission effects

Inputs: Returns data and GARCH volatility estimates
Outputs:
    - Total spillover index (system-wide interconnectedness)
    - Directional spillover (which variables are net transmitters or receivers of shocks)
    - Pairwise spillover (specific relationship between any two variables)

## Methodology Flowchart

```mermaid
flowchart TD
    Start([Pipeline Starting]) --> InputParams[Extract Config Parameters]

    subgraph "ETL"
        subgraph "Data Acquisition"
            InputParams --> SourceCheck{Pull Synthetic or<br>Actual Data?}
            SourceCheck -->|Synthetic| GenData[Generate Synthetic Data<br> Random Walk Model]
            SourceCheck -->|Actual| FetchData[Fetch Yahoo Finance Market Data]
            GenData --> PriceData[Prices DataFrame]
            FetchData --> PriceData
        end
        
        subgraph "Data Transformation"
            PriceData --> Returns[Convert to Log Returns]
            Returns --> Stationarity[Stationarity ADF Test]
            Stationarity --> Scale[Scale for GARCH Modeling]
        end
    end
    
    subgraph "ARIMA Modeling"
        Scale --> ARIMA[Run ARIMA Model -<br>First model conditional mean]
        ARIMA --> ARIMAOutputs[ARIMA Outputs]
        ARIMAOutputs --> ARIMASummary[Model Summary]
        ARIMAOutputs --> ARIMAForecast[Return Forecast]
        ARIMAOutputs --> ARIMAResiduals[Extract Residuals]
        ARIMASummary --> ARIMAInterpret[Generate human-readable<br>Interpretation]
    end
    
    subgraph "GARCH Modeling"
        ARIMAResiduals --> GARCH[Run GARCH Model -<br>Models volatility clustering<br>t-distribution for fat tails]
        GARCH --> GARCHOutputs[GARCH Outputs]
        GARCHOutputs --> GARCHSummary[Model Summary]
        GARCHOutputs --> GARCHForecast[Volatility Forecast]
        GARCHSummary --> GARCHInterpret[Generate human-readable<br>Interpretation]
    end
    
    subgraph "Spillover Analysis"
        Returns --> SpilloverCheck{Spillover<br>Enabled?}
        GARCHOutputs --> SpilloverCheck
        SpilloverCheck -->|Yes| Spillover[Analyze Volatility<br>Spillover Effects]
        SpilloverCheck -->|No| Skip[Skip Spillover Analysis]
    end
    
    ARIMAForecast --> Compile[Compile Results]
    ARIMAInterpret --> Compile
    GARCHForecast --> Compile
    GARCHInterpret --> Compile
    Stationarity --> Compile
    Spillover --> Compile
    Skip --> Compile
    
    Compile --> Response[Return API Response<br>with All Results]
    Response --> End([Pipeline Finished])
 
    classDef academic fill:#f9f,stroke:#333,stroke-width:2px
    classDef process fill:#bbf,stroke:#333,stroke-width:1px
    classDef data fill:#dfd,stroke:#333,stroke-width:1px
    classDef decision fill:#ffd,stroke:#333,stroke-width:1px
    
    class Returns,ARIMA,GARCH academic
    class GenData,FetchData,Scale,Stationarity,Spillover process
    class PriceData,ARIMAOutputs,GARCHOutputs,Response data
    class SourceCheck,SpilloverCheck decision
```mermaid
