# API Pipeline

```mermaid
flowchart TD
    Start([Start Pipeline]) --> InputParams[Extract Parameters\nfrom Input]
    
    subgraph "Data Acquisition"
        InputParams --> SourceCheck{Synthetic or\nActual Data?}
        SourceCheck -->|Synthetic| GenData[Generate Synthetic Data\nRandom Walk Model]
        SourceCheck -->|Actual| FetchData[Fetch Market Data\nfrom Yahoo Finance]
        GenData --> PriceData[Price Data DataFrame]
        FetchData --> PriceData
    end
    
    subgraph "Data Transformation"
        PriceData --> Returns[Convert to Log Returns\n"Academic studies emphasize importance"]
        Returns --> Stationarity[Test for Stationarity\nADF Test]
        Stationarity --> Scale[Scale for GARCH Modeling]
    end
    
    subgraph "ARIMA Modeling"
        Scale --> ARIMA[Run ARIMA Model\n"First model conditional mean"]
        ARIMA --> ARIMAOutputs[ARIMA Outputs]
        ARIMAOutputs --> ARIMASummary[Model Summary]
        ARIMAOutputs --> ARIMAForecast[Return Forecast]
        ARIMAOutputs --> ARIMAResiduals[Extract Residuals]
        ARIMASummary --> ARIMAInterpret[Generate Human-Readable\nInterpretation]
    end
    
    subgraph "GARCH Modeling"
        ARIMAResiduals --> GARCH[Run GARCH Model\n"Model volatility clustering"]
        GARCH --> GARCHOutputs[GARCH Outputs]
        GARCHOutputs --> GARCHSummary[Model Summary]
        GARCHOutputs --> GARCHForecast[Volatility Forecast]
        GARCHSummary --> GARCHInterpret[Generate Human-Readable\nInterpretation]
    end
    
    subgraph "Optional Analysis"
        Returns --> SpilloverCheck{Spillover\nEnabled?}
        SpilloverCheck -->|Yes| Spillover[Analyze Volatility\nSpillover Effects]
        SpilloverCheck -->|No| Skip[Skip Spillover Analysis]
    end
    
    ARIMAForecast --> Compile[Compile Results]
    ARIMAInterpret --> Compile
    GARCHForecast --> Compile
    GARCHInterpret --> Compile
    Stationarity --> Compile
    Spillover --> Compile
    Skip --> Compile
    
    Compile --> Response[Return API Response\nwith All Results]
    Response --> End([End Pipeline])
    
    classDef academic fill:#f9f,stroke:#333,stroke-width:2px
    classDef process fill:#bbf,stroke:#333,stroke-width:1px
    classDef data fill:#dfd,stroke:#333,stroke-width:1px
    classDef decision fill:#ffd,stroke:#333,stroke-width:1px
    
    class Returns,ARIMA,GARCH academic
    class GenData,FetchData,Scale,Stationarity,Spillover process
    class PriceData,ARIMAOutputs,GARCHOutputs,Response data
    class SourceCheck,SpilloverCheck decision
```mermaid
