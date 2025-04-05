# Architecture Decision Record

## ADR-001: Where to add statistical interpretations?

### Context
- Statistical interpretations could be added to either:
    1. Core computational python package
    - pros: centralized statistical logic, interpretation close to logic making them consistent and sound
    - cons: package scope creep
    2. API layer
    - pros: seperates domain/business logic from computation and presentation logic, centralized for all downstream. 
    - cons: depedency on package, api is more than just data handoff

### Decision and consequences

Decision: Add to API

Consequences: maintains seperation of concerns and place interpretation in API. Data layer -> Business layer -> Presentation layer. API is a service that provides complete, consumable info.

## ADR-002: Where to add functionality for sourcing external market data?

### Context
-  API or py package?
    1. Add to API
    - pros: API remains interface with external systems
    - cons: dupes py package's `data_generation()` functionality
    2. Add to py package
    - pros: complete, self contained package
    - cons: adds external dependencies (data provider APIs) to computation-focused package

### Decision and consequences

AD: Add to API.

Consequences: Maintains architectural consistency with ADR-001 (API is interface with external sources). It preserves the py package's computational focus. However, the API will become more complex, and it might be difficult to manage data structures returned from py packages `data_generation()` and API's ~"`get_external_data()`".

## ADR-002: Where to add functionality for sourcing external market data?

### Context

- Find a data structure that works well across the entire stack: the Python package, API, and the frontend.
    1. Nested dictionary by date:
        ```json
        {
        "2023-01-01": {"GME": 150.0, "BYND": 200.0},
        ...
        }
        ```
    - pros: clean, easy lookup by date
    - cons: awkward to load into DataFrames, harder to filter/sort/group

    2. Nested dictionary by symbol:
        ```json
        {
        "GME": {"2023-01-01": 150.0, ...},
        "BYND": {...}
        }
        ```
    - pros: efficient symbol lookups
    - cons: Must be flattened for frontend use?

    3. List of dictionaries:
        ```json
            [
            {"date": "2023-01-01", "symbol": "GME", "price": 150.0},
            {"date": "2023-01-01", "symbol": "BYND", "price": 200.0},
            {"date": "2023-01-02", "symbol": "GME", "price": 149.3}
            ]
        ```
    - pros: easily loaded into Pandas, sortable, groupable, filterable, JSON serializable, API and frontend friendly. List of records - straight out of a database. This seems like a pretty neat datastructure. Nothing interesting about lists, but the universality is neat.
    - cons: more verbose and repetitive

### Decision and consequences

Decision: Use a list of dictionaries with explicit date, symbol, and price keys.

Consequences: Easy integration across Python, API, and frontend. Clean handoff between components. enables use of standard tooling (Pandas, JSON, JS tables) without custom parsing logic.
