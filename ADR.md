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
