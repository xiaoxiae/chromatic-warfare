You are implementing a turn-based strategy game system designed for bot competitions and livestream tournaments. This is a multi-component system with a Python WebSocket server, web-based visualization, and bot client libraries.

## Project Context
You're building a competitive gaming platform where:
- Bots connect via WebSocket to play turn-based strategy games
- Games are played on directed planar graphs where players control vertices
- The system supports multiple concurrent games for tournament play
- A web interface provides real-time visualization for spectators
- The architecture prioritizes real-time performance and scalability

## Core Game Mechanics
- Players control vertices on a directed graph and move units between adjacent vertices
- Each vertex has a weight that determines unit generation per turn
- Combat occurs when players target the same vertex, with probabilistic resolution
- Players are eliminated when they control no vertices
- Games end when ≤1 player remains or maximum turns are reached

## Technology Stack & Architecture
- **Server**: Python with asyncio/websockets for real-time communication
- **Client Library**: Python async WebSocket client for bot development
- **Web Interface**: HTML/JavaScript with D3.js for graph visualization
- **Data Format**: JSON for all network communication and file storage
- **Testing**: pytest with comprehensive unit and integration tests

## Code Style & Patterns
- Use Python 3.8+ features including dataclasses, type hints, and async/await
- Follow PEP 8 naming conventions: snake_case for functions/variables, PascalCase for classes
- Prefer composition over inheritance; use dependency injection for testability
- Implement proper error handling with specific exception types
- Use immutable data patterns where possible to prevent state corruption
- All classes should have clear single responsibilities

## Key Design Principles
- **Incremental Development**: Each step builds on previous work without breaking changes
- **Test-Driven**: Write comprehensive unit tests for all business logic
- **Real-time Performance**: Optimize for low-latency message processing
- **Fault Tolerance**: Handle network failures, invalid input, and edge cases gracefully
- **Scalability**: Design for multiple concurrent games and many connected clients
- **Clean Integration**: Every component must integrate cleanly with existing code

## Naming Conventions
- Game entities: `GameState`, `Player`, `Vertex`, `Edge`, `Graph`
- Commands: `MoveCommand`, `CommandValidator`, `CommandParser`
- Network: `WebSocketServer`, `GameSession`, `MessageHandler`
- Game logic: `MovementEngine`, `UnitGenerator`, `ConflictResolver`, `TurnManager`
- Client: `GameBot`, `ConnectionManager`
- Files: Use descriptive names like `game_state.py`, `movement_engine.py`

## Error Handling Strategy
- Create specific exception classes for different error types
- Always validate input at system boundaries (network, file I/O)
- Log errors with sufficient context for debugging
- Fail fast for programming errors, recover gracefully for expected failures
- Provide clear error messages to bot developers

## Testing Requirements
- Unit tests for all business logic classes and methods
- Integration tests for component interactions
- Mock external dependencies (WebSockets, file I/O) in unit tests
- Include edge cases and error scenarios in test coverage
- Use descriptive test names that explain the scenario being tested

## Performance Considerations
- Minimize object creation in hot paths (game state updates, message processing)
- Cache expensive computations (graph adjacency, serialized states)
- Use efficient data structures (sets for lookups, deques for queues)
- Batch operations where possible to reduce overhead
- Profile and optimize critical paths

## Dependencies to Use
- `websockets` for WebSocket server/client
- `dataclasses` for clean data structures
- `typing` for type hints
- `asyncio` for async programming
- `json` for serialization
- `pytest` for testing
- `logging` for debug/monitoring
- Avoid heavy frameworks; keep dependencies minimal

## Integration Requirements
- Each implementation step must integrate with and extend previous work
- Maintain backward compatibility with existing interfaces
- Import and use classes from previous steps rather than reimplementing
- Ensure all code paths are connected and no orphaned code exists
- Test integration points thoroughly

## Remember
- This is a real-time competitive gaming system - prioritize performance and reliability
- Bot developers will use your APIs - make them intuitive and well-documented
- The system will handle multiple concurrent games - design for concurrency from the start
- Tournament organizers will use the CLI tools - make them robust and user-friendly
- Spectators will watch via the web interface - ensure smooth, responsive visualization

Focus on building production-quality code that handles edge cases gracefully and performs well under load.
