# Chromatic Warfare Viewer

A web-based viewer for Chromatic Warfare.


## Setup Instructions

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```
   This will start the Vite dev server at `http://localhost:3000`

3. **Build for production**:
   ```bash
   npm run build
   ```
   Output will be in the `dist/` directory

## Usage

The viewer connects to a WebSocket server (default: `ws://localhost:8765`) and displays:

- Live game state with animated unit movements
- Player list with real-time status updates
- Game log with turn-by-turn actions
- Canvas-based graph visualization
- Combat animations and unit generation effects

You can specify a game ID using the URL parameter: `?game=your-game-id`

## Development

The project uses:
- **Vite** for fast development and optimized builds
- **Tailwind CSS** for utility-first styling
- **PostCSS** for CSS processing
- **ES6 modules** for clean code organization

All the original game logic and animation systems are preserved while providing a modern development experience.
