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

4. **Deploy to production**:
   ```bash
   npm run deploy
   ```
   This builds the project and deploys it to the production server via rsync

## WebSocket Connection

The viewer automatically connects to the appropriate WebSocket server based on the environment:

- **Development**: Connects to `ws://localhost:8765` 
- **Production**: Connects to `/api` endpoint (with proper WebSocket protocol based on the current domain)

In production, the WebSocket server should be hosted at the `/api` path.

## Development

The project uses:
- **Vite** for fast development and optimized builds
- **Tailwind CSS** for utility-first styling
- **PostCSS** for CSS processing
- **ES6 modules** for clean code organization
