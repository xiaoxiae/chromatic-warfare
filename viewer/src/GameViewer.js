export class GameViewer {
    constructor() {
        this.socket = null;
        this.gameState = null;
        this.gameId = this.getGameIdFromURL();
        this.canvas = document.getElementById("gameCanvas");
        this.ctx = this.canvas.getContext("2d");
        this.playerColors = {};
        this.colorPalette = [
            "#dc322f",
            "#268bd2",
            "#859900",
            "#b58900",
            "#6c71c4",
            "#2aa198",
            "#cb4b16",
            "#d33682",
        ];
        this.colorIndex = 0;
        this.previousPlayers = new Map(); // Track previous player states

        // Connect settings UI to this viewer instance
        if (typeof window.connectSettingsToGameViewer === "function") {
            window.connectSettingsToGameViewer(this);
        }

        // Animation system
        this.isAnimating = false;
        this.animationData = null;
        this.lastRenderTime = 0;
        this.animationStartTime = 0;
        this.currentPhase = "idle"; // 'movement', 'state_change', 'generation', 'idle'

        // Animation states
        this.displayedVertices = new Map(); // Current displayed state
        this.targetVertices = new Map(); // Target state after animations
        this.movingUnits = []; // Units currently animating
        this.unitGenerationAnimations = []; // +Weight text animations

        this.serverTurnDuration = 1000; // Default 1 second in milliseconds
        this.maxTurns = 24; // Default max turns

        this.setupCanvas();
        this.updateUIWithGameId();
        this.connectToServer();
        this.setupResizeHandler();
        this.startRenderLoop();
    }

    sendGameSettings(maxTurns, turnDuration) {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
            this.addLogEntry(
                "Cannot send settings - not connected to server",
                "system",
            );
            return;
        }

        // Send max turns setting
        this.socket.send(
            JSON.stringify({
                type: "server_command",
                command: "set_turns",
                game_id: this.gameId,
                max_turns: maxTurns,
            }),
        );

        // Send turn duration setting
        this.socket.send(
            JSON.stringify({
                type: "server_command",
                command: "set_duration",
                game_id: this.gameId,
                duration: turnDuration,
            }),
        );

        this.addLogEntry(
            `Settings applied: ${maxTurns} max turns, ${turnDuration}s duration`,
            "system",
        );
    }

    sendMapSettings(mapType, width, height, weightMin, weightMax, removeProb) {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
            this.addLogEntry(
                "Cannot send settings - not connected to server",
                "system",
            );
            return;
        }

        // Build map command with options
        const options = [];
        if (weightMin !== undefined && weightMax !== undefined) {
            options.push(`weight_min=${weightMin}`);
            options.push(`weight_max=${weightMax}`);
        }
        options.push(`remove_prob=${removeProb}`);
        options.push(`maze_width=${width}`);
        options.push(`maze_height=${height}`);

        this.socket.send(
            JSON.stringify({
                type: "server_command",
                command: "map",
                game_id: this.gameId,
                map_type: mapType,
                options: options,
            }),
        );

        this.addLogEntry(
            `Map settings applied: ${mapType} ${width}x${height}`,
            "system",
        );
    }

    /**
     * Send reset game command to the server
     */
    sendResetGame() {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
            this.addLogEntry(
                "Cannot reset game - not connected to server",
                "system",
            );
            return;
        }

        this.socket.send(
            JSON.stringify({
                type: "server_command",
                command: "reset",
                game_id: this.gameId,
            }),
        );

        this.addLogEntry("Game reset requested", "system");
    }

    /**
     * Send force start command to the server
     */
    sendForceStart() {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
            this.addLogEntry(
                "Cannot force start - not connected to server",
                "system",
            );
            return;
        }

        this.socket.send(
            JSON.stringify({
                type: "server_command",
                command: "start",
                game_id: this.gameId,
            }),
        );

        this.addLogEntry("Force start requested", "system");
    }

    getGameIdFromURL() {
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get("game") || "default";
    }

    updateUIWithGameId() {
        // Update page title and header to show game ID (hide "default" for cleaner UI)
        const isDefaultGame = this.gameId === "default";
        const titleText = isDefaultGame
            ? "Chromatic Warfare"
            : `Chromatic Warfare - ${this.gameId}`;

        document.title = titleText;
        const header = document.querySelector("h1");
        if (header) {
            header.textContent = titleText;
        }

        // Update game ID display (show "default" in the info area but not in title)
        const gameIdElement = document.getElementById("gameId");
        if (gameIdElement) {
            gameIdElement.textContent = this.gameId;
        }
    }

    startRenderLoop() {
        const render = (timestamp) => {
            this.lastRenderTime = timestamp;
            this.updateAnimations(timestamp);
            this.renderFrame();
            requestAnimationFrame(render);
        };
        requestAnimationFrame(render);
    }

    updateAnimations(timestamp) {
        if (!this.isAnimating || !this.animationData) {
            return;
        }

        const elapsed = timestamp - this.animationStartTime;
        const progress = elapsed / this.serverTurnDuration;

        // Phase timings (as percentages of turn duration)
        const movementEnd = 0.3;
        const stateChangeEnd = 0.35; // Very brief instant snap
        const generationEnd = 0.55;

        if (progress <= movementEnd) {
            this.currentPhase = "movement";
            this.updateMovementAnimation(progress / movementEnd);
        } else if (progress <= stateChangeEnd) {
            if (this.currentPhase === "movement") {
                this.finishMovementPhase();
                this.setupStateChangePhase();
                this.currentPhase = "state_change";
            }
            // State change is instant - no animation needed
        } else if (progress <= generationEnd) {
            if (this.currentPhase === "state_change") {
                this.finishStateChangePhase();
                this.currentPhase = "generation";
                this.startGenerationAnimations();
            }
            this.updateGenerationAnimation(
                (progress - stateChangeEnd) / (generationEnd - stateChangeEnd),
            );
        } else {
            // Animation complete
            this.finishGenerationPhase();
            this.finishAnimation();
        }
    }

    setupStateChangePhase() {
        // Calculate what the vertex states should be after movement but before generation
        if (!this.animationData || !this.targetVertices) return;

        this.targetVertices.forEach((targetVertex, vertexId) => {
            const displayVertex = this.displayedVertices.get(vertexId);
            if (!displayVertex) return;

            // Calculate the pre-generation state (final state minus weight increment)
            let preGenerationUnits = targetVertex.units;

            // If this vertex generates units, subtract the weight from the target
            if (this.animationData.unit_generation) {
                const generatesUnits = this.animationData.unit_generation.find(
                    (gen) => gen.vertex_id === vertexId,
                );
                if (generatesUnits && targetVertex.controller) {
                    preGenerationUnits -= targetVertex.weight || 1;
                }
            }

            // Store the pre-generation state
            displayVertex.preGenerationUnits = preGenerationUnits;
            displayVertex.preGenerationController = targetVertex.controller;
        });
    }

    updateMovementAnimation(progress) {
        // Update positions of moving units with combat mechanics
        this.movingUnits.forEach((unit) => {
            const easedProgress = this.easeInOutCubic(progress);

            if (unit.combatResult === "survivor" && unit.finalX !== undefined) {
                // Bidirectional combat unit
                if (progress <= 0.5) {
                    // First half: move to combat point
                    const combatProgress = progress * 2;
                    unit.currentX =
                        unit.startX +
                        (unit.endX - unit.startX) * combatProgress;
                    unit.currentY =
                        unit.startY +
                        (unit.endY - unit.startY) * combatProgress;
                    unit.alpha = 1.0;
                    unit.displayedUnits = unit.originalUnits;
                } else {
                    // Second half: survivors continue to destination
                    if (unit.survivorCount > 0) {
                        const survivalProgress = (progress - 0.5) * 2;
                        unit.currentX =
                            unit.endX +
                            (unit.finalX - unit.endX) * survivalProgress;
                        unit.currentY =
                            unit.endY +
                            (unit.finalY - unit.endY) * survivalProgress;
                        unit.alpha = 1.0;
                        unit.displayedUnits = unit.survivorCount;
                    } else {
                        // No survivors - fade out at combat point
                        unit.alpha = Math.max(0, 1.0 - (progress - 0.5) * 4); // Fade out quickly
                        unit.displayedUnits = 0;
                    }
                }
            } else if (unit.combatResult === "eliminated") {
                // Units that get eliminated in combat
                if (progress <= 0.5) {
                    // Move to combat point
                    const combatProgress = progress * 2;
                    unit.currentX =
                        unit.startX +
                        (unit.endX - unit.startX) * combatProgress;
                    unit.currentY =
                        unit.startY +
                        (unit.endY - unit.startY) * combatProgress;
                    unit.alpha = 1.0;
                    unit.displayedUnits = unit.originalUnits;
                } else {
                    // Fade out at combat point
                    unit.alpha = Math.max(0, 1.0 - (progress - 0.5) * 4);
                    unit.displayedUnits = 0;
                }
            } else {
                // Single direction move - no combat
                unit.currentX =
                    unit.startX + (unit.endX - unit.startX) * easedProgress;
                unit.currentY =
                    unit.startY + (unit.endY - unit.startY) * easedProgress;
                unit.alpha = 1.0 - progress * 0.3; // Slightly fade during movement
                unit.displayedUnits = unit.units;
            }
        });
    }

    startGenerationAnimations() {
        // Initialize +Weight text animations
        this.unitGenerationAnimations = [];

        if (this.animationData && this.animationData.unit_generation) {
            this.animationData.unit_generation.forEach((genData) => {
                const displayVertex = this.displayedVertices.get(
                    genData.vertex_id,
                );
                if (displayVertex) {
                    // Get vertex position for the animation
                    const vertexPos = this.getVertexScreenPosition(
                        genData.vertex_id,
                    );
                    if (vertexPos) {
                        this.unitGenerationAnimations.push({
                            vertexId: genData.vertex_id,
                            weight: displayVertex.weight || 1,
                            x: vertexPos.x,
                            y: vertexPos.y,
                            startTime: 0, // Will be set relative to generation phase
                            alpha: 1.0,
                            offsetY: 0,
                        });
                    }
                }
            });
        }
    }

    updateGenerationAnimation(progress) {
        // Animate unit count increases from pre-generation to final state
        const easedProgress = this.easeInOutCubic(progress);

        // Update vertex size animations for unit generation
        this.unitGenerationAnimations.forEach((anim) => {
            const displayVertex = this.displayedVertices.get(anim.vertexId);
            if (!displayVertex) return;

            // There-and-back size animation (0 -> 1 -> 0 over the generation phase)
            let sizeMultiplier = 1.0;
            if (progress <= 0.5) {
                // First half: grow to 1.4x size
                sizeMultiplier = 1.0 + progress * 2 * 0.1; // 1.0 to 1.4
            } else {
                // Second half: shrink back to normal, then update units
                const shrinkProgress = (progress - 0.5) * 2; // 0 to 1 for second half
                sizeMultiplier = 1.1 - shrinkProgress * 0.1; // 1.4 back to 1.0

                // At the end of the animation (progress > 0.9), update the unit count
                if (progress > 0.9 && !displayVertex.unitsUpdated) {
                    const targetVertex = this.targetVertices.get(anim.vertexId);
                    if (targetVertex) {
                        displayVertex.units = targetVertex.units;
                        displayVertex.unitsUpdated = true;
                    }
                }
            }

            // Store the size multiplier for rendering
            displayVertex.sizeMultiplier = sizeMultiplier;
        });
    }

    getVertexScreenPosition(vertexId) {
        if (!this.gameState || !this.gameState.graph) return null;

        const canvas = this.canvas;
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        const vertices = this.gameState.graph.vertices;

        if (vertices.length === 0) return null;

        const vertex = vertices.find((v) => v.id === vertexId);
        if (!vertex) return null;

        const minX = Math.min(...vertices.map((v) => v.position[0]));
        const maxX = Math.max(...vertices.map((v) => v.position[0]));
        const minY = Math.min(...vertices.map((v) => v.position[1]));
        const maxY = Math.max(...vertices.map((v) => v.position[1]));

        const graphWidth = maxX - minX;
        const graphHeight = maxY - minY;

        const padding = 60;
        const scale = Math.min(
            (width - 2 * padding) / (graphWidth || 1),
            (height - 2 * padding) / (graphHeight || 1),
        );

        const offsetX = (width - graphWidth * scale) / 2 - minX * scale;
        const offsetY = (height - graphHeight * scale) / 2 - minY * scale;

        return {
            x: vertex.position[0] * scale + offsetX,
            y: vertex.position[1] * scale + offsetY,
        };
    }

    finishMovementPhase() {
        // Update source vertices, accounting for combat results
        this.movingUnits.forEach((unit) => {
            const sourceVertex = this.displayedVertices.get(unit.fromVertex);
            if (sourceVertex) {
                // Deduct the original units sent (regardless of combat outcome)
                sourceVertex.units = Math.max(
                    0,
                    sourceVertex.units - unit.units - unit.cost,
                );
            }
        });
        this.movingUnits = [];
    }

    finishStateChangePhase() {
        // Instantly snap to post-movement, pre-generation state
        this.displayedVertices.forEach((displayVertex, vertexId) => {
            if (displayVertex.preGenerationUnits !== undefined) {
                displayVertex.units = displayVertex.preGenerationUnits;
                displayVertex.controller =
                    displayVertex.preGenerationController;

                // Clear transition states
                displayVertex.controllerTransition = undefined;
                displayVertex.newController = undefined;
            }
        });
    }

    finishGenerationPhase() {
        // Ensure all units are at final values and clear temporary states
        this.displayedVertices.forEach((displayVertex, vertexId) => {
            const targetVertex = this.targetVertices.get(vertexId);
            if (targetVertex) {
                displayVertex.units = targetVertex.units;
                displayVertex.controller = targetVertex.controller;
                // Clear all temporary animation states
                displayVertex.baseUnits = undefined;
                displayVertex.preGenerationUnits = undefined;
                displayVertex.preGenerationController = undefined;
            }
        });

        // Clear generation animations
        this.unitGenerationAnimations = [];
    }

    finishAnimation() {
        this.isAnimating = false;
        this.animationData = null;
        this.currentPhase = "idle";
        this.movingUnits = [];
        this.unitGenerationAnimations = [];

        // Update the game state to match final values
        if (this.gameState && this.gameState.graph) {
            this.gameState.graph.vertices.forEach((vertex) => {
                const displayVertex = this.displayedVertices.get(vertex.id);
                if (displayVertex) {
                    vertex.units = displayVertex.units;
                    vertex.controller = displayVertex.controller;
                }
            });
        }
    }

    easeInOutCubic(t) {
        return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    }

    getVertexPosition(vertexId, scale, offsetX, offsetY) {
        if (!this.gameState || !this.gameState.graph) return null;

        const vertex = this.gameState.graph.vertices.find(
            (v) => v.id === vertexId,
        );
        if (!vertex) return null;

        return {
            x: vertex.position[0] * scale + offsetX,
            y: vertex.position[1] * scale + offsetY,
        };
    }

    setupCanvas() {
        const resizeCanvas = () => {
            const container = this.canvas.parentElement;
            const rect = container.getBoundingClientRect();

            // Set canvas size to match container
            this.canvas.width = rect.width * window.devicePixelRatio;
            this.canvas.height = rect.height * window.devicePixelRatio;
            this.canvas.style.width = rect.width + "px";
            this.canvas.style.height = rect.height + "px";

            // Scale context for high DPI displays
            this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

            // Re-render if we have game state
            if (this.gameState) {
                this.renderGraph(this.gameState.graph);
            }
        };

        resizeCanvas();
        window.addEventListener("resize", resizeCanvas);
    }

    setupResizeHandler() {
        let resizeTimeout;
        window.addEventListener("resize", () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.setupCanvas();
            }, 250);
        });
    }

    connectToServer(url) {
        // Determine URL based on environment
        if (!url) {
            if (import.meta.env.DEV) {
                url = "ws://localhost:8765";
            } else {
                const protocol =
                    window.location.protocol === "https:" ? "wss:" : "ws:";
                url = `${protocol}//${window.location.host}/api`;
            }
        }

        this.addLogEntry(
            `Attempting to connect to game server for game ${this.gameId}...`,
            "system",
        );

        try {
            this.socket = new WebSocket(url);

            this.socket.onopen = () => {
                this.updateConnectionStatus("connected", "Connected");
                this.addLogEntry(
                    `Successfully connected as viewer for game ${this.gameId}`,
                    "game-event",
                );

                // Join as viewer with game ID
                this.socket.send(
                    JSON.stringify({
                        type: "join_as_viewer",
                        game_id: this.gameId,
                    }),
                );
            };

            this.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleServerMessage(data);
                } catch (error) {
                    console.error("Error parsing server message:", error);
                }
            };

            this.socket.onclose = () => {
                this.updateConnectionStatus("disconnected", "Disconnected");
                this.addLogEntry(
                    `Connection lost for game ${this.gameId}. Reconnecting in 3 seconds...`,
                    "system",
                );

                // Attempt reconnection after 3 seconds
                setTimeout(() => {
                    this.connectToServer(url);
                }, 3000);
            };

            this.socket.onerror = (error) => {
                console.error("WebSocket error:", error);
                this.updateConnectionStatus("error", "Connection error");
                this.addLogEntry("Connection error occurred", "error");
            };
        } catch (error) {
            console.error("Failed to create WebSocket connection:", error);
            this.updateConnectionStatus("error", "Failed to connect");
            this.addLogEntry("Failed to connect to server", "error");
        }
    }

    handleServerMessage(data) {
        console.log("Received message:", data.type, data);

        switch (data.type) {
            case "game_state":
                this.updateGameState(data);
                break;
            case "turn_processed":
                this.handleTurnProcessed(data);
                break;
            case "game_over":
                this.showGameResult(data.final_rankings);
                this.addLogEntry(
                    "Game ended - This game will be removed in 60 seconds",
                    "system",
                );
                break;
            case "game_reset":
                this.handleGameReset();
                break;
            case "game_created":
                this.addLogEntry(
                    `Game ${this.gameId} created successfully`,
                    "game-event",
                );
                break;
            case "command_success":
                this.addLogEntry(`✓ ${data.message}`, "system");
                break;
            case "error":
                this.addLogEntry(`Server error: ${data.message}`, "error");
                break;
            default:
                console.log("Unknown message type:", data.type);
        }
    }

    getLowestAvailableColorIndex() {
        const usedIndices = new Set();

        // Collect all currently used color indices
        Object.values(this.playerColors).forEach((color) => {
            const index = this.colorPalette.indexOf(color);
            if (index !== -1) {
                usedIndices.add(index);
            }
        });

        // Find the lowest unused index
        for (let i = 0; i < this.colorPalette.length; i++) {
            if (!usedIndices.has(i)) {
                return i;
            }
        }

        // If all colors are used, cycle back (shouldn't happen with 8 colors typically)
        return usedIndices.size % this.colorPalette.length;
    }

    cleanupDisconnectedPlayerColors(currentPlayers) {
        const currentPlayerIds = new Set(currentPlayers.map((p) => p.id));

        // Remove colors for players no longer in the game
        Object.keys(this.playerColors).forEach((playerId) => {
            if (!currentPlayerIds.has(playerId)) {
                delete this.playerColors[playerId];
            }
        });
    }

    handleGameReset() {
        console.log("Game reset received");

        // Stop any ongoing animations
        this.isAnimating = false;
        this.animationData = null;
        this.currentPhase = "idle";
        this.movingUnits = [];
        this.unitGenerationAnimations = [];

        // Clear game state
        this.gameState = null;
        this.displayedVertices.clear();
        this.targetVertices.clear();

        // Reset player colors (will be reassigned when new players join)
        this.playerColors = {};
        this.previousPlayers.clear();

        // Reset UI elements
        document.getElementById("currentTurn").textContent = "-";
        document.getElementById("gameStatus").textContent = "Waiting";
        document.getElementById("turnProgress").textContent = " / -";
        document.getElementById("turnDuration").textContent = "-";

        // Clear player list
        const playerList = document.getElementById("playerList");
        playerList.innerHTML = "";

        // Hide game over overlay if it was showing
        const overlay = document.getElementById("gameOverOverlay");
        overlay.style.display = "none";

        // Clear canvas
        const ctx = this.ctx;
        const rect = this.canvas.getBoundingClientRect();
        ctx.clearRect(0, 0, rect.width, rect.height);
    }

    handleTurnProcessed(turnData) {
        console.log("Turn processed received:", turnData);

        // Update timing information if provided
        if (turnData.turn_duration_seconds !== undefined) {
            this.serverTurnDuration = turnData.turn_duration_seconds * 1000;
            document.getElementById("turnDuration").textContent =
                turnData.turn_duration_seconds.toFixed(1);
        }

        if (turnData.max_turns !== undefined) {
            this.maxTurns = turnData.max_turns;
            document.getElementById("turnProgress").textContent =
                ` / ${this.maxTurns}`;
        }

        // Log player actions for this turn
        this.logPlayerActions(turnData);

        // Don't start new animation if one is already running
        if (this.isAnimating) {
            console.log("Skipping animation - previous turn still animating");
            return;
        }

        // Ensure we have current game state
        if (!this.gameState || !this.gameState.graph) {
            console.log("No game state available for animation");
            return;
        }

        // Store the animation data
        this.animationData = turnData;

        // Check if we have any animations to perform
        const moveCount =
            turnData.move_animations && turnData.move_animations.length
                ? turnData.move_animations.length
                : 0;
        const genCount =
            turnData.unit_generation && turnData.unit_generation.length
                ? turnData.unit_generation.length
                : 0;

        console.log(
            `Animation data: ${moveCount} moves, ${genCount} generations`,
        );

        if (moveCount === 0 && genCount === 0) {
            console.log("No animations to perform");
            return;
        }

        // Initialize displayed vertices from current game state
        this.displayedVertices.clear();
        this.gameState.graph.vertices.forEach((vertex) => {
            this.displayedVertices.set(vertex.id, {
                id: vertex.id,
                controller: vertex.controller,
                units: vertex.units,
                position: vertex.position,
                weight: vertex.weight,
            });
        });

        console.log(
            "Displayed vertices initialized:",
            this.displayedVertices.size,
        );

        // Set up moving units for animation
        this.setupMovingUnits(
            turnData.move_animations && turnData.move_animations.length
                ? turnData.move_animations
                : [],
        );

        console.log("Moving units set up:", this.movingUnits.length);

        // Start the animation
        this.isAnimating = true;
        this.animationStartTime = performance.now();
        this.currentPhase = "movement";

        console.log("Animation started at:", this.animationStartTime);
    }

    logPlayerActions(turnData) {
        if (!turnData || !this.gameState) return;

        // Log turn header
        this.addLogEntry(`--- Turn ${turnData.turn} ---`, "turn-data");

        // Log player moves
        if (turnData.move_animations && turnData.move_animations.length > 0) {
            turnData.move_animations.forEach((move) => {
                const fromVertex = this.gameState.graph.vertices.find(
                    (v) => v.id === move.from_vertex,
                );
                const toVertex = this.gameState.graph.vertices.find(
                    (v) => v.id === move.to_vertex,
                );
                const fromPos = fromVertex
                    ? `(${fromVertex.position[0]}, ${fromVertex.position[1]})`
                    : move.from_vertex;
                const toPos = toVertex
                    ? `(${toVertex.position[0]}, ${toVertex.position[1]})`
                    : move.to_vertex;

                this.addLogEntry(
                    `${move.player_id}: moved ${move.units} units from ${fromPos} to ${toPos} (cost: ${move.cost})`,
                    "game-event",
                );
            });
        }

        // Log unit generation
        if (turnData.unit_generation && turnData.unit_generation.length > 0) {
            const generationByPlayer = {};
            turnData.unit_generation.forEach((gen) => {
                const vertex = this.gameState.graph.vertices.find(
                    (v) => v.id === gen.vertex_id,
                );
                if (vertex && vertex.controller) {
                    if (!generationByPlayer[vertex.controller]) {
                        generationByPlayer[vertex.controller] = 0;
                    }
                    generationByPlayer[vertex.controller] += vertex.weight || 1;
                }
            });

            Object.entries(generationByPlayer).forEach(
                ([playerId, totalGenerated]) => {
                    this.addLogEntry(
                        `${playerId}: generated +${totalGenerated} units`,
                        "game-event",
                    );
                },
            );
        }

        // Log active players and their status
        if (this.gameState.players && this.gameState.players.length > 0) {
            const activePlayers = this.gameState.players.filter(
                (p) => p.status === "active",
            );
            if (activePlayers.length > 0) {
                const playerStatus = activePlayers
                    .map((p) => `${p.id}(${p.total_units})`)
                    .join(", ");
                this.addLogEntry(
                    `Active players: ${playerStatus}`,
                    "game-event",
                );
            }

            // Note: Individual player eliminations are handled in logPlayerStatusChanges
        }
    }

    setupMovingUnits(moveAnimations) {
        this.movingUnits = [];

        if (!this.gameState || !this.gameState.graph) return;

        // Calculate current render parameters
        const canvas = this.canvas;
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;
        const vertices = this.gameState.graph.vertices;

        if (vertices.length === 0) return;

        const minX = Math.min(...vertices.map((v) => v.position[0]));
        const maxX = Math.max(...vertices.map((v) => v.position[0]));
        const minY = Math.min(...vertices.map((v) => v.position[1]));
        const maxY = Math.max(...vertices.map((v) => v.position[1]));

        const graphWidth = maxX - minX;
        const graphHeight = maxY - minY;

        const padding = 60;
        const scale = Math.min(
            (width - 2 * padding) / (graphWidth || 1),
            (height - 2 * padding) / (graphHeight || 1),
        );

        const offsetX = (width - graphWidth * scale) / 2 - minX * scale;
        const offsetY = (height - graphHeight * scale) / 2 - minY * scale;

        // Group moves by edge (bidirectional)
        const edgeMoves = new Map();

        moveAnimations.forEach((move) => {
            // Create a consistent edge key regardless of direction
            const edgeKey = [move.from_vertex, move.to_vertex].sort().join("-");

            if (!edgeMoves.has(edgeKey)) {
                edgeMoves.set(edgeKey, []);
            }
            edgeMoves.get(edgeKey).push(move);
        });

        // Process each edge's moves
        edgeMoves.forEach((moves, edgeKey) => {
            if (moves.length === 1) {
                // Single direction move - no combat
                const move = moves[0];
                const fromPos = this.getVertexPosition(
                    move.from_vertex,
                    scale,
                    offsetX,
                    offsetY,
                );
                const toPos = this.getVertexPosition(
                    move.to_vertex,
                    scale,
                    offsetX,
                    offsetY,
                );

                if (fromPos && toPos) {
                    this.movingUnits.push({
                        playerId: move.player_id,
                        fromVertex: move.from_vertex,
                        toVertex: move.to_vertex,
                        units: move.units,
                        cost: move.cost,
                        startX: fromPos.x,
                        startY: fromPos.y,
                        endX: toPos.x,
                        endY: toPos.y,
                        currentX: fromPos.x,
                        currentY: fromPos.y,
                        alpha: 1.0,
                        combatResult: "survivor", // This unit will reach its destination
                    });
                }
            } else if (moves.length === 2) {
                // Bidirectional moves - combat in the middle
                const [move1, move2] = moves;
                const pos1 = this.getVertexPosition(
                    move1.from_vertex,
                    scale,
                    offsetX,
                    offsetY,
                );
                const pos2 = this.getVertexPosition(
                    move1.to_vertex,
                    scale,
                    offsetX,
                    offsetY,
                );

                if (pos1 && pos2) {
                    // Calculate midpoint for combat
                    const midX = (pos1.x + pos2.x) / 2;
                    const midY = (pos1.y + pos2.y) / 2;

                    // Calculate combat result
                    const units1 = move1.units;
                    const units2 = move2.units;
                    const survivors1 = Math.max(0, units1 - units2);
                    const survivors2 = Math.max(0, units2 - units1);

                    // Create units for first move
                    this.movingUnits.push({
                        playerId: move1.player_id,
                        fromVertex: move1.from_vertex,
                        toVertex: move1.to_vertex,
                        units: units1,
                        cost: move1.cost,
                        startX: pos1.x,
                        startY: pos1.y,
                        endX: midX,
                        endY: midY,
                        finalX: pos2.x, // Where survivors go
                        finalY: pos2.y,
                        currentX: pos1.x,
                        currentY: pos1.y,
                        alpha: 1.0,
                        combatResult:
                            survivors1 > 0 ? "survivor" : "eliminated",
                        survivorCount: survivors1,
                        originalUnits: units1,
                    });

                    // Create units for second move
                    this.movingUnits.push({
                        playerId: move2.player_id,
                        fromVertex: move2.from_vertex,
                        toVertex: move2.to_vertex,
                        units: units2,
                        cost: move2.cost,
                        startX: pos2.x,
                        startY: pos2.y,
                        endX: midX,
                        endY: midY,
                        finalX: pos1.x, // Where survivors go
                        finalY: pos1.y,
                        currentX: pos2.x,
                        currentY: pos2.y,
                        alpha: 1.0,
                        combatResult:
                            survivors2 > 0 ? "survivor" : "eliminated",
                        survivorCount: survivors2,
                        originalUnits: units2,
                    });
                }
            }
        });
    }

    updateConnectionStatus(status, message) {
        const indicator = document.getElementById("connectionStatus");
        const statusText = document.getElementById("connectionText");

        if (indicator) {
            indicator.className = `w-2 h-2 rounded-full animate-pulse-slow ${status === "connected" ? "bg-solarized-green" : status === "disconnected" ? "bg-solarized-red" : "bg-solarized-yellow"}`;
        }

        if (statusText) {
            statusText.textContent = message;
        }
    }

    updateGameState(gameState) {
        console.log(
            "Game state update received:",
            gameState.turn,
            gameState.game_status,
        );

        const wasFirstState = !this.gameState;
        this.gameState = gameState;

        // Update timing information from server
        if (gameState.turn_duration_seconds !== undefined) {
            this.serverTurnDuration = gameState.turn_duration_seconds * 1000; // Convert to milliseconds
            document.getElementById("turnDuration").textContent =
                gameState.turn_duration_seconds.toFixed(1);
        }

        if (gameState.max_turns !== undefined) {
            this.maxTurns = gameState.max_turns;
        }

        // Update UI elements
        document.getElementById("currentTurn").textContent = gameState.turn;
        document.getElementById("turnProgress").textContent =
            ` / ${this.maxTurns}`;
        document.getElementById("gameStatus").textContent =
            gameState.game_status.charAt(0).toUpperCase() +
            gameState.game_status.slice(1);

        // Update players
        this.updatePlayers(gameState.players);

        // If this is the first state or no animation is running, render immediately
        if (wasFirstState || !this.isAnimating) {
            console.log(
                "Updating displayed vertices immediately (no animation running)",
            );

            // Log game start if this is the first state
            if (wasFirstState && gameState.game_status === "active") {
                this.addLogEntry(
                    "🎮 Game started! Players are making their moves.",
                    "game-event",
                );
            }
            // Initialize displayed vertices from game state
            if (gameState.graph) {
                this.displayedVertices.clear();
                gameState.graph.vertices.forEach((vertex) => {
                    this.displayedVertices.set(vertex.id, {
                        id: vertex.id,
                        controller: vertex.controller,
                        units: vertex.units,
                        position: vertex.position,
                        weight: vertex.weight,
                    });
                });
            }
        } else {
            console.log("Animation is running, storing target state");
            // Animation is running, store the target state
            if (gameState.graph && this.animationData) {
                this.targetVertices.clear();
                gameState.graph.vertices.forEach((vertex) => {
                    this.targetVertices.set(vertex.id, {
                        id: vertex.id,
                        controller: vertex.controller,
                        units: vertex.units,
                        position: vertex.position,
                        weight: vertex.weight,
                    });
                });

                // Set base units for generation animation
                if (this.animationData && this.animationData.unit_generation) {
                    this.animationData.unit_generation.forEach((genData) => {
                        const displayVertex = this.displayedVertices.get(
                            genData.vertex_id,
                        );
                        if (displayVertex && !displayVertex.baseUnits) {
                            displayVertex.baseUnits = displayVertex.units;
                        }
                    });
                }

                console.log(
                    "Target vertices stored for animation:",
                    this.targetVertices.size,
                );
            }
        }
    }

    renderFrame() {
        if (!this.gameState || !this.gameState.graph) return;

        const canvas = this.canvas;
        const ctx = this.ctx;
        const rect = canvas.getBoundingClientRect();
        const width = rect.width;
        const height = rect.height;

        // Clear canvas
        ctx.clearRect(0, 0, width, height);

        // Calculate scale and offset for graph positioning
        const vertices = this.gameState.graph.vertices;
        if (vertices.length === 0) return;

        const minX = Math.min(...vertices.map((v) => v.position[0]));
        const maxX = Math.max(...vertices.map((v) => v.position[0]));
        const minY = Math.min(...vertices.map((v) => v.position[1]));
        const maxY = Math.max(...vertices.map((v) => v.position[1]));

        const graphWidth = maxX - minX;
        const graphHeight = maxY - minY;

        const padding = 60;
        const scale = Math.min(
            (width - 2 * padding) / (graphWidth || 1),
            (height - 2 * padding) / (graphHeight || 1),
        );

        const offsetX = (width - graphWidth * scale) / 2 - minX * scale;
        const offsetY = (height - graphHeight * scale) / 2 - minY * scale;

        // Draw edges first
        ctx.strokeStyle =
            "rgba(147, 161, 161, 0.6)"; /* Same as neutral vertex color */
        ctx.lineWidth = 3;
        ctx.beginPath();

        this.gameState.graph.edges.forEach((edge) => {
            const fromVertex = vertices.find((v) => v.id === edge.from);
            const toVertex = vertices.find((v) => v.id === edge.to);

            if (fromVertex && toVertex) {
                const fromX = fromVertex.position[0] * scale + offsetX;
                const fromY = fromVertex.position[1] * scale + offsetY;
                const toX = toVertex.position[0] * scale + offsetX;
                const toY = toVertex.position[1] * scale + offsetY;

                ctx.moveTo(fromX, fromY);
                ctx.lineTo(toX, toY);
            }
        });

        ctx.stroke();

        // Draw vertices using displayed state
        vertices.forEach((vertex) => {
            const displayVertex = this.displayedVertices.get(vertex.id);
            if (displayVertex) {
                this.renderVertex(displayVertex, scale, offsetX, offsetY);
            } else {
                this.renderVertex(vertex, scale, offsetX, offsetY);
            }
        });

        // Draw moving units
        this.renderMovingUnits();
    }

    renderVertex(vertex, scale, offsetX, offsetY) {
        const ctx = this.ctx;
        const x = vertex.position[0] * scale + offsetX;
        const y = vertex.position[1] * scale + offsetY;

        // Calculate adaptive radius based on scale
        // Uses logarithmic scaling to make vertices smaller when zoomed in, but not linearly
        const baseRadius = 35;
        const minRadius = 18;
        const maxRadius = 100;

        // Normalize scale (assuming typical scale range of 0.1 to 10)
        const normalizedScale = Math.max(0.1, Math.min(10, scale));
        const scaleLog = Math.log(normalizedScale + 1);
        const maxScaleLog = Math.log(11); // log(10 + 1)

        // Inverse relationship: higher scale = smaller radius
        const scaleFactor = 1 - (scaleLog / maxScaleLog) * 0.6; // 0.6 controls how much scaling affects size
        let radius = Math.max(
            minRadius,
            Math.min(maxRadius, baseRadius * scaleFactor),
        );

        // Dynamic scaling based on vertex properties

        // Weight-based scaling: 1.0 to 1.1 based on weight (exponentially approaching 1.1)
        const weight = vertex.weight || 1;
        const weightScale = 1 + 0.15 * (1 - Math.exp(-(weight - 1) * 0.5));

        // Unit-based scaling: 1.0 to 1.25 based on units (exponentially approaching 1.25)
        let unitScale = 1.0;
        if (vertex.controller && vertex.units > 0) {
            unitScale = 1 + 0.15 * (1 - Math.exp(-(vertex.units - 1) * 0.1));
        }

        // Apply the dynamic scaling
        radius *= weightScale * unitScale;

        // Ensure we still respect the min/max bounds after scaling
        radius = Math.max(minRadius, Math.min(maxRadius, radius));

        if (vertex.sizeMultiplier) {
            radius = radius * vertex.sizeMultiplier;
        }

        // Determine vertex color
        let fillColor = "rgba(147, 161, 161, 1.0)"; // Fully opaque neutral vertex color

        if (vertex.controllerTransition !== undefined) {
            // Animate color transition
            const oldColor = vertex.controller
                ? this.playerColors[vertex.controller]
                : "rgba(147, 161, 161, 1.0)";
            const newColor = vertex.newController
                ? this.playerColors[vertex.newController]
                : "rgba(147, 161, 161, 1.0)";
            fillColor = this.interpolateColor(
                oldColor,
                newColor,
                vertex.controllerTransition,
            );
        } else if (vertex.controller) {
            fillColor = this.playerColors[vertex.controller] || "#93a1a1";
        }

        // Draw vertex circle (no stroke)
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fillStyle = fillColor;
        ctx.fill();

        // Draw unit count with adaptive font size
        ctx.fillStyle = "#fdf6e3"; /* Solarized light base3 for high contrast */
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        // Scale font size with radius
        const baseFontSize = vertex.controller ? 14 : 10;
        const fontSize = Math.max(8, baseFontSize * (radius / baseRadius));

        if (vertex.controller) {
            ctx.font = `bold ${fontSize}px sans-serif`;
            ctx.fillText(vertex.units.toString(), x, y);
        } else {
            ctx.font = `bold ${fontSize}px sans-serif`;
            ctx.fillText(vertex.weight.toString(), x, y);
        }
    }

    renderMovingUnits() {
        const ctx = this.ctx;

        this.movingUnits.forEach((unit) => {
            if (
                unit.alpha <= 0 ||
                (unit.displayedUnits !== undefined && unit.displayedUnits <= 0)
            )
                return;

            const color = this.playerColors[unit.playerId] || "#ffffff";
            const displayUnits =
                unit.displayedUnits !== undefined
                    ? unit.displayedUnits
                    : unit.units;

            // Draw moving unit as a smaller circle
            ctx.save();
            ctx.globalAlpha = unit.alpha;

            // Add combat effect - red glow for units in combat
            if (
                unit.combatResult &&
                unit.combatResult !== "survivor" &&
                unit.alpha > 0.5
            ) {
                ctx.shadowColor = "#dc322f";
                ctx.shadowBlur = 10;
            }

            ctx.beginPath();
            ctx.arc(unit.currentX, unit.currentY, 8, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();

            // Draw unit count only if there are units to display
            if (displayUnits > 0) {
                ctx.fillStyle = "#fdf6e3"; /* Solarized light base3 */
                ctx.font = "bold 10px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(
                    displayUnits.toString(),
                    unit.currentX,
                    unit.currentY,
                );
            }

            ctx.restore();
        });
    }

    interpolateColor(color1, color2, t) {
        // Simple color interpolation for RGB hex colors
        if (
            typeof color1 === "string" &&
            color1.startsWith("#") &&
            typeof color2 === "string" &&
            color2.startsWith("#")
        ) {
            const r1 = parseInt(color1.substr(1, 2), 16);
            const g1 = parseInt(color1.substr(3, 2), 16);
            const b1 = parseInt(color1.substr(5, 2), 16);

            const r2 = parseInt(color2.substr(1, 2), 16);
            const g2 = parseInt(color2.substr(3, 2), 16);
            const b2 = parseInt(color2.substr(5, 2), 16);

            const r = Math.round(r1 + (r2 - r1) * t);
            const g = Math.round(g1 + (g2 - g1) * t);
            const b = Math.round(b1 + (b2 - b1) * t);

            return `rgb(${r}, ${g}, ${b})`;
        }

        // Fallback to direct transition
        return t < 0.5 ? color1 : color2;
    }

    logPlayerStatusChanges(players) {
        if (!players) return;

        players.forEach((player) => {
            const prevPlayer = this.previousPlayers.get(player.id);

            if (!prevPlayer) {
                // New player joined
                this.addLogEntry(
                    `${player.id} joined the game (${player.total_units} units)`,
                    "game-event",
                );
            } else {
                // Check for significant unit count changes (threshold of 5+ units)
                const unitDiff = player.total_units - prevPlayer.total_units;
                if (Math.abs(unitDiff) >= 5) {
                    if (unitDiff > 0) {
                        this.addLogEntry(
                            `${player.id} gained ${unitDiff} units (now ${player.total_units})`,
                            "game-event",
                        );
                    } else {
                        this.addLogEntry(
                            `${player.id} lost ${Math.abs(unitDiff)} units (now ${player.total_units})`,
                            "game-event",
                        );
                    }
                }
            }

            // Update stored player state
            this.previousPlayers.set(player.id, {
                id: player.id,
                status: player.status,
                total_units: player.total_units,
            });
        });
    }

    updatePlayers(players) {
        const playerList = document.getElementById("playerList");

        // Clean up colors for disconnected players first
        this.cleanupDisconnectedPlayerColors(players);

        // Check for player status changes before updating UI
        this.logPlayerStatusChanges(players);

        playerList.innerHTML = "";

        // Use final_rankings from server if available, otherwise fall back to unit count
        let sortedPlayers;
        if (
            this.gameState &&
            this.gameState.final_rankings &&
            this.gameState.final_rankings.length > 0
        ) {
            // Use server's final rankings order
            const rankingsMap = new Map();
            this.gameState.final_rankings.forEach((playerId, index) => {
                rankingsMap.set(playerId, index);
            });

            sortedPlayers = [...players].sort((a, b) => {
                const rankA = rankingsMap.get(a.id);
                const rankB = rankingsMap.get(b.id);

                // If both players are in rankings, sort by ranking
                if (rankA !== undefined && rankB !== undefined) {
                    return rankA - rankB;
                }
                // If only one is in rankings, that one comes first
                if (rankA !== undefined) return -1;
                if (rankB !== undefined) return 1;
                // If neither is in rankings, sort by units
                return b.total_units - a.total_units;
            });
        } else {
            // Fall back to sorting by total units (descending)
            sortedPlayers = [...players].sort(
                (a, b) => b.total_units - a.total_units,
            );
        }

        const gameEnded =
            this.gameState && this.gameState.game_status === "ended";

        sortedPlayers.forEach((player, index) => {
            // Assign color using lowest available index if not already assigned
            if (!this.playerColors[player.id]) {
                const colorIndex = this.getLowestAvailableColorIndex();
                this.playerColors[player.id] = this.colorPalette[colorIndex];
            }

            const playerCard = document.createElement("div");
            let cardClass = `player-card ${player.status}`;

            // Add winner styling for game end
            if (gameEnded) {
                if (index === 0) {
                    cardClass += " winner";
                } else if (index === 1) {
                    cardClass += " second";
                } else if (index === 2) {
                    cardClass += " third";
                }
            }

            playerCard.className = cardClass;
            playerCard.style.borderLeftColor = this.playerColors[player.id];

            // Determine rank text and emoji
            let rankText = "";
            let statusEmoji = "";

            if (gameEnded) {
                // Game ended - show final rankings with medals
                if (index === 0) {
                    rankText = "🏆 ";
                } else if (index === 1) {
                    rankText = "🥈 ";
                } else if (index === 2) {
                    rankText = "🥉 ";
                } else {
                    rankText = `#${index + 1} `;
                }
            } else {
                // Game in progress - show current ranking
                rankText = `#${index + 1} `;

                // Add skull emoji for players with 0 units
                if (player.total_units === 0) {
                    statusEmoji = " 💀";
                }
            }

            playerCard.innerHTML = `
            <div class="player-name" style="color: ${this.playerColors[player.id]}">
                ${rankText}${player.id}${statusEmoji}
            </div>
            <div class="player-stats">
                <span>Units: ${player.total_units}</span>
                <span class="player-status ${player.status}">${player.status}</span>
            </div>
        `;

            playerList.appendChild(playerCard);
        });
    }

    renderGraph(graph) {
        // This method is now handled by renderFrame()
        // Keep for compatibility but delegate to renderFrame
        this.renderFrame();
    }

    updateVertex(vertex, scale, offsetX, offsetY) {
        // This method is now handled by renderVertex()
        // Keep for compatibility
        this.renderVertex(vertex, scale, offsetX, offsetY);
    }

    showGameResult(rankings) {
        // Don't show the overlay - winners will be shown in the ordered players list
        this.addLogEntry(
            "🏆 Game completed! Final rankings displayed in players list.",
            "game-event",
        );

        // Update the players list to reflect final standings
        if (this.gameState) {
            this.updatePlayers(this.gameState.players);
        }
    }

    addLogEntry(message, type = "") {
        const gameLog = document.getElementById("gameLog");
        const logContainer = gameLog.querySelector("div") || gameLog;
        const entry = document.createElement("div");
        entry.className = `log-entry ${type}`;
        entry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;

        logContainer.appendChild(entry);
        gameLog.scrollTop = gameLog.scrollHeight;

        // Limit log entries to prevent memory issues
        while (logContainer.children.length > 100) {
            logContainer.removeChild(logContainer.firstChild);
        }
    }

    // Legacy animation methods (now integrated into the main system)

    animateUnitMovement(moveData) {
        // This functionality is now handled by the main animation system
        console.log("Legacy method called - use main animation system instead");
    }

    animateUnitGeneration(generationData) {
        // This functionality is now handled by the main animation system
        console.log("Legacy method called - use main animation system instead");
    }

    animateCombatResolution(combatData) {
        // This functionality is now handled by the main animation system
        console.log("Legacy method called - use main animation system instead");
    }
}
