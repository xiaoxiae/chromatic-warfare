# ===== SERVER CONFIGURATION =====
class ServerConfig:
    """WebSocket server configuration."""
    HOST = "localhost"
    PORT = 8765

    GAME_LOG_PATH = "game_results.log"


# ===== GAME DEFAULTS =====
class GameDefaults:
    """Default settings for new games."""
    
    # Grid dimensions
    GRID_WIDTH = 7
    GRID_HEIGHT = 5
    
    # Game timing
    MAX_TURNS = 24
    TURN_DURATION = 1.0  # seconds
    
    # Player settings
    STARTING_UNITS = 1
    MINIMUM_PLAYERS = 2
    
    # Grace period before game starts
    GRACE_PERIOD = 15.0  # seconds
    
    # Map generation
    MAP_TYPE = "grid"  # "grid" or "hex"
    VERTEX_WEIGHT_RANGE = None  # (min, max) or None for weight=1
    VERTEX_REMOVE_PROBABILITY = None  # 0.0-1.0 or None for no removal


# ===== BOT MANAGEMENT =====
class BotConfig:
    """AI bot configuration."""
    
    # Available difficulty levels
    DIFFICULTIES = ["easy", "medium", "hard"]
    
    # Bot naming
    BOT_NAME_PREFIX = {
        "easy": "EasyBot",
        "medium": "MediumBot", 
        "hard": "HardBot"
    }
    
    # Auto-start behavior
    AUTO_START_WITH_BOTS = True


# ===== MAP GENERATION =====
class MapConfig:
    """Map generation settings and limits."""
    
    # Size limits
    MIN_WIDTH = 2
    MAX_WIDTH = 50
    MIN_HEIGHT = 2
    MAX_HEIGHT = 50
    
    # Weight settings
    MIN_VERTEX_WEIGHT = 1
    MAX_VERTEX_WEIGHT = 100
    
    # Removal probability limits
    MIN_REMOVE_PROBABILITY = 0.0
    MAX_REMOVE_PROBABILITY = 0.8
