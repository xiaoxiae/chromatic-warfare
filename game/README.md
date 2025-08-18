# Bot Library for Turn-Based Strategy Game

This library provides a simple API for creating bots. Inherit from `GameBot`, implement `play_turn()`, and call `bot.run()`.

## Quick Start

```python
#!/usr/bin/env python3
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from botlib import GameBot, Game, Command


class EasyBot(GameBot):
    def play_turn(self, game: Game) -> List[Command]:
        """
        - Capture neighbouring neutral vertices
        """
        return self._expand_to_neutrals()

    def _expand_to_neutrals(self) -> List[Command]:
        """Expand to neutral vertices from frontline positions using all available units."""
        commands = []
        
        for vertex in self.game.get_frontline_vertices():
            # Find weakest neutral neighbour
            neutral_neighbors = self.game.get_neutral_neighbors(vertex.id)

            if not neutral_neighbors:
                continue

            weakest_neutral = min(neutral_neighbors, key=lambda v: v.weight)
            
            if weakest_neutral is None:
                continue
            
            # Capture with all units if we can
            if self.game.can_capture(vertex.id, weakest_neutral.id, vertex.units):
                command = self.move_all(vertex, weakest_neutral)
                commands.append(command)
        
        return commands


if __name__ == "__main__":
    game_id = sys.argv[1] if len(sys.argv) > 1 else None
    player_id = sys.argv[2] if len(sys.argv) > 2 else None

    bot = EasyBot(game_id, player_id)

    bot.run()

    # alternatively, you can also request a game with bots
    # bot.run(bots=[(1, "easy"), (1, "medium"), (1, "hard")])
```

## Movement Commands

```python
# Move specific number of units
self.move(from_vertex, to_vertex, units)

# Move all units from a vertex
self.move_all(from_vertex, to_vertex)
```

## Game Object

```python
game.my_vertices            # List of vertices you control
game.enemy_vertices         # List of enemy-controlled vertices  
game.neutral_vertices       # List of uncontrolled vertices
```

## Vertex Properties

```python
vertex.id                   # Unique identifier
vertex.units                # Number of units stationed
vertex.weight               # Capture cost for neutrals
vertex.is_mine              # True if you control it
vertex.is_neutral           # True if uncontrolled
vertex.is_enemy             # True if enemy controls it
```

## Navigation

```python
# Get all adjacent vertices
neighbors = game.get_neighbors(vertex_id)

# Get neighbors by type
enemy_neighbors = game.get_enemy_neighbors(vertex_id)
neutral_neighbors = game.get_neutral_neighbors(vertex_id)
my_neighbors = game.get_my_neighbors(vertex_id)
```

## Capture & Frontline

```python
# Check if you can capture a vertex
can_win = game.can_capture(from_id, to_id, units)

# Get frontline vertices (those with non-friendly neighbors)
frontline = game.get_frontline_vertices()

# Get distance to nearest frontline
distance = game.get_distance_to_frontline(vertex_id)
```
