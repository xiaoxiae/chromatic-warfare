#!/usr/bin/env python3
import sys
from typing import List

from bot.lib import GameBot, Game, Command


class MediumBot(GameBot):
    def play_turn(self, game: Game) -> List[Command]:
        """
        - Move units from interior vertices to frontline vertices
        - Capture neighbouring neutral vertices
        """
        return self._move_to_frontline(game) + self._expand_to_neutrals(game)

    def _move_to_frontline(self, game: Game) -> List[Command]:
        """Move units from interior vertices evenly to all closer neighbors."""
        commands = []

        for vertex in game.my_vertices:
            distance_to_frontline = game.get_distance_to_frontline(vertex.id)

            # Find all neighbors that are closer to frontline
            closer_neighbors = []
            for neighbor in game.get_my_neighbors(vertex.id):
                neighbor_distance = game.get_distance_to_frontline(neighbor.id)

                if neighbor_distance >= 0 and neighbor_distance < distance_to_frontline:
                    closer_neighbors.append(neighbor)

            if not closer_neighbors:
                continue

            weakest_neighbor = min(closer_neighbors, key=lambda v: v.units)

            # Send to the weakest one
            command = self.move_all(vertex, weakest_neighbor)
            commands.append(command)

        return commands

    def _expand_to_neutrals(self, game: Game) -> List[Command]:
        """Expand to neutral vertices from frontline positions using all available units."""
        commands = []

        for vertex in game.get_frontline_vertices():
            # Find weakest neutral neighbour
            neutral_neighbors = game.get_neutral_neighbors(vertex.id)

            if not neutral_neighbors:
                continue

            weakest_neutral = min(neutral_neighbors, key=lambda v: v.weight)

            if weakest_neutral is None:
                continue

            # Capture with all units if we can
            if game.can_capture(vertex.id, weakest_neutral.id, vertex.units):
                command = self.move_all(vertex, weakest_neutral)
                commands.append(command)

        return commands


if __name__ == "__main__":
    game_id = sys.argv[1] if len(sys.argv) > 1 else None
    player_id = sys.argv[2] if len(sys.argv) > 2 else None

    bot = MediumBot(game_id, player_id)
    bot.run()
