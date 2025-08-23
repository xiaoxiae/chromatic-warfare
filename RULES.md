# Game Rules

## Overview

The game is a turn-based strategy played on a directed graph.
Players compete to control vertices and eliminate opponents through movement and combat.

## Game Setup

### Graph Structure

* The game takes place on a directed graph of vertices connected by edges.
* Each vertex has a **weight** (unit generation rate), a **position** (for visualization only), a **controller** (the owning player or neutral), and a **unit count** (units stationed there).
* Edges are one-way; units may only move in the specified direction.

### Starting Conditions

* A game requires at least two players.
* Each player begins on a random vertex with one unit.
* All other vertices start neutral and uncontrolled.

## Turn Structure

Each turn proceeds in three phases:

1. **Movement & Combat**

   * Players submit movement orders simultaneously.
   * All valid movements are executed.
   * Combat occurs where multiple forces contest a vertex.

2. **Unit Generation**

   * Each controlled vertex produces new units equal to its weight.
   * Units are placed directly on the generating vertex.

3. **Elimination Check**

   * A player with no controlled vertices or no units is eliminated.

Turns repeat until one player remains or the round limit is reached.

## Movement Rules

* Units may only move along directed edges to adjacent vertices.
* Players may only move units from vertices they control.
* A player may not move more units than are present at a vertex.
* Any number of moves may be issued in the same turn.

## Combat Rules

Combat occurs whenever units arrive at a neutral or enemy-controlled vertex.

### Attacking Neutral Vertices

* The attacking force must first pay a cost equal to the vertex weight.
* If the attack force is smaller than the weight, the attack fails.
* If equal or greater, the surplus becomes the effective attacking units.
* A single attacker claims the vertex with the effective units.
* If multiple attackers are present, a winner is chosen randomly with probability weighted by their share of the attacking force.
  The winner controls the vertex with all effective units.

### Attacking Controlled Vertices

If forces move in opposite directions along the same edge, they destroy each other unit-for-unit.
Only the net remainder continues to its destination.

* Combat is resolved by comparing attacking and defending totals (this includes defending units sent from allied vertices!):

  * If attackers have fewer units, the defender wins with the surplus.
  * If both sides are equal, the vertex becomes neutral with no units.
  * If attackers have more units, they win with the surplus.
  * With multiple attackers, a winner is chosen randomly, weighted by contribution.

### Probabilistic Resolution

When multiple attackers contest the same vertex (either neutral or controlled), each player’s chance of victory is proportional to their contribution.
The chosen winner takes control with all surviving units.

## Victory Conditions

The game ends under the following conditions:

* **Elimination Victory**: Only one player remains in control; that player wins.
* **Turn Limit**: The game reaches its maximum number of turns (default 100). Players are then ranked by survival status, remaining unit count, and elimination order.

## Ranking

* Active players at the end of a timed-out game are ranked by unit count. Equal totals result in ties.
* Eliminated players are ranked in reverse order of elimination. Players eliminated in the same round tie.
