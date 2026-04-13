# MultiAgent Project Task Split (3 People)

This file splits the Minimax and Expectimax work across three people so each person owns real algorithm code.

## Aseel: Minimax Lead

### Own these sections
- Implement the Pacman max branch in `MinimaxAgent.getAction`
- Choose the best root action from the minimax tree
- Start the recursive minimax search from the current `GameState`

### Deliverables
- The root Pacman decision is produced by minimax
- The agent returns a legal action from the best minimax branch

---

## Anjali: Shared Depth and Leaf Logic

### Own these sections
- Handle full-ply depth counting in `MinimaxAgent`
- Stop the search at the correct depth
- Use `self.evaluationFunction` on leaf states
- Apply the same depth and leaf rules in `ExpectimaxAgent`

### Deliverables
- Depth is counted as one Pacman move plus all ghost responses
- Leaf evaluation is consistent in both search agents

---

## Vicky: Ghost Branches for Both Questions

### Own these sections
- Implement the ghost min layers in `MinimaxAgent`
- Cycle through multiple ghosts in order
- Implement the chance-node logic in `ExpectimaxAgent`
- Compute uniform random expected values for ghost actions

### Deliverables
- Minimax correctly handles every ghost layer
- Expectimax correctly averages over ghost actions
- Both agents work for any number of ghosts

---

## Team Integration Notes
- Use the provided autograder commands for q2 and q4
- Run local Pacman games to confirm the agents behave correctly
- Merge the three parts into one `multiAgents.py` before submission
