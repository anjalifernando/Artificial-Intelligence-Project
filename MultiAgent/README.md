# MultiAgent Project Task Split (3 People)

This file splits implementation and testing work into Person 1, Person 2, and Person 3.

## Aseel: Reflex Agent (Q1-style)

### Own these sections
- Improve ReflexAgent.evaluationFunction in multiagent/multiAgents.py
- Tune feature scoring (food distance, ghost danger, scared ghosts, stop penalty)
- Run local play tests on small layouts

### Deliverables
- Reflex behavior is clearly better than score-only baseline
- Code is stable and does not crash in normal layouts

---

## Anjali: Minimax Agent (Q2)

### Own these sections
- Implement MinimaxAgent.getAction in multiagent/multiAgents.py
- Correct turn order handling for Pacman and all ghosts
- Correct depth handling per full ply
- Handle terminal states and no-legal-action states
- Validate with q2 tests/autograder

### Deliverables
- Q2 tests pass
- Action choices are consistent with minimax expectations

---

## Vicky: Expectimax Agent (Q4) + Final Validation

### Own these sections
- Implement ExpectimaxAgent.getAction in multiagent/multiAgents.py
- Use uniform random ghost policy in expectation node
- Validate with q4 tests/autograder
- Run final end-to-end project validation and collect results

### Deliverables
- Q4 tests pass
- Final combined code runs and grading output is clean

---
