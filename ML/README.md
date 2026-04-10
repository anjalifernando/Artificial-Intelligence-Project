# ML Project Task Split (3 People)

This file splits the ML project work into Person 1, Person 2, and Person 3.

## Vicky : Data Setup + Model Architecture

### Own these sections
- Environment setup and data loading
- Transform pipeline (ToTensor + Normalize)
- DataLoader setup for train/test
- Print dataset sample counts
- Model layers (fc2, fc3)
- Forward pass TODOs in model
- Trainable parameter count

### Deliverables
- Data pipeline runs without errors
- Model prints correctly
- Total trainable parameters are printed

---

## Aseel: Training Loop

### Own these sections
- Optimizer setup (Adam, lr=0.001)
- Number of epochs
- Full training step logic:
  - zero_grad
  - forward pass
  - loss computation
  - backward pass
  - optimizer step
- Epoch logging format (loss + accuracy)

### Deliverables
- Training runs for all epochs
- Loss decreases reasonably
- Training accuracy prints every epoch

---

## Anjali: Evaluation + Save/Load + Reflection

### Own these sections
- Test evaluation loop
- Test accuracy calculation
- Prediction visualization batch fetch
- Save model to digit_classifier.pth
- Load saved model and verify accuracy
- Confusion matrix computation
- Team reflection markdown answers

### Deliverables
- Test accuracy printed
- Training curves and prediction grid shown
- Confusion matrix shown
- Save/load verification completed
- Reflection answers completed

---

## Team Integration Checklist
- Merge all TODO completions into one notebook
- Run notebook top-to-bottom once (no skipped cells)
- Confirm all outputs required in checklist are visible
- Confirm digit_classifier.pth is generated
