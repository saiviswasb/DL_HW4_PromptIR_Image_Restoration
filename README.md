# DL HW4 - Single Model Image Restoration using Modified PromptIR

## Overview

This project implements a modified PromptIR framework for image restoration under adverse weather conditions. A single model is trained to restore both rain-degraded and snow-degraded images using the dataset provided in the course assignment.

The proposed approach extends PromptIR with:

* Deeper feature extraction stages
* Squeeze-and-Excitation attention
* Hybrid Charbonnier + FFT loss
* Exponential Moving Average (EMA)
* 8-fold Test-Time Augmentation (TTA)

The final submission achieved a public leaderboard score of **29.51 PSNR**.

---

## Dataset

The dataset contains paired degraded and clean images for:

* Rain Restoration
* Snow Restoration

A single model is trained on both degradation types.

---

## Training Configuration

| Parameter         | Value            |
| ----------------- | ---------------- |
| Model             | PromptIR         |
| Feature Dimension | 64               |
| Patch Size        | 128              |
| Batch Size        | 4                |
| Epochs            | 250              |
| Learning Rate     | 2e-4             |
| Optimizer         | AdamW            |
| Scheduler         | Cosine Annealing |
| EMA               | 0.999            |

---

## Additional Experiments

### Test-Time Augmentation

| Method      | PSNR   |
| ----------- | ------ |
| Without TTA | 27.589 |
| 8-fold TTA  | 27.884 |

TTA improved validation performance by approximately 0.30 dB.

---

## Final Result

Public Leaderboard Score:

**29.51 PSNR**

---

## Running

Training:

```bash
python main.py
```

Inference:

```bash
python main.py --infer
```

---

## Author

Basetti Sai Viswas
