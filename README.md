# DL HW4 – Single-Model Image Restoration Using Modified PromptIR

## Overview

This project presents a modified PromptIR framework for image restoration under adverse weather conditions. The objective is to train a single model capable of restoring both rain-degraded and snow-degraded images using the dataset provided in the Deep Learning HW4 assignment.

The proposed solution extends the original PromptIR architecture with several enhancements, including deeper feature extraction stages, Squeeze-and-Excitation (SE) attention modules, hybrid Charbonnier and FFT loss optimization, Exponential Moving Average (EMA) parameter updates, and 8-fold Test-Time Augmentation (TTA) during inference.

The final submission achieved a **Public Leaderboard Score of 29.51 PSNR**, demonstrating strong restoration performance on both rain and snow degradation scenarios.

---

## Final Public Leaderboard Result

**PSNR: 29.51**

<img width="988" height="43" alt="image" src="https://github.com/user-attachments/assets/9aeb298b-92cf-4549-9854-0c28c3ebf52a" />


---

## Sample Restoration Results

### Rain Restoration Example

The model successfully removes rain streaks while preserving scene structure, texture details, and color consistency.

<img width="951" height="272" alt="image" src="https://github.com/user-attachments/assets/d9f451e6-cb65-46ad-af9e-5597f0e1bf92" />


---

### Snow Restoration Example

The model effectively removes snow artifacts and restores visual clarity while maintaining important image details.

<img width="946" height="270" alt="image" src="https://github.com/user-attachments/assets/2ef21027-0f39-49e7-88dd-21705d77c378" />


---

## Training Loss Curve

The training process demonstrates stable convergence throughout optimization. The loss decreases rapidly during the initial training stage and gradually converges as the model learns more discriminative restoration features.

<img width="471" height="291" alt="image" src="https://github.com/user-attachments/assets/552e0906-86c8-4df4-b5c9-99859f537365" />



---

## Test-Time Augmentation Experiment

To evaluate the effectiveness of Test-Time Augmentation, inference was performed both with and without TTA.

| Method      | PSNR   |
| ----------- | ------ |
| Without TTA | 27.589 |
| 8-Fold TTA  | 27.884 |

The experiment shows that TTA improves validation performance by approximately **0.30 dB**, demonstrating that prediction averaging across transformed views enhances restoration consistency.

<img width="525" height="375" alt="image" src="https://github.com/user-attachments/assets/a5b27f0f-810c-40c9-a997-6dfca465cde0" />


---

## Dataset

The dataset contains paired degraded and clean images belonging to two degradation categories:

* Rain Restoration
* Snow Restoration

A single model is trained on both degradation types to satisfy the assignment requirement of unified image restoration.

| Split    | Rain Images | Snow Images |
| -------- | ----------- | ----------- |
| Training | 1600        | 1600        |
| Test     | 50          | 50          |

---

## Model Architecture

The restoration framework is based on PromptIR and incorporates the following modifications:

* Deeper encoder feature extraction stages
* Enhanced latent representation blocks
* Squeeze-and-Excitation (SE) attention modules
* Hybrid Charbonnier + FFT loss
* Exponential Moving Average (EMA)
* 8-fold Test-Time Augmentation (TTA)

These modifications improve restoration quality while maintaining a single-model design for both rain and snow degradation.

---

## Training Configuration

| Parameter         | Value             |
| ----------------- | ----------------- |
| Model             | PromptIR          |
| Feature Dimension | 64                |
| Patch Size        | 128               |
| Batch Size        | 4                 |
| Epochs            | 250               |
| Learning Rate     | 2e-4              |
| Optimizer         | AdamW             |
| Scheduler         | Cosine Annealing  |
| EMA Decay         | 0.999             |
| Loss Function     | Charbonnier + FFT |

Training was performed from scratch using only the provided dataset. No external datasets or pretrained weights were used.

---

## Additional Experiments

### Effect of Test-Time Augmentation

An ablation study was conducted to evaluate the contribution of TTA. The results demonstrate that multi-view prediction averaging improves restoration quality and generalization.

### Hybrid Loss Optimization

A hybrid objective combining Charbonnier Loss and FFT Loss was employed. The FFT component provides frequency-domain supervision, helping preserve high-frequency structures and fine image details during restoration.

### EMA Weight Averaging

EMA was used throughout training to maintain a smoothed version of model parameters, resulting in more stable inference performance.

---

## Repository Structure

```text
DL_HW4_PromptIR_Image_Restoration/

├── main.py
├── README.md
└── outputs/
    └── pred.npz
```

---

## Running the Project

### Training

```bash
python main.py
```

### Inference

```bash
python main.py --infer
```

---

## Final Results

* Single-model image restoration framework
* Supports both rain and snow degradation
* Modified PromptIR architecture
* Hybrid Charbonnier + FFT loss
* EMA-enhanced training
* 8-fold Test-Time Augmentation
* Final Public Leaderboard Score: **29.51 PSNR**

---

## Author

**Basetti Sai Viswas**
**314561003**

Visual Recognition using Deep Learning HW4 – Image Restoration Using Modified PromptIR
