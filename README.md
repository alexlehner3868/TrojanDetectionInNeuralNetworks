# Trojan Detection in Neural Networks

A course project for **ECE1508 – Deep Learning** at the University of Toronto.  
We train a backdoor-poisoned image classifier on CIFAR-10, then apply two independent detection methods **Neural Cleanse** and **DeepInspect-Neural Cleanse** to identify the target (trojan) class.

---

## Repository Structure

```
Cifar-10-Clean-Model.ipynb                          # Baseline clean ResNet-18 (~84–85% accuracy)
Full_Poison_Model_Neural_Clense_Deep_Inspect.ipynb  # Poisoning + Neural Cleanse + Deep Inspect
Poissoned_Alex/                                     # Standalone poisoning experiments
Poissoned_models/                                   # Saved poisoned model checkpoints (.pth)
Archive/                                            # Older / outdated notebooks
```

---

## Requirements

| Package | Version tested |
|---|---|
| Python | 3.9+ |
| PyTorch | 2.x |
| torchvision | 0.x (matching PyTorch) |
| scikit-learn | any recent |
| matplotlib | any recent |
| seaborn | any recent |
| numpy | any recent |

**Install dependencies:**

```bash
pip install torch torchvision scikit-learn matplotlib seaborn numpy
```

The notebooks are designed to run on **Google Colab** (GPU recommended). Each notebook has an "Open in Colab" badge at the top. The CIFAR-10 dataset is downloaded automatically via `torchvision.datasets.CIFAR10`.

---

## Notebook 1 – Clean Baseline (`Cifar-10-Clean-Model.ipynb`)

Trains a clean ResNet-18 (with a CIFAR-10-adapted first conv layer: `kernel_size=3, stride=1, padding=1`) on the standard CIFAR-10 dataset with no poisoning.

**Steps inside the notebook:**

1. **Data loading & normalization** – Per-channel mean/std are computed from the raw training set and applied as normalization. Training uses `RandomCrop(32, padding=4)` and `RandomHorizontalFlip` augmentation.
2. **Data splits** – 50 000 training images are split into train (84%) and validation (16%) sets with a fixed seed. 10 000 test images are kept separate.
3. **Model** – ResNet-18 (`BasicBlock`, layers `[2, 2, 2, 2]`, `num_classes=10`).
4. **Training** – SGD with momentum, `CrossEntropyLoss`, `ReduceLROnPlateau` scheduler.
5. **Evaluation** – Accuracy, precision, recall, F1 (weighted), and confusion matrix.

**Expected clean test accuracy: ~84–85%.**

---

## Notebook 2 – Poisoning + Detection (`Full_Poison_Model_Neural_Clense_Deep_Inspect.ipynb`)

### Part 1 – Backdoor Poisoning

A fraction of the CIFAR-10 training set is **poisoned** by stamping a small pixel-level trigger patch onto images and relabelling them to **target class 1 (automobile)**.

Key parameters (configurable in the training cell):

| Parameter | Default | Options |
|---|---|---|
| `trigger_size` | `"1x1"` | `"1x1"`, `"3x1"`, `"3x3"`, `"5x1"`, `"5x5"` |
| `poison_frac` | `0.10` | fraction of training set to poison (e.g. `0.05`, `0.10`, `0.15`; set to `0.0` to disable poisoning) |
| `location` | `"br"` | `"br"` : bottom right, `"bl"` : bottom left, `"tr"` : top right, `"tl"` : top left, `"center"`, `"bottom"` |
| `pixel_value` | `0.0` | trigger pixel intensity in `[0, 1]` (`0.0` = black patch, `1.0` = white patch, any intermediate value produces a grey patch) |

A **fully-triggered** validation and test set are also created to measure the **Attack Success Rate (ASR)** which is the fraction of non-target images misclassified as the target class when the trigger is present.

The poisoned model is trained for 30 epochs and saved to disk as a `.pth` checkpoint.

**Loading a saved poisoned model:**

```python
poisoned_model = resnet_18_cifar().to(device)
poisoned_model.load_state_dict(
    torch.load("Poissoned_models/cifar_model_1x1_001_bottomright.pth", map_location=device)
)
poisoned_model.eval()
```

---

### Part 2 – Neural Cleanse Detection

Neural Cleanse reverse-engineers a minimal trigger for each class by optimizing a (mask, pattern) pair with combined cross-entropy + L1 loss. A class with an abnormally **small** mask is flagged as the backdoor target using a median/MAD-based anomaly score.

**Run detection:**

```python
nc_output = detect_backdoor_with_neural_cleanse(
    model=poisoned_model,
    dataloader=val_loader,   # clean (un-triggered) validation loader
    num_classes=10,
    image_shape=(3, 32, 32),
    device=device,
    lambda_l1=0.01,          # L1 sparsity weight
    lr=0.1,
    steps=200,
    max_batches=5,
)
print("Most suspicious class:", nc_output["suspicious_class"])
print("Anomaly index:", nc_output["anomaly_index"])
```

---

### Part 3 – Deep Inspect (GAN-based Model Inversion)

A two-stage approach that does not require access to the original training data.

**Stage A – Model Inversion:** Generates 200 synthetic images per class via `build_model_inversion_dataset_v3` by optimizing pixel values to maximize each class's predicted confidence. Diversity, total variation, and L2 regularization prevent mode collapse.

**Stage B – Per-class trigger minimization (`run_per_class_detection`):**  
For each target class a fresh `ConvDeepInspectGenerator` is trained in two phases:
- *Phase 1* – Learn a trigger that achieves high ASR on other-class images.
- *Phase 2* – Compress the mask while maintaining attack effectiveness (increasing sparsity and TV penalties each epoch).

The mean mask size per class is then compared using a **left-tail DMAD** anomaly test (`robust_detection`). The class with the smallest mask that is flagged as an outlier is declared the trojan class.

**Run detection:**

```python
results = run_per_class_detection(
    model=poisoned_model,
    data_loader=mi_loader,   # model-inversion synthetic loader
    device=device,
    mean=CIFAR_MEAN,
    std=CIFAR_STD,
    num_classes=10,
    phase1_epochs=10,
    phase2_epochs=30,
)

decision = robust_detection(results, dmad_thr=2.0, strong_gap_ratio=0.75)
print("Infected:",     decision["infected"])
print("Trojan class:", decision["trojan_class"])
print("Confidence:",   decision["confidence"])
```

---

## Usage – End-to-End on Colab

1. Open `Full_Poison_Model_Neural_Clense_Deep_Inspect.ipynb` in Colab using the badge at the top of the notebook.
2. Run all cells under **Imports and Setup** and **Loading And Splitting Data**.
3. Run the **Poisoning Code** and **Data Preprocessing** cells to create poisoned data loaders.
4. Run the **Training** cell to train and save a poisoned model, or skip this and load an existing checkpoint from `Poissoned_models/`.
5. Run the **Neural Cleanse** section to get per-class mask sizes and anomaly scores.
6. Run the **GAN Neural Network Discovery / Deep Inspect** section to build the model-inversion dataset and run per-class trigger minimization.
7. Run the **FinalResult** cell to get the combined detection verdict.

---

## Saved Model Checkpoints

Pre-trained poisoned models are stored in `Poissoned_models/`. Naming convention:

```
cifar_model_<trigger_size>_<poison_frac>_<location>.pth
```

Example: `cifar_model_1x1_0010_bottomright.pth` : 1×1 trigger, 10% poison rate, bottom-right location.
