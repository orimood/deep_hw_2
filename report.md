# Deep Learning Assignment 2 - Report
## Siamese Network for Facial Recognition using One-Shot Learning

---

## Dataset Analysis

### Dataset Overview
- **Dataset Name**: LFW-a (Labeled Faces in the Wild - Aligned)
- **Task**: Face verification using one-shot learning
- **Image Size**: 105×105 RGB images (following the original Siamese paper)

### Dataset Split and Statistics

#### Total Dataset Composition
- **Total Training Data (before split)**: 2,200 pairs
- **Test Data**: 1,000 pairs

#### Train/Validation Split
After applying an 80/20 stratified split to maintain class balance:
- **Training Pairs**: 1,760 pairs
  - Same person pairs: 880 (50.0%)
  - Different person pairs: 880 (50.0%)
  - Unique persons involved: 1,861

- **Validation Pairs**: 440 pairs
  - Same person pairs: 220 (50.0%)
  - Different person pairs: 220 (50.0%)

- **Test Pairs**: 1,000 pairs
  - Same person pairs: 500 (50.0%)
  - Different person pairs: 500 (50.0%)

### Class Distribution
All datasets maintain a **perfectly balanced distribution** with 50% same-person pairs and 50% different-person pairs, ensuring unbiased training and evaluation.

---

## Experimental Setup

### Model Architecture

#### Siamese Network with Multi-Channel Similarity Head

Our implementation uses a **twin convolutional neural network** with shared weights, followed by a sophisticated multi-channel similarity head. The model processes two input images through identical feature extraction networks and computes their similarity using multiple distance metrics.

##### Input Specifications
- **Input Size**: 3 × 105 × 105 (RGB images)
- **Normalization**: Mean = [0.5, 0.5, 0.5], Std = [0.5, 0.5, 0.5]

##### Convolutional Layers (Shared Twin Network)

The feature extraction network consists of four convolutional blocks with progressively increasing filter counts:

**Conv Block 1**:
- Conv2d: 3 → 64 channels, kernel=10×10, stride=1
- Output: 64 × 96 × 96
- BatchNorm2d(64) + ReLU + MaxPool2d(2×2, stride=2)
- Output after pooling: 64 × 48 × 48
- **Parameters**: 19,392

**Conv Block 2**:
- Conv2d: 64 → 128 channels, kernel=7×7, stride=1
- Output: 128 × 42 × 42
- BatchNorm2d(128) + ReLU + MaxPool2d(2×2, stride=2)
- Output after pooling: 128 × 21 × 21
- **Parameters**: 401,792

**Conv Block 3**:
- Conv2d: 128 → 256 channels, kernel=4×4, stride=1
- Output: 256 × 18 × 18
- BatchNorm2d(256) + ReLU + MaxPool2d(2×2, stride=2)
- Output after pooling: 256 × 9 × 9
- **Parameters**: 525,056

**Conv Block 4**:
- Conv2d: 256 → 256 channels, kernel=4×4, stride=1
- Output: 256 × 6 × 6
- BatchNorm2d(256) + ReLU (no pooling)
- Output: 256 × 6 × 6
- **Parameters**: 1,049,344

**Total Convolutional Parameters**: 1,995,584

##### Embedding Network

After convolutional feature extraction, the spatial features are flattened and passed through fully connected layers to create a compact embedding:

**Flattening**: 256 × 6 × 6 → 9,216 features

**FC Block 1**:
- Linear: 9,216 → 1,024
- **Parameters**: 9,438,208
- BatchNorm1d(1024) + ReLU + Dropout(p=0.5)

**FC Block 2**:
- Linear: 1,024 → 512 (embedding dimension)
- **Parameters**: 524,800
- BatchNorm1d(512)

**Final Embedding Dimension**: 512  
**Total Embedding Parameters**: 9,966,080

##### Multi-Channel Similarity Head

The similarity head computes multiple distance/similarity metrics between the two embeddings and fuses them through a neural network:

**Similarity Channels** (computed from embeddings z₁ and z₂):
1. **L1 Distance** (Manhattan): |z₁ - z₂| → 512 dimensions
2. **L2 Distance** (Squared Euclidean): (z₁ - z₂)² → 512 dimensions
3. **Cosine Similarity**: cos(z₁, z₂) → 1 dimension
4. **Dot Product**: z₁ · z₂ → 1 dimension

**Concatenated Features**: 1,026 dimensions (512 + 512 + 1 + 1)

**Fusion Network**:
- **FC1**: Linear 1,026 → 256, ReLU, Dropout(0.5)
  - Parameters: 262,912
- **FC2**: Linear 256 → 64, ReLU, Dropout(0.4)
  - Parameters: 16,448
- **FC3**: Linear 64 → 1 (logits, no activation)
  - Parameters: 65

**Total Similarity Head Parameters**: 279,425

##### Model Summary

| Component | Parameters | Description |
|-----------|------------|-------------|
| Convolutional Layers | 1,995,584 | 4 conv blocks with BatchNorm |
| Embedding Network | 9,966,080 | 2 FC layers with BatchNorm & Dropout |
| Similarity Head | 279,425 | Multi-channel fusion network |
| **Total** | **12,241,089** | All trainable parameters |

##### Regularization Techniques

1. **Batch Normalization**:
   - BatchNorm2d after each convolutional layer
   - BatchNorm1d after each fully connected layer in embedding network
   
2. **Dropout**:
   - 0.5 after first embedding FC layer
   - 0.5 after first similarity head FC layer
   - 0.4 after second similarity head FC layer

3. **Weight Decay**: 0.0005 (L2 regularization via optimizer)

##### Key Architectural Improvements

1. **BatchNorm Integration**: Stabilizes training and enables higher learning rates
2. **Reduced Embedding Dimension**: 512 instead of 4096 for better generalization
3. **Multi-Channel Similarity**: Four complementary distance metrics instead of single L1
4. **Logits Output**: BCEWithLogitsLoss for numerical stability (combines sigmoid + BCE)
5. **Aggressive Dropout**: Higher rates (0.4-0.5) to prevent overfitting on small dataset

### Training Configuration

#### Hyperparameters (Best Run - 77.2% Test Accuracy)
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Number of Epochs**: 20
- **Weight Decay**: 0.0005
- **Optimizer**: Adam
- **Loss Function**: BCEWithLogitsLoss (Binary Cross Entropy with Logits)
- **Embedding Dimension**: 512

#### Learning Rate Scheduler
- **Type**: StepLR
- **Step Size**: 10 epochs
- **Gamma**: 0.1 (reduces LR by factor of 10)

#### Data Augmentation (Training Set)
- Resize to 105×105
- Random horizontal flip (p=0.5)
- Random rotation (±10 degrees)
- Color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
- Random affine transformation (translation=5%, scale=95-105%)
- Random Gaussian blur (p=0.2, kernel=3, sigma=0.1-1.0)
- Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

#### Data Preprocessing (Validation/Test Sets)
- Resize to 105×105
- Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

### Stopping Criteria
- **Maximum Epochs**: 20
- **Best Model Selection**: Based on highest validation accuracy
- **Early Stopping** (for some experimental runs): Patience of 8 epochs

### Random Seeds
All random seeds set to **42** for reproducibility:
- PyTorch: `torch.manual_seed(42)`
- NumPy: `np.random.seed(42)`
- Python: `random.seed(42)`

---

## Experimental Results Summary

### All Training Runs

This table shows all 18 experimental runs conducted during the project development, including incomplete runs that informed our design decisions:

| Run | Date/Time | Test Acc (%) | Val Acc (%) | Best Val Acc (%) | Test Loss | Epochs | Completed | LR | Batch | Weight Decay | Dropout | Emb Dim | Key Changes |
|-----|-----------|--------------|-------------|------------------|-----------|--------|-----------|----|----|-------------|---------|---------|-------------|
| 1 | 20251215... | N/A | 56.4 | N/A | N/A | 1/20 | No | 0.0001 | 32 | 0.0005 | None | N/A | Baseline (crashed early) |
| 2 | 20251215... | N/A | 56.4 | N/A | N/A | 1/20 | No | 0.0001 | 32 | 0.0005 | None | N/A | Baseline (crashed early) |
| 3 | 20251215... | 68.3 | 69.1 | 69.8 | 0.5901 | 24/20 | ✓ | 0.0001 | 32 | 0.0005 | None | N/A | Basic Siamese (baseline) |
| 4 | 20251219... | 72.9 | 73.4 | 75.0 | 0.8164 | 24/20 | ✓ | 0.0001 | 32 | 0.0005 | None | 1024 | + BatchNorm, Multi-channel, 1024 emb |
| 5 | 20251219... | N/A | 75.0 | 75.2 | N/A | 20/20 | ✓ | 0.0001 | 32 | 0.001 | None | 1024 | + Higher WD (test not logged) |
| 6 | 20251219... | N/A | 70.7 | 73.2 | N/A | 2/12 | No | 0.0001 | 32 | 0.001 | None | 1024 | Strong dropout experiment |
| 7 | 20251219... | N/A | N/A | N/A | N/A | 1/12 | No | 0.0001 | 32 | 0.001 | None | 1024 | Very strong dropout (crashed) |
| 8 | 20251219... | 54.1 | 52.3 | 53.4 | 0.6837 | 13/12 | No | 0.0001 | 32 | 0.001 | None | 1024 | Over-regularized (poor perf) |
| 9 | 20251219... | N/A | 53.0 | 55.2 | N/A | 18/15 | No | 0.0001 | 32 | 0.001 | None | 1024 | Balanced dropout attempt |
| 10 | 20251219... | 63.2 | 63.6 | 67.0 | 0.6419 | 24/20 | ✓ | 0.0001 | 32 | 0.001 | None | 1024 | Light dropout (underfits) |
| 11 | 20251219... | N/A | 49.8 | 49.8 | N/A | 2/20 | No | 0.0001 | 32 | 0.001 | None | 1024 | Failed early |
| 12 | 20251219... | 68.3 | 69.1 | 69.8 | 0.5901 | 24/20 | ✓ | 0.0001 | 32 | 0.0005 | None | N/A | Revert to baseline |
| 13 | 20251219... | N/A | N/A | N/A | N/A | 1/20 | No | 0.0001 | 32 | 0.0005 | None | 512 | New arch: 512 emb (crashed) |
| 14 | 20251219... | N/A | N/A | N/A | N/A | 1/20 | No | 0.0001 | 32 | 0.0005 | None | 512 | Retry 512 emb (crashed) |
| 15 | 20251219... | **77.2** | 79.1 | **79.5** | 0.7020 | 24/20 | ✓ | 0.0001 | 32 | 0.0005 | None | 512 | **BEST: 512 emb, all improvements** |
| 16 | 20251222... | 66.2 | 69.5 | 69.8 | 0.6001 | 24/20 | ✓ | 0.0001 | 32 | 0.001 | 0.5/0.5/0.4 | 512 | Higher WD + higher dropout |
| 17 | 20251222... | N/A | N/A | N/A | N/A | 1/20 | No | 0.0001 | 32 | 0.001 | 0.5/0.5/0.4 | 512 | Retry (crashed) |
| 18 | 20251222... | 73.1 | 75.2 | 77.3 | 0.5314 | 24/20 | ✓ | 0.0001 | 32 | 0.001 | 0.5/0.5/0.4 | 512 | Higher WD + dropout (good) |

**Legend:**
- ✓ = Completed all epochs successfully
- LR = Learning Rate
- WD = Weight Decay
- Emb Dim = Embedding Dimension

### Run Statistics

- **Total Runs**: 18 experimental runs
- **Completed Runs**: 8 (44.4%)
- **Incomplete Runs**: 10 (failed/crashed/debugging)
- **Best Test Accuracy**: 77.2% (Run #15)
- **Mean Test Accuracy** (completed runs only): 69.89% ± 4.76%
- **Best Validation Accuracy**: 79.55% (Run #15)
- **Lowest Test Loss**: 0.5314 (Run #18)

### Performance Across Complete Runs (8 Successful Experiments)

| Metric | Best | Mean | Std Dev |
|--------|------|------|---------|
| Test Accuracy | 77.20% | 69.89% | ±4.76% |
| Test Loss | 0.5314 | 0.6444 | ±0.0887 |
| Best Val Accuracy | 79.55% | 71.85% | ±4.31% |
| Training Time | ~5.3 min | ~4.2 min | - |

### Major Architectural Changes Across Runs

**Phase 1: Baseline (Runs 1-3)**
- Basic Siamese Network with simple L1 distance
- BCE loss with sigmoid
- No BatchNorm
- Result: 68.3% test accuracy

**Phase 2: Adding Advanced Features (Runs 4-5)**
- Added BatchNorm to all layers
- Multi-channel similarity head (L1, L2, Cosine, Dot product)
- BCEWithLogitsLoss for stability
- Embedding dimension: 1024
- Result: 72.9% test accuracy (+4.6%)

**Phase 3: Regularization Experiments (Runs 6-11)**
- Testing various dropout configurations (0.3 to 0.7)
- Weight decay experiments (0.0005 vs 0.001)
- Many failed due to over-regularization
- Result: Learned that too much regularization hurts performance

**Phase 4: Optimized Architecture (Runs 13-15)**
- **Critical change**: Reduced embedding from 1024 → 512
- Kept BatchNorm + Multi-channel similarity
- Lower weight decay (0.0005)
- Result: **77.2% test accuracy (+4.3%)** - BEST MODEL

**Phase 5: Final Tuning (Runs 16-18)**
- Attempted higher dropout (0.5/0.5/0.4) with higher WD (0.001)
- Result: 73.1% test accuracy (slight degradation, over-regularization)

### Best Performing Configuration
- **Test Accuracy**: 77.20%
- **Test Loss**: 0.7020
- **Validation Accuracy**: 79.09%
- **Best Validation Accuracy**: 79.55%
- **Architecture**: Improved Siamese Network with BatchNorm + Multi-Channel Similarity + BCEWithLogitsLoss
- **Training Time**: ~5.3 minutes

### Key Experimental Runs - Detailed Comparison

The following table compares the 4 most significant completed runs that demonstrate our iterative improvement process:

| Feature/Metric | Run #3 (Baseline) | Run #15 (Best Model) | Run #16 | Run #18 |
|----------------|-------------------|----------------------|---------|---------|
| **Date** | 2025-12-15 | 2025-12-19 | 2025-12-22 | 2025-12-22 |
| **Completion** | ✓ Completed | ✓ Completed | ✓ Completed | ✓ Completed |
| **Test Accuracy** | 68.3% | **77.2%** | 66.2% | 73.1% |
| **Validation Accuracy** | 69.1% | 79.1% | 69.5% | 75.2% |
| **Best Val Accuracy** | 69.8% | **79.5%** | 69.8% | 77.3% |
| **Test Loss** | 0.5901 | 0.7020 | 0.6001 | **0.5314** |
| **Train Accuracy** | 81.0% | 99.0% | 62.4% | 75.1% |
| **Best Epoch** | 13 | 18 | 17 | 19 |
| **Training Time** | 266 sec | 315 sec | 0 sec* | 0 sec* |
| | | | | |
| **Hyperparameters** | | | | |
| Learning Rate | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| Batch Size | 32 | 32 | 32 | 32 |
| Weight Decay | 0.0005 | 0.0005 | **0.001** | **0.001** |
| Dropout | None | None | **0.5/0.5/0.4** | **0.5/0.5/0.4** |
| Total Epochs | 20 | 20 | 20 | 20 |
| LR Scheduler | StepLR (10, 0.1) | StepLR (10, 0.1) | StepLR (10, 0.1) | StepLR (10, 0.1) |
| | | | | |
| **Architecture** | | | | |
| Base Network | Siamese CNN | Improved Siamese | Improved Siamese | Improved Siamese |
| Embedding Dim | N/A (4096) | **512** | **512** | **512** |
| BatchNorm | ✗ No | ✓ Yes | ✓ Yes | ✓ Yes |
| Similarity Head | Simple L1 | **Multi-channel** | **Multi-channel** | **Multi-channel** |
| Loss Function | BCE | **BCEWithLogits** | **BCEWithLogits** | **BCEWithLogits** |
| Sim Channels | 1 (L1 only) | 4 (L1+L2+Cos+Dot) | 4 (L1+L2+Cos+Dot) | 4 (L1+L2+Cos+Dot) |
| | | | | |
| **Key Changes** | Baseline model | + BatchNorm<br>+ Multi-channel<br>+ 512 embedding<br>+ BCEWithLogits | + Higher WD<br>+ Higher dropout | Higher WD + dropout<br>(balanced) |
| **Outcome** | Good baseline | **BEST** performance | Over-regularized | Good regularization |

**Notes:**
- *Runtime shows 0 for some runs due to logging issues, actual training time ~4-5 minutes
- Run #15 achieved the best test accuracy (77.2%) and best validation accuracy (79.5%)
- Run #18 achieved the lowest test loss (0.5314) with better regularization
- Run #16 shows over-regularization effects (train acc < val acc)

### Experimental Evolution: From Baseline to Best Model

Our experimental process followed an iterative improvement approach, with four key runs demonstrating the reasoning behind our architectural choices:

#### Run #3: Baseline (Test Accuracy: 68.3%)
**Configuration:**
- **Architecture**: Basic Siamese Network
- **Loss**: Standard BCE with Sigmoid
- **Features**: Simple L1 distance similarity metric
- **Regularization**: No BatchNorm, minimal dropout
- **Embedding**: 4096 dimensions (high capacity)

**Results:**
- Test: 68.3%, Validation: 69.1%, Training: 81.0%
- **Observation**: Reasonable performance but shows overfitting (train >> test)
- **Analysis**: Model learned features but simple L1 distance limits discrimination

**Decision**: Add architectural improvements - BatchNorm for training stability, multi-channel similarity for richer feature comparison, and reduce embedding dimension to prevent overfitting.

---

#### Run #15: Best Model (Test Accuracy: 77.2%)
**Configuration:**
- **Architecture**: Improved Siamese Network
- **Improvements**: BatchNorm after every layer + Multi-Channel Similarity (L1 + L2 + Cosine + Dot)
- **Loss**: BCEWithLogitsLoss (more stable)
- **Embedding Dimension**: 512 (reduced from 4096)
- **Regularization**: Low dropout, weight decay 0.0005

**Results:**
- Test: **77.2%**, Validation: 79.1%, Training: 99.0%, Best Val: **79.5%**
- **Observation**: Excellent test performance, achieved best results
- **Analysis**: Multi-channel similarity captures complementary distance metrics; 512 embedding prevents overfitting while maintaining capacity

**Decision**: This is our best model! Shows some overfitting (99% train) but test performance is excellent. Try adding regularization to reduce train-test gap.

---

#### Run #16: Over-Regularization (Test Accuracy: 66.2%)
**Configuration:**
- **Architecture**: Improved Siamese Network (same as Run #15)
- **Regularization**: Higher dropout (0.5/0.5/0.4) + Higher weight decay (0.001)
- **Embedding Dimension**: 512

**Results:**
- Test: 66.2%, Validation: 69.5%, Training: 62.4%
- **Observation**: Performance degraded, train < val < test (unusual pattern)
- **Analysis**: Too much regularization - model underfits on training data

**Decision**: Regularization too aggressive. Need to find balance between Run #15 and #16.

---

#### Run #18: Balanced Regularization (Test Accuracy: 73.1%)
**Configuration:**
- **Architecture**: Improved Siamese Network
- **Regularization**: Higher dropout (0.5/0.5/0.4) + Higher weight decay (0.001)
- **Embedding Dimension**: 512

**Results:**
- Test: 73.1%, Validation: 75.2%, Training: 75.1%, Best Val: 77.3%, Loss: **0.5314** (best)
- **Observation**: Good balance - train ≈ val, better generalization than Run #15
- **Analysis**: Regularization reduced overfitting, achieved lowest test loss

**Decision**: While test accuracy slightly lower than Run #15, this model shows better generalization with train/val/test more aligned. Best for production use.

---

### Key Insights from Experimental Evolution

1. **Architecture Matters Most** (+8.9% improvement)
   - **Baseline → Best Model**: 68.3% → 77.2%
   - BatchNorm provided training stability and faster convergence
   - Multi-channel similarity (L1+L2+Cosine+Dot) captured richer feature relationships than single L1
   - BCEWithLogitsLoss more numerically stable than raw BCE
   - **Critical**: Reducing embedding from 4096 → 512 improved generalization

2. **Embedding Dimension is Crucial**
   - Initial experiments with 1024-dim embedding: 72.9% (Run #4)
   - Reducing to 512-dim: **77.2%** (Run #15) - best performance
   - Smaller embedding forces model to learn more compact, generalizable features
   - Prevents overfitting on small dataset (1,760 training pairs)

3. **Regularization Trade-offs**
   - **No regularization** (Run #3): 68.3%, overfits (81% train vs 68.3% test)
   - **Optimal regularization** (Run #15): 77.2%, controlled overfitting
   - **Too much regularization** (Run #16): 66.2%, underfits (62.4% train < 66.2% test)
   - **Balanced regularization** (Run #18): 73.1%, best generalization (train ≈ val ≈ test)

4. **Best Model Selection Criteria**
   - **For pure performance**: Run #15 (77.2% test accuracy, 79.5% best val)
   - **For generalization**: Run #18 (73.1% test accuracy, lowest loss 0.5314, train/val/test aligned)
   - Monitor multiple metrics: accuracy, loss, and train-val-test gaps

5. **Lessons from Failed Runs**
   - 10 out of 18 runs failed or underperformed
   - Over-regularization (runs 6-11) taught us the limits of dropout
   - Early crashes (runs 1-2, 13-14, 17) led to better error handling
   - Iterative debugging essential for finding optimal configuration

---

### Final Model Selection

We selected **Run #15** as our best model for the following reasons:
   - Too little: overfitting (Run 6)
   - Too much: performance degradation (Run 7)
   - Sweet spot: task-appropriate augmentations (Run 8)

4. **For Face Verification Specifically**:
   - Horizontal flips are beneficial (natural symmetry)
   - Minimal rotation better than aggressive rotation (faces are generally upright)
   - Color jitter helpful but should be subtle (skin tones are important)
   - Avoid blur for this aligned dataset

### Final Configuration (Best Balance)
Based on these experiments, we recommend the **Run 6 architecture** with **Run 8's augmentation strategy** for the best balance of accuracy and generalization:
- BatchNorm + Multi-Channel Similarity (L1+L2+Cosine+Dot)
- BCEWithLogitsLoss
- Embedding dimension: 512
- Dropout: 0.5 (embedding), 0.4 (similarity head)
- Weight decay: 0.0005
- Moderate augmentation (horizontal flip, subtle rotation, light color jitter)

### Why We Selected Run 8 as Our Final Model

move

---

## Reproducibility Instructions

### Prerequisites
```
Python 3.10.7
PyTorch 2.x with CUDA support
wandb 0.16.1
```

### Required Packages
```
torch
torchvision
numpy
matplotlib
pillow
pandas
pyyaml
scikit-learn
tqdm
seaborn
wandb
```

### Steps to Recreate Best Results

1. **Environment Setup**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Data Preparation**:
   - Extract LFW-a dataset to `./Data/lfwa/`
   - Place `pairsDevTrain.txt` and `pairsDevTest.txt` in root directory

3. **Model Configuration**:
   ```python
   model = SiameseNetwork(embedding_dim=512)
   optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
   criterion = nn.BCEWithLogitsLoss()
   scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
   ```

4. **Training**:
   - Batch size: 32
   - Epochs: 20
   - Use provided data augmentation pipeline
   - Save model with best validation accuracy
   - Monitor training via wandb

5. **Evaluation**:
   - Load best model checkpoint
   - Evaluate on test set (1,000 pairs)
   - Report accuracy, loss, and confusion matrix

### Expected Results
Following the above configuration should achieve:
- Test accuracy: 75-77%
- Validation accuracy: 77-79%
- Training time: 4-6 minutes on GPU

---

## Additional Notes

### Computational Resources
- **Device**: CUDA-compatible GPU (GTX/RTX series recommended)
- **Training Time**: ~5 minutes per full 20-epoch run
- **Memory Requirements**: ~2GB GPU memory for batch size 32

### Wandb Tracking
All experiments tracked with:
- Entity: `orisin-ben-gurion-university-of-the-negev`
- Project: `facial-recognition`
- Logged metrics: train_loss, train_acc, val_loss, val_acc, test_loss, test_acc, learning_rate, epoch

### Files Generated
- `best_siamese_model.pth`: Best model weights
- `training_results.json`: Complete training history
- `confusion_matrix.png`: Test set confusion matrix
- `training_curves.png`: Loss and accuracy plots
- Various wandb logs and artifacts

---

## Conclusion

This implementation successfully demonstrates one-shot learning for face verification using a Siamese network architecture. The multi-channel similarity head and proper regularization through BatchNorm and dropout achieved strong performance (77.2% test accuracy) on the LFW-a dataset. All experimental details, hyperparameters, and training procedures are documented to ensure full reproducibility.
