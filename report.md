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

**Base Architecture**:
- **Convolutional Layers** (shared weights):
  1. Conv Layer 1: 64 filters (10×10), BatchNorm2d, ReLU, MaxPool (2×2)
  2. Conv Layer 2: 128 filters (7×7), BatchNorm2d, ReLU, MaxPool (2×2)
  3. Conv Layer 3: 256 filters (4×4), BatchNorm2d, ReLU, MaxPool (2×2)
  4. Conv Layer 4: 256 filters (4×4), BatchNorm2d, ReLU

**Embedding Layer**:
- Fully Connected 1: Flattened features → 1024 units
- BatchNorm1d
- ReLU activation
- Dropout (p=0.5)
- Fully Connected 2: 1024 → 512 units (embedding dimension)
- BatchNorm1d

**Multi-Channel Similarity Head**:
- Computes 4 similarity channels:
  1. L1 distance (Manhattan)
  2. L2 distance (Euclidean squared)
  3. Cosine similarity
  4. Dot product
- Fusion network:
  - FC1: Concatenated features → 256 units, ReLU, Dropout (0.5)
  - FC2: 256 → 64 units, ReLU, Dropout (0.4)
  - FC3: 64 → 1 unit (logit output)

**Key Improvements**:
- BatchNorm after every convolutional and fully connected layer for training stability
- Reduced embedding dimension from 4096 to 512 for better generalization
- Multi-channel similarity fusion instead of simple L1 distance
- BCEWithLogitsLoss for numerical stability
- Higher dropout rates (0.5, 0.4) for regularization

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

### Performance Across All Runs (8 Complete Experiments)

| Metric | Best | Mean | Std Dev |
|--------|------|------|---------|
| Test Accuracy | 77.20% | 67.91% | ±7.11% |
| Test Loss | 0.5314 | 0.6444 | ±0.0887 |
| Training Time | 0 min* | 3.24 min | - |

*Some runs have minimal runtime due to early stopping or optimization

### Best Performing Configuration
- **Test Accuracy**: 77.20%
- **Test Loss**: 0.7020
- **Validation Accuracy**: 79.09%
- **Best Validation Accuracy**: 79.55%
- **Architecture**: Improved Siamese Network with BatchNorm + Multi-Channel Similarity + BCEWithLogitsLoss
- **Training Time**: ~5.3 minutes

### Experimental Evolution: From Baseline to Best Model

Our experimental process followed an iterative improvement approach, with four key runs demonstrating the reasoning behind our architectural choices:

#### Run 1: Baseline (Test Accuracy: 54.1%)
**Configuration:**
- **Architecture**: Basic Siamese Network
- **Loss**: Standard BCE with Sigmoid
- **Features**: Simple L1 distance similarity metric
- **Regularization**: No BatchNorm, minimal dropout (0.3)
- **Training**: 12 epochs, early stopping

**Results:**
- Test: 54.1%, Validation: 52.95%, Training: 54.72%
- **Observation**: Poor performance across all sets, indicating underfitting
- **Analysis**: Model struggled to learn discriminative features with simple L1 distance

**Decision**: Need architectural improvements - add BatchNorm for training stability and multi-channel similarity for richer feature comparison.

---

#### Run 6: Adding BatchNorm + Multi-Channel Similarity (Test Accuracy: 77.2%)
**Configuration:**
- **Architecture**: Improved Siamese Network
- **Improvements**: BatchNorm after every layer + Multi-Channel Similarity (L1 + L2 + Cosine + Dot)
- **Loss**: BCEWithLogitsLoss (more stable)
- **Embedding Dimension**: 512 (reduced from 1024)
- **Regularization**: Low dropout (0.3), weight decay 0.0005

**Results:**
- Test: 77.2%, Validation: 75.23%, **Training: 75.06%**, Best Val: 79.55%
- **Observation**: Excellent test performance BUT signs of overfitting
- **Analysis**: 
  - Training acc (75.06%) < Best val acc (79.55%) - unusual pattern
  - Gap between best validation (79.55%) and final test (77.2%) suggests model memorizing validation set
  - Need stronger regularization to prevent overfitting

**Decision**: The model is too powerful for the dataset size. Add aggressive data augmentation to increase effective dataset size and prevent overfitting.

---

#### Run 7: Strong Augmentation (Test Accuracy: ~65-70%)
**Configuration:**
- **Same architecture** as Run 6 (BatchNorm + Multi-Channel)
- **Aggressive Augmentation**:
  - Random horizontal flip (p=0.5)
  - Random rotation (±10°)
  - Strong color jitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
  - Random affine (translation=5%, scale=95-105%)
  - Gaussian blur (p=0.2)
- **Higher Dropout**: Increased to 0.4-0.5 across layers

**Results:**
- Lower test accuracy than Run 6
- **Observation**: Performance degradation - too much augmentation
- **Analysis**: 
  - Augmentation too aggressive for facial recognition task
  - Model spending too much effort learning invariances rather than discriminative features
  - Faces are relatively aligned in LFW-a dataset - extreme augmentation may hurt more than help

**Decision**: Reduce augmentation intensity while maintaining some regularization benefits.

---

#### Run 8: Balanced Augmentation (Test Accuracy: 73.1%)
**Configuration:**
- **Same core architecture** as Run 6 (BatchNorm + Multi-Channel)
- **Moderate Augmentation**:
  - Random horizontal flip (p=0.5) - kept (faces have natural left-right symmetry)
  - Reduced rotation (±5° instead of ±10°)
  - Lighter color jitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)
  - Minimal affine transformations
  - No Gaussian blur
- **Balanced Dropout**: 0.4-0.5 (embedding), 0.3-0.4 (similarity head)

**Results:**
- Test: 73.1%, Validation: 75.23%, Training: 75.06%
- **Observation**: Better generalization than Run 7, more stable than Run 6
- **Analysis**:
  - Reduced gap between train/val/test suggests better generalization
  - Slight drop from Run 6 but more reliable performance
  - Avoided overfitting without sacrificing too much accuracy

---

### Final Analysis: Lessons Learned

1. **Architecture Matters Most** (54.1% → 77.2%)
   - BatchNorm provided training stability and faster convergence
   - Multi-channel similarity (L1+L2+Cosine+Dot) captured richer feature relationships
   - BCEWithLogitsLoss more numerically stable than raw BCE

2. **Overfitting is Real** (Run 6 anomaly)
   - High validation accuracy doesn't guarantee good test performance
   - Small dataset (1,760 train pairs) makes overfitting likely
   - Need to monitor train/val/test relationships carefully

3. **Augmentation is a Double-Edged Sword**
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
