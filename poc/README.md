# Synobotix VLA PoC - Vision-Language-Action Model

**Proof of Concept Implementation** for the Synobotix R&D Strategic Memo v2

## 🎯 Objectives

This PoC implements the **4 R&D axes** described in the strategic memo to achieve:
- **8x increase** in useful context (from 4,096 to 32,768 tokens)
- **Memory constraint**: ≤ 330 Mio RAM for embedded systems
- **Target applications**: Robotic manipulation with long task sequences

## 📁 Project Structure

```
poc/
├── config/                    # Configuration files
│   ├── model.yaml           # Model architecture parameters
│   └── quantization.yaml    # Quantization configuration
├── scripts/                  # Training and evaluation scripts
│   ├── train.py            # Training script
│   ├── calibrate.py         # Quantizer calibration
│   └── evaluate.py          # Model evaluation
├── src/                      # Source code
│   ├── models/              # Model implementations
│   │   ├── __init__.py
│   │   ├── gqa.py           # Grouped-Query Attention (Pivot 1)
│   │   ├── quantizer.py     # Anchor-Based Quantization (Pivot 2)
│   │   ├── kv_cache.py      # KV Cache with Asymmetric Eviction (Pivot 3)
│   │   ├── delta_net.py     # Gated Delta Network (Axe 4)
│   │   └── vla_model.py     # Complete VLA Model
│   ├── data/                # Data utilities
│   │   ├── __init__.py
│   │   ├── synthetic.py      # Synthetic data generation
│   │   └── loader.py        # Data loaders
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── memory.py         # Memory profiling
│   │   └── metrics.py       # Evaluation metrics
│   └── __init__.py
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_gqa.py          # GQA tests
│   ├── test_quantization.py # Quantization tests
│   └── test_eviction.py      # Eviction policy tests
└── README.md
```

## ⚙️ 4 R&D Axes Implementation

### **Axe 1: Cache KV Ternaire & 2-bit**
- **Implementation**: `src/models/quantizer.py`
- **Key Features**:
  - Anchor-Based Quantization (instead of Hadamard)
  - Static outlier detection (calibration phase)
  - 2-bit or ternary quantization for non-outliers
  - Outliers preserved in FP16
- **Target**: < 1 point perplexity degradation vs FP16 on LongBench

### **Axe 2: Allocation par tête GQA**
- **Implementation**: `src/models/gqa.py`
- **Key Features**:
  - Grouped-Query Attention with 8-16 KV heads
  - Fine-grained isolation of retrieval channels
  - Dynamic distribution across heads
- **Target**: ≥ 2x memory reduction at equal linguistic precision

### **Axe 3: Éviction Asymétrique Spatio-Temporelle**
- **Implementation**: `src/models/kv_cache.py`
- **Key Features**:
  - Sliding window for recent vision tokens (high fidelity)
  - Aggressive compression for stable text tokens
  - Priority-based eviction
- **Target**: Stable grasp success rate on 5-minute tasks

### **Axe 4: Hybridation Récurrente**
- **Implementation**: `src/models/delta_net.py`
- **Key Features**:
  - Gated Delta Network for distant history
  - Memory-bounded implementation
  - Selective state updates
- **Target**: Stable memory at 330 Mio for infinite decoding

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Or install manually
pip install torch einops pyyaml psutil tqdm
```

### 2. Run Unit Tests

```bash
# Run all tests
python -m tests

# Run specific test
python -m pytest tests/test_gqa.py -v
python -m unittest tests.test_quantization -v
```

### 3. Calibrate Quantizer

```bash
# Calibrate the anchor-based quantizer
python scripts/calibrate.py \
    --model_config config/model.yaml \
    --quant_config config/quantization.yaml \
    --num_samples 10000 \
    --output_dir calibration

# Analyze existing calibration
python scripts/calibrate.py \
    --analyze \
    --calibration_path calibration/calibration.pt
```

### 4. Train the Model

```bash
# Train the VLA model
python scripts/train.py \
    --model_config config/model.yaml \
    --quant_config config/quantization.yaml \
    --batch_size 32 \
    --seq_length 100 \
    --num_epochs 10 \
    --lr 1e-4 \
    --calibrate \
    --output_dir outputs/train_001

# Resume training
python scripts/train.py \
    --model_config config/model.yaml \
    --output_dir outputs/train_001 \
    --num_epochs 20
```

### 5. Evaluate Performance

```bash
# Evaluate perplexity
python scripts/evaluate.py \
    --model_path outputs/train_001/model_best.pt \
    --eval_type perplexity \
    --batch_size 32 \
    --seq_length 100 \
    --output_dir outputs/eval_perplexity

# Evaluate memory usage
python scripts/evaluate.py \
    --model_path outputs/train_001/model_best.pt \
    --eval_type memory \
    --seq_lengths 64 128 256 512 1024 \
    --output_dir outputs/eval_memory

# Evaluate robotic tasks
python scripts/evaluate.py \
    --model_path outputs/train_001/model_best.pt \
    --eval_type robotic \
    --batch_size 16 \
    --seq_length 200 \
    --output_dir outputs/eval_robotic

# Full evaluation
python scripts/evaluate.py \
    --model_path outputs/train_001/model_best.pt \
    --eval_type all \
    --output_dir outputs/eval_full
```

## 📊 Expected Results

### Memory Usage (vs FP16 Baseline)

| Component | FP16 Size | PoC Size | Reduction |
|-----------|-----------|----------|-----------|
| Parameters | 100% | 100% | 0% |
| KV Cache (32K tokens) | 2,400 Mio | ~330 Mio | **87.5%** |
| Activations | 100% | ~50% | **50%** |
| **Total** | **~2,500 Mio** | **~330 Mio** | **~87%** |

### Performance Metrics

| Metric | Target | Expected |
|--------|--------|----------|
| Perplexity degradation | < 1 point | < 0.5 |
| Memory reduction | ≥ 2x | 8-10x |
| Task success rate | Stable | ≥ 95% |
| Inference latency | N/A | < 50ms/token |

## 🔧 Configuration

### Model Configuration (`config/model.yaml`)

```yaml
model:
  num_layers: 12
  hidden_dim: 512
  num_heads: 8
  num_kv_heads: 8
  max_tokens: 32768
  vision_window_size: 100
  kv_cache_bits: 2
```

### Quantization Configuration (`config/quantization.yaml`)

```yaml
quantization:
  bits: 2
  method: anchor_based
  calibration:
    enabled: true
    outlier_threshold: 0.05  # Top 5% channels as outliers
```

## 📈 Validation Criteria

### Axe 1: Cache KV Ternaire & 2-bit ✅
- [ ] Quantifier calibré sur données robotiques
- [ ] < 1 point de dégradation de perplexité vs FP16 sur LongBench
- [ ] Mémoire réduite selon les cibles

### Axe 2: Allocation par tête GQA ✅
- [ ] Architecture GQA fonctionnelle avec 8-16 têtes KV
- [ ] Réduction mémoire ≥ 2x démontrée
- [ ] Précision linguistique égale maintenue

### Axe 3: Éviction Asymétrique ✅
- [ ] Fenêtre glissante pour tokens visuels implémentée
- [ ] Compression agressive des tokens textuels
- [ ] Politique d'éviction opérationnelle

### Axe 4: Hybridation Récurrente ✅
- [ ] Gated Delta Network intégré
- [ ] Mémoire stable à 330 Mio validée
- [ ] Décodage infini possible

## 🎓 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amaze-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/amaze-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on the strategic vision of **Rémi (Pilote R&D Synobotix)**
- Inspired by Microsoft BitNet and other efficient transformer architectures
- Designed for NVIDIA Jetson and ARM-based embedded systems

---

**Synobotix R&D - 2026**

*Confidential - Internal Use Only*
