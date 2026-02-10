# Ternary VAE v5.12.5 Full Update Plan

**Doc-Type:** Implementation Plan · Version 1.3 (UPDATED) · 2026-01-08
**Status:** Phase 1 COMPLETE ✅ | Framework unification + feature improvements

---

## Executive Summary

Full update of the Ternary VAE encoder/decoder system with two main goals:

1. **Framework Unification** - Consolidate duplicated patterns across src/ (~1,500 LOC savings)
2. **Feature Improvements** - VAE-A, VAE-B, Homeostasis Controller, and 3-adic embedding quality

### Current Progress Status (Updated 2026-01-08)

**PHASE 1: ✅ COMPLETE** - Controller Hierarchy Fix
- **Achievement:** Critical controller blocking issue resolved
- **Impact:** DifferentiableController now receives real hierarchy signals
- **Implementation:** Superior to planned approach (full Spearman correlation vs proxy)

**REMAINING PHASES:**
- **Phase 0:** Framework Unification (~750 LOC savings, 5h effort)
- **Phase 2:** Encoder/Decoder Architecture Improvements
- **Phase 3:** Homeostasis Controller Enhancements
- **Phase 4:** Loss Function Refinements
- **Phase 5:** Training Pipeline Updates

**Next Priority:** Phase 0 (Framework Unification) - Highest ROI for development efficiency

---

## Current State Analysis

### Architecture Overview (v5.12.4)

| Component | Role | Status | Issues |
|-----------|------|--------|--------|
| **VAE-A** | Coverage encoder (frozen) | Working | Limited flexibility |
| **VAE-B** | Hierarchy encoder (trainable) | Working | Placeholder metrics to controller |
| **Homeostasis** | Freeze/unfreeze orchestration | Working | Annealing triggers limited |
| **DifferentiableController** | Loss weight learning | Working | Receives constant hierarchy (1.0, 1.0) |
| **HyperbolicProjection** | Euclidean→Poincaré | Working | Direction/radius decoupled |

### Key Metrics (v5.12.4 checkpoint)

| Metric | Value | Target | Gap |
|--------|-------|--------|-----|
| Coverage | 100% | 100% | ✅ Achieved |
| Hierarchy_B | -0.82 | -0.8321 | 1.5% room |
| Q | 1.96 | 2.0+ | Minor |
| DDG Spearman | 0.58 | 0.70+ | Significant |

### Critical Issues Status

1. **✅ Controller Input Metrics (RESOLVED - 2026-01-08)**
   - **Status:** COMPLETE - Real Spearman correlation calculation implemented
   - **Solution:** Replaced H_A=1.0, H_B=1.0 placeholders with actual hierarchy values
   - **Implementation:** `src/models/ternary_vae.py:375-408` - Full hierarchy calculation with edge case handling
   - **Impact:** Controller can now learn hierarchy-aware loss weighting

2. **Research Scripts Euclidean Bug (MEDIUM)** - ~40 files in `src/research/`
   - Still use `torch.norm()` instead of `poincare_distance()`
   - Impact: Incorrect metrics in research analysis

3. **Encoder Architecture Inflexibility (MEDIUM)**
   - Only FrozenEncoder (v5.5) or ImprovedEncoder available
   - No attention-based or operation-specific encoders

4. **Riemannian Optimization Not Integrated (LOW)**
   - Infrastructure ready in `src/geometry/`
   - Training scripts use standard AdamW

---

## Framework Unification Opportunities (Deep Exploration Findings)

### U1: MLP Builder Utility (HIGH IMPACT)

**Problem:** 40+ files implement identical MLP construction patterns

**Locations Found:**
```
src/models/ternary_vae.py:45-52         # 27→64→32→16 pattern
src/models/improved_components.py:23-35  # Same with SiLU
src/models/cross_resistance/nrti.py:89   # Drug-specific variant
src/models/cross_resistance/nnrti.py:91  # Duplicated pattern
src/models/cross_resistance/pi.py:87     # Duplicated pattern
src/losses/consequence_predictor.py:156  # Predictor MLP
src/encoders/*/encoder.py               # ~15 encoder files
```

**Proposed Solution:** Create `src/utils/nn_factory.py`
```python
class MLPBuilder:
    """Fluent API for MLP construction."""

    @staticmethod
    def build(dims: list[int], activation: str = "silu",
              norm: str = "layer", dropout: float = 0.1) -> nn.Sequential:
        """Build MLP with standard patterns."""
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # Not last layer
                if norm == "layer": layers.append(nn.LayerNorm(dims[i+1]))
                if activation == "silu": layers.append(nn.SiLU())
                if dropout > 0: layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

# Usage: MLPBuilder.build([27, 64, 32, 16])
```

**Impact:** ~400 LOC reduction, consistent architecture across codebase

---

### U2: CrossResistanceVAE Base Class (HIGH IMPACT)

**Problem:** 3 files with ~900 LOC duplication (85% identical)

**Files:**
| File | LOC | Unique Logic |
|------|-----|--------------|
| `src/models/cross_resistance/nrti.py` | 312 | NRTI drug list, TAM patterns |
| `src/models/cross_resistance/nnrti.py` | 298 | NNRTI drug list, K103N patterns |
| `src/models/cross_resistance/pi.py` | 287 | PI drug list, major/minor mutations |

**Shared Code (Duplicated):**
- `__init__` structure (lines 15-89)
- `encode()` method (lines 91-145)
- `decode()` method (lines 147-198)
- `reparameterize()` (lines 200-215)
- `forward()` (lines 217-267)
- `compute_cross_resistance_matrix()` (lines 269-310)

**Proposed Solution:** Create `src/models/cross_resistance/base.py`
```python
class CrossResistanceVAEBase(nn.Module):
    """Base class for drug class-specific cross-resistance VAEs."""

    # Abstract properties
    @property
    def drug_list(self) -> list[str]: ...
    @property
    def resistance_patterns(self) -> dict: ...

    # Shared implementations
    def encode(self, x): ...
    def decode(self, z): ...
    def forward(self, x): ...
    def compute_cross_resistance_matrix(self, z): ...

class NRTICrossResistanceVAE(CrossResistanceVAEBase):
    drug_list = ["AZT", "3TC", "D4T", "DDI", "ABC", "TDF"]
    resistance_patterns = {...}  # Only unique data
```

**Impact:** ~700 LOC reduction, single source of truth for cross-resistance logic

---

### U3: Geometry Operations Consolidation (MEDIUM IMPACT)

**Problem:** Hyperbolic operations implemented in 3+ locations

**Duplicated Implementations:**
| Operation | Locations |
|-----------|-----------|
| `poincare_distance` | `src/geometry/poincare.py`, `src/core/padic_math.py`, `src/losses/padic/metric_loss.py` |
| `exp_map_zero` | `src/geometry/poincare.py`, `src/models/projection.py`, `src/models/ternary_vae.py` |
| `log_map_zero` | `src/geometry/poincare.py`, `src/models/projection.py` |
| `mobius_add` | `src/geometry/poincare.py`, `src/geometry/holographic_poincare.py` |

**Proposed Solution:** Enforce single import point
```python
# ALL hyperbolic ops from src/geometry (geoopt-backed)
from src.geometry import (
    poincare_distance,
    exp_map_zero,
    log_map_zero,
    mobius_add,
    project_to_ball,
)

# Delete duplicate implementations in:
# - src/core/padic_math.py (lines 234-267)
# - src/losses/padic/metric_loss.py (lines 45-78)
# - src/models/projection.py (inline implementations)
```

**Impact:** ~200 LOC reduction, eliminates geometry inconsistencies

---

### U4: Loss Computation Deduplication (MEDIUM IMPACT)

**Problem:** Radial/hierarchy loss computed identically in 4+ files

**Duplicated Files:**
```
src/losses/rich_hierarchy.py:67-98        # Primary implementation
src/losses/padic/geodesic.py:45-89        # Near-identical copy
src/losses/objectives/hierarchy_loss.py   # Another copy
scripts/epsilon_vae/train_*.py            # Inline computations
```

**Shared Pattern (Repeated):**
```python
# This exact pattern appears 4+ times:
radii = poincare_distance(z_hyp, origin, c=curvature)
target_radii = 0.85 - (valuations / 9) * 0.75
radial_error = (radii - target_radii).abs().mean()
```

**Proposed Solution:** Centralize in `src/losses/radial.py`
```python
def compute_radial_targets(valuations: Tensor, max_v: int = 9) -> Tensor:
    """Compute target radii from valuations. v=0→0.85, v=9→0.10"""
    return 0.85 - (valuations / max_v) * 0.75

def radial_loss(z_hyp: Tensor, valuations: Tensor, c: float = 1.0) -> Tensor:
    """Canonical radial hierarchy loss."""
    origin = torch.zeros_like(z_hyp)
    radii = poincare_distance(z_hyp, origin, c=c)
    targets = compute_radial_targets(valuations)
    return (radii - targets).abs().mean()
```

**Impact:** ~150 LOC reduction, consistent radial computation

---

### U5: Protocol/Interface Consolidation (LOW IMPACT)

**Problem:** Type definitions scattered across multiple files

**Fragmented Locations:**
```
src/types.py                    # VAEOutput, TrainingConfig
src/interfaces.py               # EncoderProtocol, DecoderProtocol
src/models/types.py             # ModelConfig, CheckpointData
src/training/types.py           # TrainingState, BatchData
src/core/padic_math.py:15-45    # TernaryOperation, Valuation
```

**Proposed Solution:** Consolidate to `src/types/`
```
src/types/
├── __init__.py          # Re-exports all
├── core.py              # TernaryOperation, Valuation, Codon
├── models.py            # VAEOutput, ModelConfig, CheckpointData
├── training.py          # TrainingState, BatchData, Metrics
└── protocols.py         # EncoderProtocol, DecoderProtocol
```

**Impact:** ~50 LOC reduction, cleaner imports, better discoverability

---

### U6: Encoder Variant Registry (MEDIUM IMPACT)

**Problem:** 23 encoder files with overlapping patterns

**Analysis Summary:**
| Category | Files | Pattern |
|----------|-------|---------|
| Core Ternary | 3 | `27→hidden→16` (frozen, improved, attention) |
| Codon | 4 | `64→hidden→16` (basic, esm2, structure, padic) |
| Disease | 8 | `128→hidden→32` (hiv, tb, cancer, etc.) |
| Cross-Resistance | 3 | `256→hidden→64` (nrti, nnrti, pi) |
| Specialized | 5 | Various (holographic, ptm, crispr, etc.) |

**Proposed Solution:** Encoder registry with factory
```python
# src/encoders/registry.py
ENCODER_REGISTRY = {
    "ternary_frozen": (FrozenEncoder, {"input": 27, "latent": 16}),
    "ternary_improved": (ImprovedEncoder, {"input": 27, "latent": 16}),
    "codon_basic": (CodonEncoder, {"input": 64, "latent": 16}),
    "disease_hiv": (HIVEncoder, {"input": 128, "latent": 32}),
    ...
}

def create_encoder(name: str, **overrides) -> nn.Module:
    cls, defaults = ENCODER_REGISTRY[name]
    return cls(**{**defaults, **overrides})
```

**Impact:** Standardized encoder creation, easier experimentation

---

### U7: Research Scripts Audit (DEFERRED)

**Problem:** ~40 research scripts use Euclidean norm on hyperbolic embeddings

**Already Documented:** See `.claude/CLAUDE.md` section "V5.12.2 COMPLETE FIX LIST"

**Status:** Deferred to separate PR after v5.12.5 core updates

**Tracking:**
- Full list: `docs/audits/v5.12.2-hyperbolic/`
- Priority 6-10 in audit document
- ~12 usages each in HIV, RA, genetic_code, spectral_analysis scripts

---

### Unification Summary

| ID | Opportunity | LOC Saved | Priority | Effort |
|----|-------------|-----------|----------|--------|
| U1 | MLP Builder | ~400 | HIGH | Low |
| U2 | CrossResistanceVAE Base | ~700 | HIGH | Medium |
| U3 | Geometry Consolidation | ~200 | MEDIUM | Low |
| U4 | Loss Deduplication | ~150 | MEDIUM | Low |
| U5 | Protocol Consolidation | ~50 | LOW | Low |
| U6 | Encoder Registry | N/A | MEDIUM | Medium |
| U7 | Research Scripts | N/A | DEFERRED | High |
| **Total** | | **~1,500** | | |

---

## Proposed Updates

### ✅ Phase 1: Controller Metric Fix (COMPLETE - 2026-01-08)

**Goal:** Pass actual hierarchy metrics to DifferentiableController ✅ ACHIEVED

**Status:** COMPLETE - Superior implementation to original plan
**Completion Date:** 2026-01-08
**Implementation:** Ralph Loop Iteration 1

#### What Was Implemented

**Files Modified:**
- ✅ `src/models/ternary_vae.py:83-90` - Added required imports
- ✅ `src/models/ternary_vae.py:375-408` - Replaced placeholders with real hierarchy calculation

**Actual Implementation (Better than planned):**
Instead of the planned "radial variance proxy", implemented **full Spearman correlation**:

```python
# IMPLEMENTED SOLUTION (Superior to planned proxy)
batch_indices = TERNARY.from_ternary(x)  # Convert ternary ops to indices
batch_valuations = TERNARY.valuation(batch_indices).cpu().numpy()  # Get valuations

# Calculate actual radii using hyperbolic geometry
radii_A_batch = poincare_distance(z_A_hyp, origin, c=curvature).detach().cpu().numpy()
radii_B_batch = poincare_distance(z_B_hyp, origin, c=curvature).detach().cpu().numpy()

# Real Spearman correlation (not proxy)
if len(batch_valuations) > 1 and len(np.unique(batch_valuations)) > 1:
    hierarchy_A = spearmanr(batch_valuations, radii_A_batch)[0]
    hierarchy_B = spearmanr(batch_valuations, radii_B_batch)[0]
    # Handle NaN, edge cases...
else:
    hierarchy_A = 0.0  # Fallback for uniform data
    hierarchy_B = 0.0

# Real values replace placeholders
torch.tensor(hierarchy_A, device=x.device, dtype=torch.float32),  # Real H_A
torch.tensor(hierarchy_B, device=x.device, dtype=torch.float32),  # Real H_B
```

#### Key Improvements Over Plan

1. **Better Calculation:** Used actual Spearman correlation instead of variance proxy
2. **No API Changes:** Used `TERNARY.from_ternary(x)` - no need for new parameters
3. **Robust Edge Handling:** NaN values, small batches, uniform valuations all handled
4. **Full Validation:** Comprehensive testing confirmed controller receives real values

#### Impact Achieved

- ✅ Controller receives dynamic hierarchy values (-1.0 to +1.0 range)
- ✅ Values adapt to different input patterns
- ✅ Homeostatic training capabilities unlocked
- ✅ Foundation for hierarchy-aware loss weighting established

**Files Remaining:** `src/models/ternary_vae_optionc.py` may need same fix (investigation needed)

---

### Phase 2: Encoder/Decoder Architecture Improvements

**Goal:** More flexible and powerful encoding

#### 2.1 Attention-Based Encoder (NEW)

**Location:** `src/models/attention_encoder.py`

```python
class AttentionEncoder(nn.Module):
    """Operation-aware encoder with self-attention."""
    # Input: 9 ternary digits
    # Self-attention over digit positions
    # Output: 16-dim latent (mu, logvar)
```

**Benefits:**
- Learns position-specific importance
- Captures digit interactions (not just MLP)
- Interpretable attention weights

#### 2.2 Residual Decoder Enhancement

**Location:** `src/models/improved_components.py`

```python
class ResidualDecoder(nn.Module):
    """Decoder with skip connections for gradient flow."""
    # Residual blocks: 16→32→64→27
    # Skip connections preserve low-level features
```

**Benefits:**
- Better gradient flow through deeper networks
- Preserves fine-grained reconstruction detail

#### 2.3 Encoder Factory Pattern

**Location:** `src/factories/encoder_factory.py`

```python
def create_encoder(encoder_type: str, config: dict) -> nn.Module:
    """Factory for encoder creation."""
    # "frozen" → FrozenEncoder (v5.5)
    # "improved" → ImprovedEncoder (v5.12.4)
    # "attention" → AttentionEncoder (NEW)
    # "custom" → User-provided class
```

---

### Phase 3: Homeostasis Controller Enhancements

**Goal:** More responsive and accurate adaptation

#### 3.1 Per-Epoch Annealing

**Current:** Annealing only on freeze→unfreeze transitions
**Proposed:** Epoch-level micro-annealing based on metric trends

```python
def update(self, epoch, metrics):
    # Current: Only anneal on state transition
    # New: Also micro-anneal on significant metric change
    if abs(metrics.hierarchy - self.prev_hierarchy) > 0.01:
        self._micro_anneal(direction=sign(delta_Q))
```

#### 3.2 Warmup-Hysteresis Balance

**Current:** 5-epoch warmup, 3-epoch hysteresis (may miss early dynamics)
**Proposed:** Graduated warmup with reduced initial hysteresis

```python
# Epoch 1-3: No homeostasis (pure exploration)
# Epoch 4-6: Soft homeostasis (1-epoch hysteresis)
# Epoch 7+: Full homeostasis (3-epoch hysteresis)
```

#### 3.3 Q-Metric Enhancement

**Current:** Q = dist_corr + 1.5 × |hierarchy|
**Proposed:** Add richness component

```python
Q_enhanced = dist_corr + 1.5 * |hierarchy| + 0.5 * log(richness + 1e-6)
```

---

### Phase 4: Loss Function Refinements

**Goal:** Better balance of hierarchy, richness, and coverage

#### 4.1 Adaptive Loss Weighting

**Location:** `src/losses/adaptive_rich_hierarchy.py`

```python
class AdaptiveRichHierarchyLoss:
    """Loss with epoch-dependent weight scheduling."""

    def compute_weights(self, epoch, metrics):
        # Early: High coverage weight (establish reconstruction)
        # Mid: High hierarchy weight (learn ordering)
        # Late: High richness weight (preserve diversity)
```

#### 4.2 Triplet Loss Integration

**Combine with RichHierarchyLoss:**
```python
# Level-mean hierarchy (current)
+ triplet_weight * triplet_loss  # Fine-grained ordering
```

---

### Phase 5: Training Pipeline Updates

**Goal:** Integrate all improvements into cohesive training

#### 5.1 Updated Training Script

**Location:** `src/scripts/training/train_v5_12_5.py`

**Features:**
- Encoder selection via config
- **Riemannian optimizer by default** (RiemannianAdam for projection params)
- Enhanced homeostasis
- Adaptive loss scheduling
- Comprehensive logging

**Riemannian Optimizer Integration:**
```python
from geoopt.optim import RiemannianAdam

# Separate param groups
projection_params = model.projection.parameters()
other_params = [p for n, p in model.named_parameters() if 'projection' not in n]

optimizer = RiemannianAdam([
    {'params': projection_params, 'lr': lr},           # Riemannian for hyperbolic
    {'params': other_params, 'lr': lr, 'manifold': None}  # Standard for Euclidean
])
```

#### 5.2 Configuration Schema

**Location:** `src/configs/v5_12_5.yaml`

```yaml
model:
  encoder_type: "improved"   # "frozen", "improved", "attention"
  decoder_type: "improved"   # "frozen", "improved", "residual"
  learnable_curvature: false # Start conservative

training:
  optimizer:
    type: "riemannian"       # DEFAULT: RiemannianAdam for projection
    lr: 1e-3
    weight_decay: 1e-4

  homeostasis:
    enable_micro_annealing: true
    graduated_warmup: true
    q_metric: "enhanced"

  loss:
    hierarchy_weight: 5.0
    coverage_weight: 1.0
    richness_weight: 2.0
    separation_weight: 3.0
```

---

## Implementation Priority (Updated with Unification)

### Phase 0: Quick Wins (Framework Unification)

| Task | ID | LOC Saved | Effort | Dependencies |
|------|----|-----------|--------|--------------|
| MLP Builder utility | U1 | ~400 | 2h | None |
| Geometry consolidation | U3 | ~200 | 2h | None |
| Loss deduplication | U4 | ~150 | 1h | U3 (geometry) |
| **Subtotal** | | **~750** | **5h** | |

### Phase 1: Controller Metric Fix (Critical)

| Task | LOC | Effort | Dependencies |
|------|-----|--------|--------------|
| Fix placeholder metrics | ~50 | 2h | None |
| Add hierarchy proxy | ~30 | 1h | U3 (geometry) |
| Update ternary_vae_optionc.py | ~20 | 0.5h | Above |
| **Subtotal** | **~100** | **3.5h** | |

### Phase 2: Medium Consolidation

| Task | ID | LOC Saved | Effort | Dependencies |
|------|----|-----------|--------|--------------|
| CrossResistanceVAE base | U2 | ~700 | 4h | U1 (MLP) |
| Encoder registry | U6 | N/A | 3h | None |
| **Subtotal** | | **~700** | **7h** | |

### Phase 3: Homeostasis Enhancements

| Task | LOC | Effort | Dependencies |
|------|-----|--------|--------------|
| Per-epoch micro-annealing | ~40 | 2h | Phase 1 |
| Q-metric enhancement | ~20 | 1h | U4 (radial loss) |
| Graduated warmup | ~30 | 1h | None |
| **Subtotal** | **~90** | **4h** | |

### Phase 4: Training Pipeline Integration

| Task | LOC | Effort | Dependencies |
|------|-----|--------|--------------|
| train_v5_12_5.py | ~300 | 3h | All above |
| configs/v5_12_5.yaml | ~50 | 0.5h | None |
| Riemannian optimizer | ~20 | 0.5h | None |
| **Subtotal** | **~370** | **4h** | |

### Implementation Order (Recommended)

```
Week 1 (Quick Wins):
  U1 (MLP) → U3 (Geometry) → U4 (Loss) → Phase 1 (Controller)

Week 2 (Consolidation):
  U2 (CrossResistance) → U6 (Registry) → Phase 3 (Homeostasis)

Week 3 (Integration):
  Phase 4 (Training) → Validation → v5.12.5 Release
```

**Total Estimated Effort:** ~23.5 hours
**Total LOC Reduction:** ~1,500

### User Decisions (Confirmed)
- **Priority:** Controller fix first (quick win, high impact)
- **Research scripts:** Defer to separate PR, document in CLAUDE.md
- **Compatibility:** Full backward compatibility required (v5.12.4 → v5.12.5)

---

## Success Criteria

| Metric | Current | Target | Validation |
|--------|---------|--------|------------|
| Coverage | 100% | 100% | Must not regress |
| Hierarchy | -0.82 | -0.8321 | Reach ceiling |
| Richness | ~0.002 | 0.008+ | Match homeostatic_rich |
| DDG Spearman | 0.58 | 0.65+ | Bioinformatics validation |
| DDG Pearson | 0.79 | 0.85+ | Bioinformatics validation |

---

## Files to Create/Modify

### New Files (Unification)
| File | Purpose | Phase |
|------|---------|-------|
| `src/utils/nn_factory.py` | MLPBuilder utility | U1 |
| `src/losses/radial.py` | Canonical radial loss | U4 |
| `src/models/cross_resistance/base.py` | CrossResistanceVAE base class | U2 |
| `src/encoders/registry.py` | Encoder factory/registry | U6 |

### New Files (Features)
| File | Purpose | Phase |
|------|---------|-------|
| `src/models/attention_encoder.py` | Attention-based ternary encoder | 2 |
| `src/scripts/training/train_v5_12_5.py` | Unified training script | 4 |
| `src/configs/v5_12_5.yaml` | v5.12.5 configuration | 4 |

### Modified Files (Critical Path)
| File | Changes | Phase |
|------|---------|-------|
| `src/models/ternary_vae.py:375-376` | Replace H_A/H_B placeholders | 1 |
| `src/models/ternary_vae_optionc.py` | Same controller fix | 1 |
| `src/models/homeostasis.py` | Micro-annealing, Q enhancement | 3 |

### Modified Files (Consolidation)
| File | Changes | Phase |
|------|---------|-------|
| `src/geometry/__init__.py` | Export all ops, deprecate duplicates | U3 |
| `src/core/padic_math.py` | Remove geometry duplicates | U3 |
| `src/losses/rich_hierarchy.py` | Import from radial.py | U4 |
| `src/models/cross_resistance/nrti.py` | Extend base class | U2 |
| `src/models/cross_resistance/nnrti.py` | Extend base class | U2 |
| `src/models/cross_resistance/pi.py` | Extend base class | U2 |

### Files to Delete (After Consolidation)
| File | Replaced By | Phase |
|------|-------------|-------|
| Inline MLP patterns | `src/utils/nn_factory.py` | U1 |
| Duplicate geometry in losses | `src/geometry/` | U3 |
| Duplicate radial loss | `src/losses/radial.py` | U4 |

---

## Resolved Questions

1. ✅ **Encoder Priority:** Controller fix first, then encoder improvements
2. ✅ **Backward Compatibility:** Full compatibility required (v5.12.4 loadable in v5.12.5)
3. ✅ **Research Scripts:** Defer to separate PR, document in CLAUDE.md

## All Questions Resolved

4. ✅ **Riemannian Optimization:** Enable by default (RiemannianAdam for projection params)

## Documentation Update Required

Add to `.claude/CLAUDE.md`:
```markdown
## Known Issues (Deferred)

**~40 Research Scripts Using Euclidean Norm**
- Location: `src/research/` and related bioinformatics scripts
- Issue: Use `torch.norm()` instead of `poincare_distance()` on hyperbolic embeddings
- Impact: Incorrect metrics in research analysis (not affecting core training)
- Status: Deferred to separate PR after v5.12.5
- Tracking: See `docs/audits/v5.12.2-hyperbolic/` for full list
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Coverage regression | Low | High | Freeze VAE-A, monitor each epoch |
| Training instability | Medium | Medium | Graduated warmup, checkpoint frequently |
| Increased training time | Medium | Low | Attention encoder adds ~20% overhead |
| Breaking changes | Low | Medium | Version config schema, migration guide |
| **Refactor regressions** | Medium | Medium | Run full test suite after each U-phase |
| **Import chain breaks** | Low | Low | Deprecation warnings before removal |

---

## Migration & Compatibility

### Backward Compatibility Guarantees

1. **Checkpoint Loading**: v5.12.4 checkpoints loadable in v5.12.5
2. **Config Schema**: Old configs work with deprecation warnings
3. **API Stability**: Public functions maintain signatures

### Deprecation Strategy

```python
# Phase U3: Geometry consolidation
# In src/core/padic_math.py:
import warnings

def poincare_distance(...):  # OLD location
    warnings.warn(
        "poincare_distance moved to src.geometry. "
        "Import from src.geometry instead. "
        "This will be removed in v5.13.",
        DeprecationWarning, stacklevel=2
    )
    from src.geometry import poincare_distance as _pd
    return _pd(...)
```

### Test Coverage Requirements

| Phase | Test Requirement |
|-------|------------------|
| U1 (MLP) | Unit tests for MLPBuilder |
| U2 (CrossResistance) | Preserve existing test coverage |
| U3 (Geometry) | All geometry tests pass |
| U4 (Radial) | Loss computation tests |
| Phase 1 | Controller output validation |
| Phase 3 | Homeostasis state transition tests |
| Phase 4 | Full integration test suite |

---

## Appendix: Detailed File Locations

### U1: MLP Patterns (40+ instances to consolidate)

```
src/models/ternary_vae.py:45-52
src/models/improved_components.py:23-35
src/models/cross_resistance/nrti.py:89-98
src/models/cross_resistance/nnrti.py:91-100
src/models/cross_resistance/pi.py:87-96
src/losses/consequence_predictor.py:156-168
src/models/homeostasis.py:78-89
src/encoders/codon_encoder.py:67-78
src/encoders/holographic_encoder.py:45-56
src/encoders/ptm_encoder.py:34-45
... (30+ more in src/encoders/, src/diseases/)
```

### U2: CrossResistance Shared Code

```
# These blocks are 85% identical across 3 files:
nrti.py:15-89    ↔ nnrti.py:15-91  ↔ pi.py:15-87    # __init__
nrti.py:91-145   ↔ nnrti.py:93-147 ↔ pi.py:89-143   # encode()
nrti.py:147-198  ↔ nnrti.py:149-200↔ pi.py:145-196  # decode()
nrti.py:200-215  ↔ nnrti.py:202-217↔ pi.py:198-213  # reparameterize()
nrti.py:217-267  ↔ nnrti.py:219-269↔ pi.py:215-265  # forward()
nrti.py:269-310  ↔ nnrti.py:271-312↔ pi.py:267-308  # cross_resistance_matrix()
```

### U3: Geometry Duplicates

```
# Canonical (keep):
src/geometry/poincare.py:45-89           # poincare_distance
src/geometry/poincare.py:91-134          # exp_map_zero
src/geometry/poincare.py:136-178         # log_map_zero

# Duplicates (remove):
src/core/padic_math.py:234-267           # poincare_distance copy
src/losses/padic/metric_loss.py:45-78    # inline implementation
src/models/projection.py:89-123          # exp_map copy
```

### U4: Radial Loss Duplicates

```
# Canonical (keep):
src/losses/rich_hierarchy.py:67-98

# Duplicates (consolidate to radial.py):
src/losses/padic/geodesic.py:45-89
src/losses/objectives/hierarchy_loss.py:34-67
scripts/epsilon_vae/train_homeostatic_rich.py:145-178
scripts/epsilon_vae/train_hierarchy_focused.py:134-167
```

---

---

## Part II: Comprehensive Reference Documentation

This section provides detailed context for future development sessions.

---

## Architecture Deep Dive

### The Dual-Encoder Philosophy

The Ternary VAE uses two complementary encoders (VAE-A and VAE-B) to solve a fundamental tension:

**The Coverage-Hierarchy Tradeoff:**
- **Coverage** requires reconstructing all 19,683 ternary operations correctly
- **Hierarchy** requires embedding operations at radii proportional to their 3-adic valuation
- These goals initially conflict: maximizing coverage often leads to frequency-based encoding (common operations clustered), not valuation-based

**Solution: Dual Encoders**
```
Input (27-dim one-hot)
    ↓
┌───────────────┬───────────────┐
│    VAE-A      │    VAE-B      │
│  (Coverage)   │  (Hierarchy)  │
│   May freeze  │   Trainable   │
└───────────────┴───────────────┘
    ↓                   ↓
    z_A (16-dim)        z_B (16-dim)
    ↓                   ↓
┌───────────────┬───────────────┐
│ Hyperbolic    │ Hyperbolic    │
│ Projection A  │ Projection B  │
└───────────────┴───────────────┘
    ↓                   ↓
    z_A_hyp             z_B_hyp
    (Poincaré ball)     (Poincaré ball)
    ↓                   ↓
┌─────────────────────────────────┐
│     DifferentiableController    │
│   Learns loss weights from:     │
│   - Coverage metrics            │
│   - Hierarchy metrics (BROKEN)  │
│   - Distance correlations       │
└─────────────────────────────────┘
    ↓
    Loss weights: rho, weight_geodesic, beta_A, beta_B, tau
    ↓
┌─────────────────────────────────┐
│      HomeostasisController      │
│   - Freeze/unfreeze decisions   │
│   - Q-metric gating             │
│   - Annealing schedules         │
└─────────────────────────────────┘
```

### Component Details

#### VAE-A: Coverage Encoder

**Purpose:** Ensure all 19,683 operations can be reconstructed

**Architecture (v5.12.4 ImprovedEncoder):**
```python
class ImprovedEncoder(nn.Module):
    def __init__(self, input_dim=27, latent_dim=16, hidden_dim=64, dropout=0.1):
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # 27 → 64
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), # 64 → 32
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)      # 32 → 16
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)  # 32 → 16

    def forward(self, x):
        h = self.layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10, 2)  # Prevent KL collapse/explosion
        return mu, logvar
```

**Key Design Decisions:**
- SiLU activation (smoother than ReLU)
- LayerNorm (more stable than BatchNorm for VAEs)
- Logvar clamping [-10, 2] (prevents numerical issues)
- Dropout 0.1 (regularization without hurting reconstruction)

#### VAE-B: Hierarchy Encoder

**Purpose:** Learn 3-adic valuation → radial position mapping

**Expected Behavior:**
```
Valuation 0 (13,122 ops) → Radius ~0.85 (outer)
Valuation 1 (4,374 ops)  → Radius ~0.77
Valuation 2 (1,458 ops)  → Radius ~0.68
...
Valuation 9 (2 ops)      → Radius ~0.10 (center)
```

**Training Strategy:**
- Start with VAE-A frozen (preserve coverage)
- Train VAE-B with hierarchy-focused loss
- Use lower learning rate (encoder_b_lr_scale = 0.1)
- Monitor Spearman correlation (target: -0.83 to -1.0)

#### Hyperbolic Projection

**Purpose:** Map Euclidean latent space to Poincaré ball

```python
class HyperbolicProjection(nn.Module):
    def __init__(self, latent_dim=16, curvature=1.0, max_radius=0.99):
        self.curvature = curvature
        self.max_radius = max_radius
        # Direction network: learns angular position
        self.direction_net = nn.Sequential(...)
        # Radius network: learns radial position
        self.radius_net = nn.Sequential(...)

    def forward(self, z_euclidean):
        # Get direction (normalized to unit sphere)
        direction = self.direction_net(z_euclidean)
        direction = F.normalize(direction, dim=-1)

        # Get radius (clamped to ball)
        radius = self.radius_net(z_euclidean)
        radius = torch.sigmoid(radius) * self.max_radius

        # Combine to get Poincaré point
        z_hyp = radius.unsqueeze(-1) * direction
        return z_hyp
```

**V5.12.2 FIX:** Radius computation must use hyperbolic distance, not Euclidean norm:
```python
# WRONG: radius = torch.norm(z_hyp, dim=-1)
# CORRECT:
origin = torch.zeros_like(z_hyp)
radius = poincare_distance(z_hyp, origin, c=self.curvature)
```

#### DifferentiableController

**Purpose:** Learn optimal loss weights from training dynamics

**Architecture:**
```python
class DifferentiableController(nn.Module):
    def __init__(self, input_dim=8):
        # Input: [coverage, dist_corr, H_A, H_B, r_v0_A, r_v0_B, r_v9_A, r_v9_B]
        self.net = nn.Sequential(
            nn.Linear(8, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 6),  # Output: 6 loss weights
            nn.Softplus(),     # Ensure positive weights
        )

    def forward(self, metrics):
        weights = self.net(metrics)
        return {
            'rho': weights[0],
            'weight_geodesic': weights[1],
            'beta_A': weights[2],
            'beta_B': weights[3],
            'tau': weights[4],
            'lambda_rich': weights[5],
        }
```

**CRITICAL BUG (Phase 1 Fix):**
```python
# Current (broken) - src/models/ternary_vae.py:375-376
metrics = torch.stack([
    coverage_metric,
    dist_corr,
    torch.tensor(1.0, device=x.device),  # H_A PLACEHOLDER!
    torch.tensor(1.0, device=x.device),  # H_B PLACEHOLDER!
    radii_A[valuations == 0].mean(),
    radii_B[valuations == 0].mean(),
    radii_A[valuations >= 8].mean(),
    radii_B[valuations >= 8].mean(),
])
```

The controller receives constant H_A=1.0 and H_B=1.0 instead of actual hierarchy metrics, preventing it from learning hierarchy-aware weighting.

#### HomeostasisController

**Purpose:** Orchestrate freeze/unfreeze and training phase transitions

**Key Mechanisms:**

1. **Q-Metric Gating:**
```python
def compute_Q(metrics):
    """Structure quality metric."""
    dist_corr = metrics.distance_correlation
    hierarchy = abs(metrics.hierarchy_B)
    return dist_corr + 1.5 * hierarchy
```

2. **Freeze/Unfreeze Logic:**
```python
def should_freeze_A(self, metrics, epoch):
    if epoch < self.warmup_epochs:
        return True  # Always frozen during warmup

    if metrics.coverage < self.coverage_threshold:
        return True  # Freeze to recover coverage

    if self._Q_plateaued(epochs=3):
        return False  # Unfreeze to explore

    return self.current_freeze_state
```

3. **Annealing Schedule:**
```python
def anneal(self, transition_type):
    if transition_type == "freeze_to_unfreeze":
        # Reduce learning rate, increase beta
        self.lr_scale = 0.5
        self.beta_warmup_epochs = 3
    elif transition_type == "unfreeze_to_freeze":
        # Reset to conservative settings
        self.lr_scale = 1.0
```

---

## Mathematical Foundations

### 3-Adic (P-Adic) Number System

**Definition:** The 3-adic valuation of an integer n is the largest power of 3 that divides n.

```python
def valuation_3(n):
    """Compute 3-adic valuation of n."""
    if n == 0:
        return float('inf')
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v
```

**For Ternary Operations:**

Each of the 19,683 operations is indexed 0 to 19,682. The valuation distribution:

| Valuation | Count | Percentage | Example Indices |
|-----------|-------|------------|-----------------|
| 0 | 13,122 | 66.7% | 1, 2, 4, 5, 7, 8, ... |
| 1 | 4,374 | 22.2% | 3, 6, 12, 15, 21, ... |
| 2 | 1,458 | 7.4% | 9, 18, 36, 45, 63, ... |
| 3 | 486 | 2.5% | 27, 54, 108, 135, ... |
| 4 | 162 | 0.8% | 81, 162, 324, 405, ... |
| 5 | 54 | 0.3% | 243, 486, 972, ... |
| 6 | 18 | 0.09% | 729, 1458, 2916, ... |
| 7 | 6 | 0.03% | 2187, 4374, 8748, ... |
| 8 | 2 | 0.01% | 6561, 13122 |
| 9 | 2 | 0.01% | 0, 19683 (boundary) |

**Mathematical Ceiling for Hierarchy:**

The maximum achievable Spearman correlation is **-0.8321** due to the heavy imbalance (v=0 has 66.7% of samples). This is a mathematical limit, not a training failure.

Proof sketch:
- Spearman ranks ties by giving them the average rank
- With 13,122 samples at v=0, even perfect radial ordering within other levels can't overcome the tie-averaging effect
- Only "RadialSnapProjection" (snapping to exact radii) achieves -1.0, but this eliminates all within-level variance (richness = 0)

### Hyperbolic Geometry (Poincaré Ball)

**Why Hyperbolic?**
- 3-adic numbers have a natural tree structure (ultrametric)
- Hyperbolic space can embed trees with low distortion
- The Poincaré ball provides bounded, differentiable embedding

**Key Operations:**

```python
def poincare_distance(x, y, c=1.0):
    """Distance in Poincaré ball."""
    sqrt_c = torch.sqrt(torch.tensor(c))
    norm_x = torch.norm(x, dim=-1, keepdim=True).clamp(max=1-1e-5)
    norm_y = torch.norm(y, dim=-1, keepdim=True).clamp(max=1-1e-5)

    diff = x - y
    num = 2 * c * torch.sum(diff ** 2, dim=-1)
    denom = (1 - c * norm_x ** 2) * (1 - c * norm_y ** 2)

    return torch.acosh(1 + num / denom.squeeze() + 1e-7) / sqrt_c

def exp_map_zero(v, c=1.0):
    """Exponential map from origin (Euclidean → Poincaré)."""
    sqrt_c = torch.sqrt(torch.tensor(c))
    norm_v = torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-7)
    return torch.tanh(sqrt_c * norm_v) * v / (sqrt_c * norm_v)

def log_map_zero(y, c=1.0):
    """Logarithmic map to origin (Poincaré → Euclidean)."""
    sqrt_c = torch.sqrt(torch.tensor(c))
    norm_y = torch.norm(y, dim=-1, keepdim=True).clamp(min=1e-7, max=1-1e-5)
    return torch.atanh(sqrt_c * norm_y) * y / (sqrt_c * norm_y)
```

**Curvature:**
- c = 1.0 is standard (unit Poincaré ball)
- Higher c = tighter ball, stronger curvature
- Learnable curvature is experimental (disabled by default)

---

## Checkpoint Reference (Complete)

### Production-Ready Checkpoints

| Checkpoint | Coverage | Hier_A | Hier_B | Richness | r_v0 | r_v9 | Use Case |
|------------|----------|--------|--------|----------|------|------|----------|
| **homeostatic_rich** | 100% | -0.69 | -0.8321 | 0.00787 | 0.89 | 0.19 | **RECOMMENDED** |
| **v5_11_homeostasis** | 99.9% | -0.51 | -0.83 | 0.00126 | 0.56 | 0.48 | Stable, conservative |
| **v5_11_structural** | 100% | -0.69 | -0.74 | 0.00304 | 0.55 | 0.42 | Bioinformatics codon encoder |
| **v5_12_4** | 100% | N/A | -0.82 | ~0.002 | N/A | N/A | Latest stable |

### Checkpoints with Issues

| Checkpoint | Issue | Details |
|------------|-------|---------|
| v5_11_overnight | **COLLAPSED** | Training collapsed, metrics are artifacts |
| v5_11_progressive | **INVERTED** | VAE-B shows +0.78 (wrong sign) |
| v5_11_annealing | **INVERTED** | Both VAEs show positive hierarchy |
| max_hierarchy | Low richness | Achieved -0.8321 but richness = 0.00028 |
| radial_target | **INVERTED** | VAE-B shows +0.68 |

### Checkpoint Loading

```python
import torch
from src.models import TernaryVAEV5_11_PartialFreeze

# Load checkpoint
checkpoint = torch.load('checkpoints/homeostatic_rich/best.pt')

# Create model with matching architecture
model = TernaryVAEV5_11_PartialFreeze(
    latent_dim=16,
    hidden_dim=64,
    max_radius=0.99,
    curvature=1.0,
    use_controller=True,
    use_dual_projection=True,
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

---

## Training Strategies

### Strategy 1: Homeostatic Rich (Recommended)

**Script:** `src/scripts/epsilon_vae/train_homeostatic_rich.py`

**Philosophy:** Balance hierarchy and richness through adaptive weighting

```yaml
# Key hyperparameters
loss_weights:
  hierarchy_weight: 5.0      # Strong push toward target radii
  coverage_weight: 1.0       # Maintain reconstruction
  richness_weight: 2.0       # Preserve within-level variance
  separation_weight: 3.0     # Ensure level ordering

training:
  freeze_encoder_a: true     # Protect coverage
  encoder_b_lr_scale: 0.1    # Slow adaptation for B
  epochs: 100
  batch_size: 256

homeostasis:
  enable: true
  warmup_epochs: 5
  hysteresis_epochs: 3
  coverage_threshold: 0.99
```

**Results:**
- Achieves hierarchy ceiling (-0.8321)
- 5.8x more richness than baseline
- 100% coverage maintained

### Strategy 2: Hierarchy Focused

**Script:** `src/scripts/epsilon_vae/train_hierarchy_focused.py`

**Philosophy:** Maximize hierarchy at cost of richness

```yaml
loss_weights:
  hierarchy_weight: 10.0     # Very strong hierarchy push
  coverage_weight: 1.0
  richness_weight: 0.0       # Ignore richness
  separation_weight: 5.0

training:
  variance_weight: 100       # Collapse within-level variance
```

**Results:**
- Achieves hierarchy ceiling (-0.8321)
- Very low richness (0.00028)
- Good if only ordering matters

### Strategy 3: Progressive Unfreezing

**Philosophy:** Gradually unfreeze layers to preserve learned features

```yaml
training:
  phase_1:
    epochs: 20
    freeze_encoder_a: true
    freeze_encoder_b: true
    train_projection_only: true

  phase_2:
    epochs: 30
    freeze_encoder_a: true
    freeze_encoder_b: false
    encoder_b_lr_scale: 0.1

  phase_3:
    epochs: 50
    freeze_encoder_a: false
    all_lr_scale: 0.01
```

---

## Loss Functions Reference

### RichHierarchyLoss

**Location:** `src/losses/rich_hierarchy.py`

**Components:**
1. **Level-Mean Loss:** Push level means to target radii
2. **Separation Loss:** Ensure levels don't overlap
3. **Richness Loss:** Preserve within-level variance

```python
class RichHierarchyLoss(nn.Module):
    def forward(self, z_hyp, valuations, curvature=1.0):
        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, c=curvature)

        # Target radii per valuation
        targets = 0.85 - (valuations / 9) * 0.75

        # Level-mean loss
        unique_v = valuations.unique()
        level_means = torch.stack([radii[valuations == v].mean() for v in unique_v])
        level_targets = torch.stack([targets[valuations == v][0] for v in unique_v])
        hierarchy_loss = (level_means - level_targets).abs().mean()

        # Separation loss (penalize overlapping levels)
        sorted_means = level_means.sort(descending=True)[0]
        gaps = sorted_means[:-1] - sorted_means[1:]
        separation_loss = F.relu(-gaps + 0.01).sum()  # Min gap of 0.01

        # Richness loss (preserve within-level variance)
        level_vars = torch.stack([radii[valuations == v].var() for v in unique_v])
        richness_loss = -level_vars.mean()  # Negative because we want to maximize

        return hierarchy_loss + separation_loss + self.richness_weight * richness_loss
```

### TripletLoss (P-Adic Ranking)

**Location:** `src/losses/padic/ranking_loss.py`

**Purpose:** Fine-grained ordering using triplet comparisons

```python
class PAdicTripletLoss(nn.Module):
    def forward(self, z_hyp, valuations, curvature=1.0):
        # Sample triplets: anchor, positive (closer valuation), negative (farther)
        triplets = self.sample_triplets(valuations)

        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, c=curvature)

        loss = 0
        for anchor, pos, neg in triplets:
            d_pos = abs(radii[anchor] - radii[pos])
            d_neg = abs(radii[anchor] - radii[neg])
            loss += F.relu(d_pos - d_neg + self.margin)

        return loss / len(triplets)
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/models/test_mlp_builder.py
def test_mlp_builder_dimensions():
    mlp = MLPBuilder.build([27, 64, 32, 16])
    x = torch.randn(8, 27)
    y = mlp(x)
    assert y.shape == (8, 16)

def test_mlp_builder_activation():
    mlp = MLPBuilder.build([10, 20], activation="silu")
    # Check SiLU is in the sequential
    assert any(isinstance(m, nn.SiLU) for m in mlp.modules())

# tests/unit/geometry/test_poincare.py
def test_poincare_distance_symmetry():
    x = torch.randn(8, 16) * 0.5  # Within ball
    y = torch.randn(8, 16) * 0.5
    d_xy = poincare_distance(x, y)
    d_yx = poincare_distance(y, x)
    assert torch.allclose(d_xy, d_yx, atol=1e-6)

def test_poincare_distance_origin():
    x = torch.randn(8, 16) * 0.5
    origin = torch.zeros_like(x)
    d = poincare_distance(x, origin)
    # Should match norm formula for distance from origin
    expected = 2 * torch.atanh(torch.norm(x, dim=-1))
    assert torch.allclose(d, expected, atol=1e-5)
```

### Integration Tests

```python
# tests/integration/test_training_loop.py
def test_training_does_not_regress_coverage():
    model = TernaryVAEV5_11_PartialFreeze(...)
    model.load_state_dict(checkpoint['model_state_dict'])

    initial_coverage = evaluate_coverage(model)
    assert initial_coverage >= 0.99

    # Train for 5 epochs
    train(model, epochs=5)

    final_coverage = evaluate_coverage(model)
    assert final_coverage >= initial_coverage - 0.01  # Max 1% regression

def test_hierarchy_improves_with_training():
    model = TernaryVAEV5_11_PartialFreeze(...)

    initial_hier = evaluate_hierarchy(model)
    train(model, epochs=10, hierarchy_weight=5.0)
    final_hier = evaluate_hierarchy(model)

    assert final_hier < initial_hier  # Hierarchy should become more negative
```

### Validation Procedures

```python
# scripts/validate_checkpoint.py
def validate_checkpoint(checkpoint_path):
    """Full validation of a trained checkpoint."""
    model = load_checkpoint(checkpoint_path)

    # 1. Coverage check
    coverage = evaluate_coverage(model)
    assert coverage >= 0.99, f"Coverage {coverage} below threshold"

    # 2. Hierarchy check
    hier_A, hier_B = evaluate_hierarchy(model)
    assert hier_B < -0.5, f"Hierarchy_B {hier_B} not negative enough"

    # 3. Richness check
    richness = evaluate_richness(model)
    assert richness > 0.0001, f"Richness {richness} too low (collapsed)"

    # 4. Radial separation check
    r_v0, r_v9 = evaluate_radial_endpoints(model)
    assert r_v0 > 0.7, f"r_v0 {r_v0} too low"
    assert r_v9 < 0.3, f"r_v9 {r_v9} too high"

    # 5. Numerical stability check
    for name, param in model.named_parameters():
        assert not torch.isnan(param).any(), f"NaN in {name}"
        assert not torch.isinf(param).any(), f"Inf in {name}"

    print(f"✓ Checkpoint {checkpoint_path} validated")
```

---

## Future Improvements (Beyond v5.12.5)

### v5.13 Candidates

| Feature | Priority | Effort | Impact |
|---------|----------|--------|--------|
| Learnable curvature | Medium | Medium | Better manifold fit |
| Multi-head attention encoder | Medium | High | Position-specific learning |
| Contrastive pre-training | Low | High | Better initialization |
| Research scripts audit (U7) | High | High | Correct all 40 files |

### v6.0 Vision

| Feature | Description |
|---------|-------------|
| **Unified Disease VAE** | Single model for all 11 diseases |
| **Structure-Aware Encoding** | AlphaFold2/ESMFold integration |
| **Hierarchical Latent Space** | Multi-scale 3-adic structure |
| **Online Learning** | Continual adaptation to new sequences |

### Research Directions

1. **Theoretical:** Prove tighter bounds on hierarchy ceiling
2. **Empirical:** Benchmark against traditional ML on drug resistance
3. **Applied:** Clinical decision support tool validation
4. **Scalability:** Distributed training for large sequence databases

---

## Known Issues & Technical Debt

### Critical (Must Fix)

| Issue | Location | Impact | Fix |
|-------|----------|--------|-----|
| Controller placeholder | `ternary_vae.py:375-376` | High | Phase 1 |
| Research scripts Euclidean | 40 files | Medium | U7 (deferred) |

### Medium (Should Fix)

| Issue | Location | Impact | Fix |
|-------|----------|--------|-----|
| Geometry duplication | 3 locations | Medium | U3 |
| MLP pattern repetition | 40+ files | Low | U1 |
| Loss code duplication | 4 files | Low | U4 |

### Low (Nice to Have)

| Issue | Location | Impact | Fix |
|-------|----------|--------|-----|
| Type fragmentation | 5 files | Low | U5 |
| Encoder sprawl | 23 files | Low | U6 |
| Deprecated geometry_utils | `src/core/` | Low | Delete in v5.13 |

### Won't Fix (By Design)

| Issue | Reason |
|-------|--------|
| Hierarchy ceiling -0.8321 | Mathematical limit, not fixable |
| VAE-A inverted hierarchy | Expected behavior (frequency-based) |
| Slow training | Hyperbolic ops are inherently slower |

---

## Troubleshooting Guide

### Training Issues

**Problem:** Coverage drops below 99%
```
Symptoms: recon_loss increasing, coverage metric declining
Diagnosis: VAE-A weights changed too much
Solution:
  1. Verify freeze_encoder_a = True
  2. Reduce overall learning rate
  3. Increase coverage_weight in loss
  4. Consider loading frozen checkpoint
```

**Problem:** Hierarchy stuck positive (inverted)
```
Symptoms: Spearman correlation > 0 for VAE-B
Diagnosis: Model learning frequency-based instead of valuation-based
Solution:
  1. Increase hierarchy_weight to 10.0
  2. Add triplet loss for fine-grained ordering
  3. Check target radii formula is correct
  4. Try progressive unfreezing strategy
```

**Problem:** Richness collapsed to near-zero
```
Symptoms: richness < 0.0001, all points at exact radii
Diagnosis: Over-optimized for hierarchy, killed variance
Solution:
  1. Increase richness_weight
  2. Decrease variance_weight
  3. Use homeostatic_rich training strategy
  4. Check min_richness_ratio setting
```

**Problem:** NaN in loss or gradients
```
Symptoms: loss = nan, gradients contain NaN
Diagnosis: Numerical instability in hyperbolic ops
Solution:
  1. Clamp norms before exp_map (max_radius=0.99)
  2. Add epsilon to denominators in poincare_distance
  3. Check logvar clamping in encoder
  4. Reduce learning rate
```

### Evaluation Issues

**Problem:** Different hierarchy values in training vs evaluation
```
Symptoms: Training shows -0.8, evaluation shows -0.6
Diagnosis: Batch norm or dropout behavior differs
Solution:
  1. Ensure model.eval() is called
  2. Use torch.no_grad() for evaluation
  3. Verify same data is used
```

**Problem:** Checkpoint won't load
```
Symptoms: KeyError or shape mismatch during load
Diagnosis: Architecture mismatch between saved and current model
Solution:
  1. Check hidden_dim, latent_dim match
  2. Verify use_controller, use_dual_projection flags
  3. Use strict=False to diagnose missing keys
  4. Create model with exact same config as training
```

---

## Bioinformatics Applications

### HIV Drug Resistance Prediction

**Current Results (v5.12.4):**
- Spearman correlation: 0.58
- Pearson correlation: 0.79
- Target: 0.65+ / 0.85+

**Pipeline:**
```python
from src.diseases import HIVAnalyzer
from src.encoders import CodonEncoder

# Encode HIV protease sequence
encoder = CodonEncoder.from_checkpoint('homeostatic_rich')
embedding = encoder.encode(hiv_sequence)

# Predict resistance
analyzer = HIVAnalyzer()
resistance = analyzer.predict(embedding, drugs=['EFV', 'NVP'])
```

### Protein Stability (DDG) Prediction

**Benchmark:** S669 dataset

**Current Results:**
| Method | Spearman | MAE |
|--------|----------|-----|
| Heuristic (v1) | 0.53 | 1.91 |
| P-adic Embeddings (v2) | 0.58 | 0.73 |
| Target | 0.70+ | 0.50 |

**Key Finding:** P-adic embeddings encode force constants (rho = 0.86) which relate to stability.

### Vaccine Target Selection

**Current Pipeline:**
1. Embed pathogen sequences using p-adic VAE
2. Identify conserved regions (low radial variance across strains)
3. Rank by evolutionary stability x immunogenicity score
4. Filter by MHC binding predictions

**Results:** 387 vaccine targets ranked for HIV

---

## Partner Packages Context

### Structure

```
deliverables/partners/
├── jose_colbes/          # Protein stability (DDG)
│   ├── scripts/          # C1-C4 scripts
│   ├── results/          # validation/, rosetta_blind/
│   └── docs/             # BENCHMARK_COMPARISON.md
├── carlos_brizuela/      # AMP design
│   ├── scripts/          # B1, B8, B10
│   └── results/          # pareto, microbiome, pathogen
├── alejandra_rojas/      # Arbovirus primers
│   ├── scripts/          # A2, trajectory, scanner
│   └── results/          # pan_arbovirus_primers/
└── hiv_research_package/ # Drug resistance
    ├── scripts/          # H6, H7
    └── results/          # tdr_screening/, la_selection/
```

### Integration Status

| Partner | src.core.padic_math | VAE Interface | Status |
|---------|---------------------|---------------|--------|
| jose_colbes | Integrated | Yes | Production |
| carlos_brizuela | Partial | Yes | Development |
| alejandra_rojas | Integrated | Planned | Development |
| hiv_research | Stanford HIVdb | Yes | Production |

---

## Session Recovery Guide

### If This Session Is Interrupted

1. **Read this plan file** to understand full context
2. **Check `.claude/CLAUDE.md`** for project-level context
3. **Review git status** to see what was in progress
4. **Continue from Implementation Order** section

### Key Files to Review

```
# Understanding current state
.claude/CLAUDE.md                    # Project context
checkpoints/        # Available checkpoints
src/models/ternary_vae.py           # Core model
src/models/homeostasis.py           # Controller logic

# Understanding this plan
docs/plans/V5_12_5_IMPLEMENTATION_PLAN.md  # This file

# Key implementation targets
src/models/ternary_vae.py:375-376   # Controller bug
src/utils/nn_factory.py             # New (U1)
src/losses/radial.py                # New (U4)
```

### Quick Start Commands

```bash
# Validate environment
python -m pytest tests/unit/core/

# Check current checkpoint metrics
python src/scripts/epsilon_vae/analyze_all_checkpoints.py

# Run training (when ready)
python src/scripts/training/train_v5_12_5.py --config src/configs/v5_12_5.yaml
```

---

## Appendix: Full File Lists

### All Files in src/models/ (Relevant to v5.12.5)

```
src/models/
├── __init__.py
├── ternary_vae.py              # Main VAE (CRITICAL - Phase 1 fix)
├── ternary_vae_optionc.py      # PartialFreeze variant (Phase 1 fix)
├── homeostasis.py              # HomeostasisController (Phase 3)
├── improved_components.py       # ImprovedEncoder/Decoder
├── projection.py               # HyperbolicProjection
├── attention_encoder.py        # NEW (Phase 2)
└── cross_resistance/
    ├── __init__.py
    ├── base.py                 # NEW (U2)
    ├── nrti.py                 # Extend base (U2)
    ├── nnrti.py                # Extend base (U2)
    └── pi.py                   # Extend base (U2)
```

### All Files in src/losses/ (Relevant to v5.12.5)

```
src/losses/
├── __init__.py
├── radial.py                   # NEW (U4)
├── rich_hierarchy.py           # Use radial.py (U4)
├── consequence_predictor.py    # MLP patterns (U1)
└── padic/
    ├── geodesic.py             # Remove duplicate (U4)
    ├── metric_loss.py          # Remove geometry (U3)
    └── ranking_loss.py         # Keep
```

### All Files in src/geometry/ (Canonical Location)

```
src/geometry/
├── __init__.py                 # Export all (U3)
├── poincare.py                 # Canonical implementations
├── lorentz.py                  # Lorentz model (alternate)
└── holographic_poincare.py     # Holographic variant
```

---

## Version History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-03 | Claude | Initial plan |
| 1.1 | 2026-01-03 | Claude | Added user decisions, Riemannian default |
| 1.2 | 2026-01-03 | Claude | Added U1-U7 unification opportunities |
| 1.3 | 2026-01-03 | Claude | Expanded to comprehensive reference (~1500 lines) |
| **1.4** | **2026-01-08** | **Claude** | **Phase 1 COMPLETED - Controller hierarchy fix implemented** |

---

*Plan Version 1.4 (PHASE 1 COMPLETE)*
*Document Type: Implementation Plan + Reference Documentation*
*Total Sections: 25+ | Estimated Lines: ~1500*
*Last Updated: 2026-01-08*
*Purpose: Future-proof session recovery and implementation guide*
