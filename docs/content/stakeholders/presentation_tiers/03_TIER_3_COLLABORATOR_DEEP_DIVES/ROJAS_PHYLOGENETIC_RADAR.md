# Project Beta: Phylogenetic Radar

**Technical Deep Dive | Lead: Alejandra Rojas**

---

## 1. System Architecture

```mermaid
graph LR
    A[NCBI Virus API] -->|Weekly Fetch| B[Data Ingestion]
    B -->|Sequences| C[Hyperbolic VAE Encoder]
    C -->|Poincare Coordinates| D[Riemannian Forecaster]
    D -->|Future Coordinates| E[Risk Analysis]
```

## 2. The Core Innovation: Hyerbolic Forecasting

Viral evolution is treelike, not linear. Euclidean vectors cannot accurately predict drift.

### The Math

- **Manifold:** Poincare Ball $\mathbb{B}^n$.
- **Metric:** $ds^2 = \frac{4|dx|^2}{(1-|x|^2)^2}$
- **Operation:** Parallel Transport along Geodesics.

## 3. Implementation Details

### A. Data Pipeline (`src.data_pipeline`)

- **Source:** NCBI Virus (Entrez API).
- **Filter:** `TaxID: 12637` (Dengue), `Geo: South America`.
- **Frequency:** Weekly cron job.

### B. Geometry Engine (`src.geometry`)

- **Library:** `geomstats`.
- **Embedding:** VAE Inference (Encoder).
- **Forecasting:**
  $$ z*{t+1} = \exp*{z*t}(v_t \cdot \Delta t) $$
    Where $v_t$ is the tangent velocity vector transported from $z*{t-1}$.

## 4. Key Challenges & Solutions

- **Challenge:** Distortion at the edge of the Poincare disk.
  - _Solution:_ Use Gyrovector operations for addition/subtraction.
- **Challenge:** Sparse data in early season.
  - _Solution:_ Bayesian priors based on historical serotype cycles.

## 5. Next Specifications

To actuate this:

1.  Configure `NCBIFetcher` with API Key.
2.  Run `src/scripts/arbovirus_hyperbolic_trajectory.py` on the new data stream.
