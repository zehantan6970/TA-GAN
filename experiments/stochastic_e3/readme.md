# Reviewer 1 Enrichment E3: Stochastic Candidate Evaluation

## 1. Status and decision

**Decision:** TA-GAN demonstrably uses its latent noise and does not collapse to one deterministic output. Across five independent base seeds and 20 candidates per history, mean pairwise endpoint diversity is 0.0657 m and no window's seed-averaged endpoint diversity is below 0.01 m. Aggregate accuracy and diversity are stable across the five base seeds.

This supports the claim that TA-GAN is a **noise-conditioned stochastic candidate generator**. It does not, by itself, prove that the candidates form semantically distinct intent modes. 

## 2. Metric definitions

| Name | Definition | Deployability |
|---|---|---|
| Single sample | First stochastic candidate for each history | Yes, one stochastic draw |
| Expected error | Error averaged across all 20 candidates | Distribution-quality statistic; not one selected path |
| Ensemble mean | Error of the coordinate-wise mean of 20 candidates | Requires 20 candidates; may average distinct futures |
| minADE/minFDE@K | Lowest error among the first K candidates | Oracle only; uses ground truth |
| Endpoint diversity | Mean pairwise distance between candidate endpoints | Ground-truth independent spread metric |
| Trajectory diversity | Mean pairwise displacement across all future steps | Ground-truth independent spread metric |

ADE and FDE are in metres. Confidence intervals below resample source files so overlapping windows are not treated as independent clusters.

## 3. Main results

Results are averages across five base seeds. Intervals are 95% source-file cluster-bootstrap confidence intervals calculated from per-window seed-averaged metrics.

| Statistic | ADE (95% CI) | FDE (95% CI) | Interpretation |
|---|---:|---:|---|
| Single stochastic sample | 0.0792 [0.0764, 0.0819] | 0.1553 [0.1493, 0.1610] | Deployable single draw |
| Expected error over 20 | 0.0791 [0.0763, 0.0818] | 0.1550 [0.1491, 0.1607] | Primary distribution statistic |
| Mean of 20 candidates | 0.0725 [0.0697, 0.0753] | 0.1454 [0.1394, 0.1511] | Ensemble requiring 20 draws |
| min@20 | 0.0500 [0.0477, 0.0523] | 0.0950 [0.0899, 0.0997] | Oracle coverage only |


### Best-of-K coverage

| K | minADE@K | minFDE@K |
|---:|---:|---:|
| 1 | 0.0792 | 0.1553 |
| 2 | 0.0690 | 0.1349 |
| 5 | 0.0596 | 0.1155 |
| 10 | 0.0543 | 0.1042 |
| 20 | 0.0500 | 0.0950 |


### Diversity and seed stability

| Metric | Mean (95% CI) | Median | Additional finding |
|---|---:|---:|---|
| Pairwise endpoint diversity | 0.0657 [0.0642, 0.0673] | 0.0621 | Seed-averaged: 0% below 0.01 m; 30.77% below 0.05 m |
| Pairwise trajectory diversity | 0.0388 [0.0379, 0.0397] | 0.0371 | Averaged over all 20 future steps |

Across the five base seeds, aggregate standard deviations are 0.000044 m for expected ADE, 0.000082 m for expected FDE, 0.000094 m for endpoint diversity, and 0.000054 m for trajectory diversity. The stochastic evaluation is stable at the aggregate level.

### Descriptive motion regimes

| Regime | Windows | Single ADE | Expected ADE | minADE@20 | Endpoint diversity |
|---|---:|---:|---:|---:|---:|
| Low motion | 2,244 | 0.0767 | 0.0766 | 0.0541 | 0.0477 |
| Straight | 3,561 | 0.0686 | 0.0685 | 0.0368 | 0.0705 |
| Transition | 1,060 | 0.0808 | 0.0808 | 0.0486 | 0.0749 |
| Turning | 1,303 | 0.1111 | 0.1109 | 0.0805 | 0.0762 |

