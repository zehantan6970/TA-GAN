# Reviewer R1.3 / E2: Constant-Velocity Baseline Comparison

## 1. Status and decision

**Main finding:** Reviewer R1.3's concern is valid. On the retained single-agent windows, the strongest simple baseline, CVM-last, has significantly lower ADE than TA-GAN mean@20. TA-GAN has only a small, statistically inconclusive FDE advantage over CVM-last. TA-GAN clearly outperforms the smoother all-history CVM-LS variant and has a small average advantage on turning windows, but the current evidence does **not** support a claim that TA-GAN is universally better than constant-velocity extrapolation in low-speed indoor motion.

This result should lead to narrower paper claims, not be hidden or replaced by the more favorable CVM-LS comparison.


## 2. Frozen protocol

| Item | Setting |
|---|---|
| Dataset | `IndoorNar-Trajectory dataset` |
| Raw columns | timestamp in microseconds, x in metres, y in metres |
| Scene selection | SHA-256 deterministic 20% assignment |
| Selected scenes | scene4, scene13, scene21, scene28, scene29 |
| Parsed trajectory files | 336 |
| Empty/unusable text files | 3 |
| Evaluation windows | 8,168 |
| Observation / prediction | 20 / 20 points |
| Window stride | 10 points |
| CVM-last | velocity from final two observed points |
| CVM-LS | linear least-squares fit over all 20 observations |
| Timestamp handling | actual per-sample timestamps; no fixed-rate assumption |
| TA-GAN stochastic evaluation | 20 fixed Gaussian samples, seeds 20260811--20260830 |
| Primary TA-GAN statistic | per-window expected error, `mean@20` |
| Secondary oracle statistic | `min@20`, explicitly labeled oracle |
| Confidence intervals | 2,000-iteration source-file cluster bootstrap |
| Hardware | NVIDIA GeForce RTX 3060 |


Overlapping windows from the same source file remain together during cluster bootstrap resampling. This avoids treating all 8,168 windows as statistically independent.

## 3. Main results

All errors are in metres. Intervals are 95% source-file cluster-bootstrap confidence intervals.

| Method | ADE (95% CI) | FDE (95% CI) | Interpretation |
|---|---:|---:|---|
| **CVM-last** | **0.0745** [0.0706, 0.0790] | 0.1583 [0.1498, 0.1671] | Strongest non-oracle ADE |
| CVM-LS | 0.0956 [0.0908, 0.1002] | 0.1765 [0.1677, 0.1854] | Smoother but less responsive |
| TA-GAN mean@20 | 0.0791 [0.0763, 0.0817] | **0.1550** [0.1493, 0.1609] | Primary stochastic statistic |
| TA-GAN min@20 | 0.0499 [0.0477, 0.0522] | 0.0947 [0.0897, 0.0996] | Oracle; not a fair deterministic CVM comparison |

TA-GAN zero-noise produces ADE/FDE `1.0351/1.9088` and is not representative of sampling from the trained latent distribution. It must not be used as the primary deterministic score.

### Paired tests

The difference is defined as `TA-GAN mean@20 - CVM`; positive values favor CVM.

| Paired comparison | Mean difference (m) | 95% CI (m) | TA-GAN lower-error windows | Conclusion |
|---|---:|---:|---:|---|
| ADE vs CVM-last | +0.0046 | [+0.0012, +0.0072] | 33.84% | CVM-last significantly better |
| FDE vs CVM-last | -0.0033 | [-0.0100, +0.0016] | 40.87% | Difference crosses zero |
| ADE vs CVM-LS | -0.0165 | [-0.0190, -0.0140] | 47.04% | TA-GAN significantly better |
| FDE vs CVM-LS | -0.0215 | [-0.0253, -0.0178] | 48.22% | TA-GAN significantly better |

Relative to CVM-last, TA-GAN mean@20 increases ADE by approximately 6.1% and reduces FDE by approximately 2.1%; the FDE difference is not conclusive. Relative to CVM-LS, TA-GAN reduces ADE by approximately 17.3% and FDE by approximately 12.2%.

The window win fractions and aggregate means are not contradictory: TA-GAN can lose on more windows while gaining more on a smaller number of difficult windows.

## 4. Motion-regime analysis

Regimes are post-hoc descriptive labels, not model inputs. Heading change compares a five-step displacement at the end of observation with a five-step displacement at the end of the future. `straight` is at most 10 degrees, `turning` is at least 20 degrees, and `transition` lies between. A window is `low_motion` when total travel is below 0.10 m or either heading vector is too short (below 0.02 m) for a stable angle.

| Regime | Windows | CVM-last ADE | CVM-LS ADE | TA-GAN mean@20 ADE | TA - CVM-last | TA better than CVM-last |
|---|---:|---:|---:|---:|---:|---:|
| Low motion | 2,244 | 0.0737 | 0.0962 | 0.0765 | +0.0028 | 37.12% |
| Straight | 3,561 | **0.0588** | 0.0793 | 0.0685 | +0.0098 | 29.57% |
| Transition | 1,060 | 0.0818 | 0.0955 | **0.0808** | -0.0010 | 35.09% |
| Turning | 1,303 | 0.1128 | 0.1391 | **0.1109** | -0.0019 | 38.83% |

This decomposition directly addresses R1.3:

- CVM-last is clearly preferable on near-straight trajectories.
- TA-GAN's average advantage over CVM-last appears only in transition/turning subsets and is small in this one-agent protocol.
- TA-GAN's advantage over CVM-LS grows on turning trajectories (`-0.0282 m` ADE), showing that a long-history velocity fit reacts poorly to changes in direction.

