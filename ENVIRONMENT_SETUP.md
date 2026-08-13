# TA-GAN Windows Development Environments

The repository was cloned at commit `10217f4eee8f1c68122a043e12ac72f29ffafe7f`.

## Recommended environment for this computer

Use `ta-gan-rtx` for Reviewer 1 experiments and model development:

```powershell
conda activate ta-gan-rtx
Set-Location D:\ta-gan\paper\TA-GAN
$env:PYTHONPATH = "$PWD\ta_gan"
```

Verified versions:

| Package | Version |
|---|---|
| Python | 3.8.20 |
| PyTorch | 1.13.1+cu117 |
| TorchVision | 0.14.1+cu117 |
| TorchAudio | 0.13.1+cu117 |
| CUDA runtime | 11.7 |
| NumPy | 1.24.4 |
| SciPy | 1.5.4 |
| scikit-learn | 0.24.2 |
| Matplotlib | 3.3.4 |
| timm | 0.4.12 |
| attrdict | 2.0.1 |
| GPU | NVIDIA GeForce RTX 3060 |

The following checks passed in `ta-gan-rtx`:

- `torch.cuda.is_available()` returned `True`;
- a CUDA tensor addition returned the expected result;
- both Transformer variants, the restored Social-GAN baseline, the data loader, and ADE/FDE loss modules imported successfully;
- all 25 unit tests passed, including CUDA forward passes on the RTX 3060.

## Legacy reproduction environment

The README specifies `Python 3.6`, `PyTorch 1.10.1`, and CUDA 10.2. That environment is preserved as `ta-gan`:

```powershell
conda activate ta-gan
Set-Location D:\ta-gan\paper\TA-GAN
```

This legacy environment is useful for checking the original dependency specification, but it is not suitable for GPU experiments on this computer. The installed PyTorch build does not contain kernels for the RTX 3060 compute capability 8.6; a minimal CUDA tensor operation timed out. Use CPU only in this environment unless the hardware or PyTorch build changes.

## Recovered model sources

The clone omitted `sgan/models_transformer.py` and `sgan/models.py`, although CPython 3.6 bytecode remained. Both sources are now restored, annotated, and covered by regression tests. Device-safe tensor creation replaces historical hard-coded `.cuda()` calls without changing checkpoint parameter names or shapes.

Important architecture distinction:

- `models_transformer.py` is the recovered two-block attention model (5,954 parameters);
- `models_transformer_ori.py` is the one-block model (4,834 parameters);
- `best_model_indoor.pt` strictly loads only into `models_transformer_ori.py`;
- `models.py` is the Social-GAN baseline imported by `scripts/evaluate_model.py`.

Run the regression suite with:

```powershell
conda run -n ta-gan-rtx python -m unittest discover -s tests -v
```

If an original source file is recovered from the old computer, retain it separately and compare it with the tested restoration before replacing anything.

## Offline trajectory experiments

The offline scripts do not require ROS. The real-time LiDAR scripts (`main.py` and `test.py`) import `rospy` and ROS message types; native Windows ROS support is a separate issue and is not part of this Python environment setup.

Before running Reviewer 1 experiments, record:

1. the repository commit;
2. the exact model variant, source provenance, and checkpoint hash;
3. the dataset split manifest and preprocessing command;
4. random seeds and checkpoint path; and
5. per-sample predictions, metrics, and timing logs.
