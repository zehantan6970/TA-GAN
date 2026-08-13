# Reviewer 2 Comment 3: Parameter Count and Inference Latency

## 1. conclusion

For one independent obstacle history, prediction-only mean wall-clock latency was **0.8003 ms on an NVIDIA GeForce RTX 3060** and **0.4077 ms on one CPU thread of an Intel Core i7-14700**. The corresponding p95 values were 0.8611 ms
and 0.4259 ms.

The zero-parameter CVM-last baseline was substantially faster: 0.0391 ms on the GPU and 0.0136 ms on the CPU. Thus, the defensible conclusion is that the evaluated TA-GAN generator is compact and has low prediction-only latency on the revision machine, while CVM retains a clear computational advantage. 

## 2. Primary batch-one result

| Device | Method | Parameters | Mean (ms) | Median (ms) | Std. (ms) | p95 (ms) |
|---|---|---:|---:|---:|---:|---:|
| RTX 3060 | TA-GAN | 4,834 | 0.8003 | 0.7840 | 0.0696 | 0.8611 |
| RTX 3060 | CVM-last | 0 | 0.0391 | 0.0350 | 0.0093 | 0.0632 |
| i7-14700, 1 thread | TA-GAN | 4,834 | 0.4077 | 0.3959 | 0.0745 | 0.4259 |
| i7-14700, 1 thread | CVM-last | 0 | 0.0136 | 0.0131 | 0.0018 | 0.0173 |

The GPU wall-clock and CUDA-event values agree closely (TA-GAN means of 0.8003 and 0.8001 ms), indicating that synchronization is handled correctly. CVM is about 20.5 times faster on the GPU and 30.1 times faster on the single CPU thread. This speed gap should be reported rather than hidden: E2 addresses the accuracy trade-off, while E4 quantifies the computational trade-off.

## 3. Batch scaling (supplementary)

| Device | Batch | TA-GAN mean (ms/call) | CVM mean (ms/call) | TA-GAN histories/s |
|---|---:|---:|---:|---:|
| RTX 3060 | 1 | 0.8003 | 0.0391 | 1,249.5 |
| RTX 3060 | 8 | 0.8313 | 0.0352 | 9,623.2 |
| RTX 3060 | 32 | 0.8383 | 0.0365 | 38,170.5 |
| RTX 3060 | 128 | 0.8278 | 0.0366 | 154,621.3 |
| i7-14700, 1 thread | 1 | 0.4077 | 0.0136 | 2,453.1 |
| i7-14700, 1 thread | 8 | 0.6318 | 0.0132 | 12,662.6 |
| i7-14700, 1 thread | 32 | 1.3860 | 0.0132 | 23,088.3 |
| i7-14700, 1 thread | 128 | 4.3069 | 0.0136 | 29,720.1 |

Batch throughput is supplementary only. Batch=1 is the primary number for an online obstacle prediction call.
