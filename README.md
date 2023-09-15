# TA-GAN
The TA-GAN trajectory prediction model synergizes the Transformer, Attention, and GAN networks, adeptly overcoming the limitations inherent in previous trajectory prediction models, notably in managing intricate interactions and long-term dependencies. We further introduce a distinctive feature fusion strategy anchored on the Attention mechanism. This strategy effectively addresses the information loss associated with max-pooling and the oversight of interactions among dynamic obstacles. Recognizing the constraints posed by camera field-of-view limitations and the prohibitive costs of multi-line LiDAR in indoor trajectory datasets, we introduce the pioneering IndoorNar-Trajectory dataset: the first 2D LiDAR-based benchmark for indoor dynamic obstacle trajectory prediction.
## Download
The IndoorNar-Trajectory dataset can be downloaded from:
[Google Drive]
## PyraBiNet++
PyraBiNet is an innovative hybrid model optimized for lightweight semantic segmentation tasks. This model ingeniously merges the merits of Convolutional Neural Networks (CNNs) and Transformers.
For details, see the PyraBiNet++ directory
## Installation

Our project has been tested on torch=1.13 cuda=11.7 python=3.7.5

## Train

You can use the following scripts for the train: T_V_en_base_with_F_4L_e4_BERT.py

## Inference

You can use the example demo_eval_iou.py to perform RGB images, depth images, and text descriptions. 

## Comparison Experiment

See the experiment directory.
## Acknowledgement
We appreciate the open-source of the following projects:     [Social GAN](https://github.com/agrimgupta92/sgan), and [ScanNet](https://github.com/ScanNet/ScanNet).
