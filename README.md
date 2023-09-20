# TA-GAN
The TA-GAN trajectory prediction model synergizes the Transformer, Attention, and GAN networks, adeptly overcoming the limitations inherent in previous trajectory prediction models, notably in managing intricate interactions and long-term dependencies. We further introduce a distinctive feature fusion strategy anchored on the Attention mechanism. This strategy effectively addresses the information loss associated with max-pooling and the oversight of interactions among dynamic obstacles. Recognizing the constraints posed by camera field-of-view limitations and the prohibitive costs of multi-line LiDAR in indoor trajectory datasets, we introduce the pioneering IndoorNar-Trajectory dataset: the first 2D LiDAR-based benchmark for indoor dynamic obstacle trajectory prediction.
## Download
The IndoorNar-Trajectory dataset can be downloaded from:
[Google Drive]

## Installation

Our project has been tested on torch=1.10.1 cuda=10.2 python=3.6

## Train

You can use the following scripts for the train: train_transformer_GAN.py 

## Test

You can use the following scripts for the test: transformer_test.py 

## Inference

You can use the following scripts for the inference: demo_transformer.py 

## Comparison Experiment

See the experiment directory.
## Acknowledgement
We appreciate the open-source of the following project:     [Social GAN](https://github.com/agrimgupta92/sgan).
