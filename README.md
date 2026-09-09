# TBAGNet

**TBAGNet: A Triple-Branch Adaptive Gated Network for Distributed Acoustic Sensing Event Classification**

This repository contains the official implementation of TBAGNet.
The manuscript is currently under submission, and additional
reproducibility documentation will be provided upon publication.

## Review-Stage Notice

This repository accompanies the manuscript **“TBAGNet: A Triple-Branch Adaptive Gated Network for Distributed Acoustic Sensing Event Classification.”**

The manuscript is currently under submission. This repository is provided at this stage primarily for academic review and reproducibility assessment.

The current release includes the core model implementation, training and evaluation pipelines, dataset configurations, and fixed experimental splits. Public datasets are not redistributed.

Additional implementation details, dataset preparation instructions, pretrained checkpoints, and complete reproduction materials may be released after acceptance or publication.

The current repository is not an open-source release. Please see `LICENSE` for the applicable terms.

## Overview

TBAGNet is a multi-domain feature learning framework for distributed acoustic sensing (DAS) event classification. It learns complementary representations from three parallel branches:

1. **Temporal Branch** learns time-domain waveform features directly from raw DAS signals.
2. **Frequency Branch** uses rFFT and a real-imaginary spectral representation to learn frequency-domain features.
3. **Wavelet Time-Frequency Branch** uses a Real Morlet wavelet representation to model both local textures and global relations in the time-frequency domain.

The three branch representations are fused adaptively through **Adaptive Gated Aggregation (AGA)** before classification.

## Architecture

The paper-final TBAGNet configuration consists of the following components:

- **Temporal Branch:** a Conv1D-based encoder with depthwise-separable convolution for waveform feature extraction.
- **Frequency Branch:** rFFT with real and imaginary spectral views, spectral patch embedding, four PGI blocks, feature gating, and max pooling.
- **Wavelet Time-Frequency Branch:** a 64-scale Real Morlet wavelet representation resized to `64 × 64`.
  - **Wavelet Local Texture Path:** four convolutional stages for local texture learning.
  - **Wavelet Global Relation Path:** wavelet patch embedding followed by one PGI block and max pooling.
- **Adaptive Gated Aggregation:** AGA normalizes branch features, predicts sample- and channel-dependent branch weights, applies softmax across branches, and produces an adaptively weighted representation.

The final configuration uses Frequency PGI depth `4`, Wavelet Global Relation Path PGI depth `1`, Wavelet Local Texture Path depth `4`, max pooling, and AGA fusion.

## Repository Structure

```text
TBAGNet/
├── LICENSE
├── CITATION.cff
├── configs/
│   ├── railway_das.yaml
│   ├── fiberrisk.yaml
│   ├── bjdas.yaml
│   ├── efficient_dvs.yaml
│   └── metro_data.yaml
├── data/
│   ├── splits/
│   ├── dataset_adapters.py
│   ├── preprocessing.py
│   ├── railway_split.py
│   └── split_utils.py
├── scripts/
│   └── generate_railway_split.py
├── src/
│   └── tbagnet/
├── tests/
├── train.py
├── test.py
├── requirements.txt
└── environment.yml
```

## Experimental Reproduction

The repository provides the TBAGNet source code, experiment configurations, fixed data-split files, and training and evaluation workflows for:

- Railway-DAS
- FiberRisk
- BJDAS
- Efficient-DVS
- Metro-data

Dataset split and manifest files are stored under `data/splits/`. Raw datasets are not redistributed; users must prepare each dataset locally and provide its root path for training or evaluation.

## Data Preparation

The original datasets are not included in this repository due to dataset ownership and distribution policies. Users should obtain the datasets from their official sources and prepare them according to the corresponding dataset licenses and requirements.

The predefined train/validation/test splits are provided in `data/splits/`. These split files ensure consistent experimental evaluation.

After preparing the datasets, specify the dataset root path through `--data-root`. The dataset adapters load samples according to the provided manifests and split files.

## Environment and Installation

The implementation requires Python, PyTorch, and a CUDA-capable environment for GPU evaluation. Minimum package requirements are provided in:

- `requirements.txt`
- `environment.yml`

Either environment specification can be used to install the required dependencies.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Training

For example, train TBAGNet on Railway-DAS with:

```bash
python train.py \
  --config configs/railway_das.yaml \
  --data-root <DATASET_PATH> \
  --output-dir outputs/railway_das
```

## Evaluation

After training, evaluate a saved model with its corresponding YAML configuration:

```bash
python test.py \
  --config configs/railway_das.yaml \
  --checkpoint <CHECKPOINT_PATH> \
  --data-root <DATASET_PATH>
```

The YAML file is the only source of model configuration. The evaluation script loads only the state dictionary from the supplied checkpoint and reports a strict compatibility error if its structure does not match the YAML configuration.

## License

This repository is currently provided for manuscript review and reproducibility assessment only. All rights are reserved by the authors.

Please see `LICENSE` for the applicable review-stage terms.

A formal open-source license may be adopted for a future public release after acceptance or publication of the manuscript.

