# CRNN Dataset Generator

This Python script generates a synthetic dataset of Baybayin characters and text images for training and evaluating Optical Character Recognition (OCR) models, specifically Convolutional Recurrent Neural Networks (CRNNs).

## Features

- **Variations & Augmentation**:
  - Random character spacing and scaling.
  - Rotation, Gaussian blur, and noise injection.
  - Background texture blending with variable opacity.

- **Output Formats**:
  - Image files (PNG).
  - Label files in `.txt` format.
  - Optional grayscale conversion and resizing.

## Requirements

- [Python 3](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads/)

## Getting Started

1. Clone the repository:

```bash
git clone https://github.com/bscs-clan/crnn-dataset-gen.git
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Generate the dataset:

```bash
python3 ./src/generate.py --num-samples 1000 --height 64
```

### Options

| Option          | Data Type | Description                          |
| :-------------- | --------- | :----------------------------------- |
| `--num-samples` | `Integer` | Number of samples to generate.       |
| `--height`      | `Integer` | Final height of the images (pixels). |

## License

This project is licensed under the terms described in the [LICENSE](./LICENSE) file.
