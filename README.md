# BaybAI OCR - Synthetic Dataset

Generates a synthetic dataset of Baybayin characters and text images for training and evaluating Optical Character Recognition (OCR) models.

## Requirements

- [Python 3](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads/)

## Getting Started

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

2. Activate the virtual environment:

   ```bash
   # On Windows
   .venv\Scripts\activate
   # On Unix or MacOS
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Put all your background textures inside the `backgrounds` folder.

5. Generate the dataset:

   ```bash
   python3 src/generate.py --num-samples 1000 --height 48
   ```

### Options

| Option          | Data Type | Description                          |
| :-------------- | --------- | :----------------------------------- |
| `--num-samples` | `Integer` | Number of samples to generate.       |
| `--height`      | `Integer` | Final height of the images (pixels). |

## License

This project is licensed under the terms described in the [LICENSE](./LICENSE) file.
