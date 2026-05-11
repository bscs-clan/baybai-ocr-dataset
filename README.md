# BaybAI OCR - Synthetic Dataset

Generates a synthetic dataset of Baybayin characters and text images for training and evaluating Optical Character Recognition (OCR) models.

Labels are written as Unicode Baybayin text using the supported training alphabet U+1700 through U+1714, including U+170D TAGALOG LETTER RA. U+1715 PAMUDPOD and U+171F ARCHAIC RA are not generated unless matching assets are added.

## Requirements

- [Python 3](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads/)

SVG rendering uses CairoSVG and the native Cairo library. On Windows, if Cairo is not installed, the generator automatically falls back to font-only rendering when `--font-prob` is greater than `0`.

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
| `--font-prob` | `Float` | Probability of rendering from the bundled font instead of SVGs. Use `1` for fastest generation. |
| `--augment-prob` | `Float` | Probability of applying visual augmentation. Lower values are faster. |
| `--no-augment` | `Flag` | Disable augmentation for faster generation. |

For the fastest built-in path:

```bash
python3 src/generate.py --num-samples 1000 --height 48 --font-prob 1 --no-augment
```

## Tests

```bash
python -m pytest
```

## License

This project is licensed under the terms described in the [LICENSE](./LICENSE) file.
