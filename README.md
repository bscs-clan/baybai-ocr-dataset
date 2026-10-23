# BaybAI OCR - Synthetic Dataset

Generates a synthetic dataset of random Baybayin text images for training and evaluating Optical Character Recognition (OCR) models. Each sample contains one to three pseudo-words, with two to five Baybayin glyphs per word, spaces, and Baybayin punctuation; the text is generated without a dictionary. Across the dataset, every Baybayin Unicode codepoint from U+1700 through U+1714 and both punctuation marks U+1735 and U+1736 occur equally often in labels. Spaces are excluded from that balance. At least two samples are needed to balance these characters while keeping each sample to at most three words.

Labels use Baybayin Unicode characters from U+1700 through U+1714, including U+170D TAGALOG LETTER RA, plus U+1735 and U+1736 for punctuation. Spaces separate words. U+1715 PAMUDPOD and U+171F ARCHAIC RA are not generated.

## Requirements

- [Python 3](https://www.python.org/downloads/)
- [Git](https://git-scm.com/downloads/)


## Getting Started

SVG samples use CairoSVG and the native Cairo library. The generator uses SVG characters by default for 70% of samples and Noto Sans Tagalog for 30%; use `--font-prob 0` for SVG-only or `--font-prob 1` for font-only generation. SVG text samples include vector-rendered spaces, commas, and periods.

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
   python3 src/generate.py --num-samples 1000 --height 48 --font-prob 0.1 --augment-prob 0.1
   ```

### Options

| Option          | Data Type | Description                          |
| :-------------- | --------- | :----------------------------------- |
| `--num-samples` | `Integer` | Number of samples to generate.       |
| `--height`      | `Integer` | Final height of the images (pixels). |
| `--font-prob` | `Float` | Probability of using Noto Sans Tagalog instead of SVG character assets (default: `0.30`). Set to `0` for SVG only. |
| `--font-path` | `Path` | Font path used when font rendering is selected. |
| `--augment-prob` | `Float` | Probability of applying visual augmentation. Lower values are faster. |
| `--no-augment` | `Flag` | Disable augmentation for faster generation. |

For the fastest built-in path:

```bash
python3 src/generate.py --num-samples 1000 --height 48 --no-augment
```

## Tests

```bash
python -m pytest
```

## License

This project is licensed under the terms described in the [LICENSE](./LICENSE) file.
