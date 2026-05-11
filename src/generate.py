import os
import re
import sys
import math
import random
import logging
import argparse
import numpy as np
import cairosvg
from io import BytesIO
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


BAYBAYIN_DIR    = "baybayin"
BACKGROUNDS_DIR = "backgrounds"
WORDLIST_FILE   = "dictionaries/filipino-wordlist.txt"
OUTPUT_DIR      = "dataset"
IMAGES_SUBDIR   = "images"
LABELS_FILE     = "labels.txt"

NUM_SAMPLES     = 6_000

TARGET_HEIGHT   = 48
SVG_RENDER_SIZE = 256

NOTO_TAGALOG_FONT_PATH = "fonts/NotoSansTagalog-Regular.ttf"
FONT_RENDER_PROB       = 0.5
FONT_RENDER_SCALE      = 4

CHAR_SPACING_MIN  = 2
CHAR_SPACING_MAX  = 10
CHAR_PADDING_X    = 0
CHAR_PADDING_Y    = 0
CHAR_SCALE_MIN    = 0.9
CHAR_SCALE_MAX    = 1

SVG_STROKE_NOMINAL = 32
SVG_STROKE_JITTER  =  8

ROTATION_RANGE     = (-3, 3)
BLUR_SIGMA_RANGE   = (0, 0.5)
NOISE_STD_RANGE    = (0, 14)
SCALE_DISTORT_PROB = 0.40
SCALE_DISTORT_X    = (0.90, 1.10)
SCALE_DISTORT_Y    = (0.90, 1.10)
AUGMENT_PROB       = 0.85

BG_JITTER_FRAC  = 0.04
ALPHA_MIN       = 0.85
ALPHA_MAX       = 1

GRAYSCALE_OUTPUT = True

RANDOM_SEED = None


logging.basicConfig(
   level=logging.INFO,
   format="%(asctime)s  %(levelname)-8s  %(message)s",
   datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FOLDER_TO_UNICODE: dict[str, str] = {
      "a":        "\u1700",
   "e_i":      "\u1701",
   "o_u":      "\u1702",

      "ka":       "\u1703",
   "ke_ki":    "\u1703\u1712",
   "ko_ku":    "\u1703\u1713",
   "k":        "\u1703\u1714",

      "ga":       "\u1704",
   "ge_gi":    "\u1704\u1712",
   "go_gu":    "\u1704\u1713",
   "g":        "\u1704\u1714",

      "nga":      "\u1705",
   "nge_ngi":  "\u1705\u1712",
   "ngo_ngu":  "\u1705\u1713",
   "ng":       "\u1705\u1714",

      "ta":       "\u1706",
   "te_ti":    "\u1706\u1712",
   "to_tu":    "\u1706\u1713",
   "t":        "\u1706\u1714",

      "da":       "\u1707",
   "de_di":    "\u1707\u1712",
   "do_du":    "\u1707\u1713",
   "d":        "\u1707\u1714",

      "na":       "\u1708",
   "ne_ni":    "\u1708\u1712",
   "no_nu":    "\u1708\u1713",
   "n":        "\u1708\u1714",

      "pa":       "\u1709",
   "pepi":     "\u1709\u1712",
   "po_pu":    "\u1709\u1713",
   "p":        "\u1709\u1714",

      "ba":       "\u170A",
   "be_bi":    "\u170A\u1712",
   "bo_bu":    "\u170A\u1713",
   "b":        "\u170A\u1714",

      "ma":       "\u170B",
   "me_mi":    "\u170B\u1712",
   "mo_mu":    "\u170B\u1713",
   "m":        "\u170B\u1714",

      "ya":       "\u170C",
   "ye_yi":    "\u170C\u1712",
   "yo_yu":    "\u170C\u1713",
   "y":        "\u170C\u1714",

      "ra":       "\u170D",
   "re_ri":    "\u170D\u1712",
   "ro_ru":    "\u170D\u1713",
   "r":        "\u170D\u1714",

      "la":       "\u170E",
   "le_li":    "\u170E\u1712",
   "lo_lu":    "\u170E\u1713",
   "l":        "\u170E\u1714",

      "wa":       "\u170F",
   "we_wi":    "\u170F\u1712",
   "wo_wu":    "\u170F\u1713",
   "w":        "\u170F\u1714",

      "sa":       "\u1710",
   "se_si":    "\u1710\u1712",
   "so_su":    "\u1710\u1713",
   "s":        "\u1710\u1714",

      "ha":       "\u1711",
   "he_hi":    "\u1711\u1712",
   "ho_hu":    "\u1711\u1713",
   "h":        "\u1711\u1714",
}

_TAGALOG_VALID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz")

WORD_MIN_LEN = 2
WORD_MAX_LEN = 24

def _clean_word(raw: str) -> str:
   word = raw.strip()
   word = word.replace("\u00ad", "")
   word = word.lower()
   return word

def _is_valid_tagalog_word(word: str) -> bool:
   if not (WORD_MIN_LEN <= len(word) <= WORD_MAX_LEN):
      return False
   if "-" in word or "'" in word or "\'" in word:
      return False

   return all(c in _TAGALOG_VALID_CHARS for c in word)

def load_wordlist(filepath: str) -> list[str]:
   path = Path(filepath)
   if not path.is_file():
      raise FileNotFoundError(f"Wordlist not found: {filepath}")

   words: list[str] = []
   skipped = 0
   with open(path, "r", encoding="utf-8") as fh:
      for line in fh:
         if not line.strip() or line.lstrip().startswith("#"):
            continue
         word = _clean_word(line)
         if _is_valid_tagalog_word(word):
            words.append(word)
         else:
            skipped += 1

   if not words:
      raise ValueError(
         f"No valid words found in '{filepath}'.  "
         "Check WORD_MIN_LEN / WORD_MAX_LEN or the file contents."
      )

   seen: set[str] = set()
   unique: list[str] = []
   for w in words:
      if w not in seen:
         seen.add(w)
         unique.append(w)

   log.info(
      "Loaded %d unique words from '%s'  (skipped %d invalid entries)",
      len(unique), filepath, skipped,
   )
   return unique

_CONSONANTS = frozenset("bdghklmnprstwy")

_VOWELS     = frozenset("aeiou")

def _get_folder(consonant: Optional[str], vowel: Optional[str]) -> Optional[str]:
   if consonant is None:
      if vowel == "a":
         return "a"
      elif vowel in ("e", "i"):
         return "e_i"
      elif vowel in ("o", "u"):
         return "o_u"
      return None

   if vowel is None:
      return consonant

   if vowel == "a":
      return ("nga" if consonant == "ng" else consonant + "a")

   elif vowel in ("e", "i"):
      if consonant == "ng":
         return "nge_ngi"
      elif consonant == "p":
         return "pepi"
      else:
         return f"{consonant}e_{consonant}i"

   elif vowel in ("o", "u"):
      if consonant == "ng":
         return "ngo_ngu"
      else:
         return f"{consonant}o_{consonant}u"

   return None

def transliterate_to_baybayin(word: str) -> tuple[list[str], str]:
   word = word.lower().strip()
   folders: list[str] = []
   i, n = 0, len(word)

   while i < n:
      ch = word[i]

      if ch == "n" and i + 1 < n and word[i + 1] == "g":
         next_ch = word[i + 2] if i + 2 < n else None
         if next_ch in _VOWELS:
            folder = _get_folder("ng", next_ch)
            i += 3
         else:
            folder = _get_folder("ng", None)
            i += 2
         if folder and folder in FOLDER_TO_UNICODE:
            folders.append(folder)
         continue

      if ch in _CONSONANTS:
         next_ch = word[i + 1] if i + 1 < n else None
         if next_ch in _VOWELS:
            folder = _get_folder(ch, next_ch)
            i += 2
         else:
            folder = _get_folder(ch, None)
            i += 1
         if folder and folder in FOLDER_TO_UNICODE:
            folders.append(folder)
         continue

      if ch in _VOWELS:
         folder = _get_folder(None, ch)
         if folder and folder in FOLDER_TO_UNICODE:
            folders.append(folder)
         i += 1
         continue

      i += 1

   unicode_str = "".join(FOLDER_TO_UNICODE[f] for f in folders)
   return folders, unicode_str

def build_svg_library(baybayin_dir: str) -> dict[str, list[str]]:
   base = Path(baybayin_dir)
   if not base.is_dir():
      raise FileNotFoundError(f"Baybayin asset directory not found: {baybayin_dir}")

   library: dict[str, list[str]] = {}
   for folder in sorted(base.iterdir()):
      if not folder.is_dir():
         continue
      name = folder.name
      if name not in FOLDER_TO_UNICODE:
         log.warning("Skipping unknown Baybayin folder: '%s'", name)
         continue
      svgs = sorted(folder.glob("*.svg"))
      if not svgs:
         log.warning("No SVG files found in '%s'", folder)
         continue
      library[name] = [str(p) for p in svgs]
      log.debug("  %s  →  %d variant(s)", name, len(svgs))

   log.info("SVG library built: %d character classes loaded", len(library))
   return library

def _patch_svg(svg_content: str, stroke_width: float) -> str:

   svg_content = re.sub(
      r'<rect\b[^>]*\bfill\s*=\s*["\'](?:white|#fff|#ffffff)["\'][^>]*/?>',
      "",
      svg_content,
      flags=re.IGNORECASE,
   )

   sw_str = f"{stroke_width:.2f}"
   svg_content = re.sub(
      r'stroke-width\s*=\s*"[^"]*"',
      f'stroke-width="{sw_str}"',
      svg_content,
   )

   svg_content = re.sub(
      r'(stroke-width\s*:\s*)[\d.]+',
      lambda m: m.group(1) + sw_str,
      svg_content,
   )

   return svg_content

def _make_white_transparent(img: Image.Image, threshold: int = 220) -> Image.Image:
   arr = np.array(img, dtype=np.uint8)
   r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
   white_mask = (r >= threshold) & (g >= threshold) & (b >= threshold)
   arr[:, :, 3] = np.where(white_mask, 0, a)
   return Image.fromarray(arr, "RGBA")

def load_svg_character(
   svg_path: str,
   char_height: int,
   stroke_width: float = SVG_STROKE_NOMINAL,
) -> Image.Image:
   with open(svg_path, "r", encoding="utf-8") as fh:
      svg_text = fh.read()

   svg_text = _patch_svg(svg_text, stroke_width)

   png_bytes = cairosvg.svg2png(
      bytestring=svg_text.encode("utf-8"),
      output_width=SVG_RENDER_SIZE,
      output_height=SVG_RENDER_SIZE,
      background_color=None,
   )

   img = Image.open(BytesIO(png_bytes)).convert("RGBA")

   img = _make_white_transparent(img, threshold=220)

   img = img.resize((char_height, char_height), Image.LANCZOS)
   return img


# ---------------------------------------------------------------------------
# Font-based rendering (Noto Sans Tagalog TTF)
# ---------------------------------------------------------------------------

_font_cache: dict[int, ImageFont.FreeTypeFont] = {}

def _get_font(size: int, font_path: str = NOTO_TAGALOG_FONT_PATH) -> ImageFont.FreeTypeFont:
   """Return a cached FreeTypeFont at the requested pixel size."""
   if size not in _font_cache:
      _font_cache[size] = ImageFont.truetype(font_path, size)
   return _font_cache[size]


def render_word_image_font(
   unicode_str: str,
   target_height: int,
   font_path: str = NOTO_TAGALOG_FONT_PATH,
   render_scale: int = FONT_RENDER_SCALE,
) -> Optional[Image.Image]:
   """Render a Baybayin Unicode string with the Noto Sans Tagalog TTF font.

   Renders at ``target_height * render_scale`` internally, then downscales
   to ``target_height`` for crisp anti-aliasing, returning an RGBA image
   with a transparent background and black glyphs.
   """
   if not unicode_str:
      return None

   render_h = target_height * render_scale
   font = _get_font(render_h, font_path)

   # Measure bounding box at render resolution
   dummy = Image.new("RGBA", (1, 1))
   draw  = ImageDraw.Draw(dummy)
   bbox  = draw.textbbox((0, 0), unicode_str, font=font)
   text_w = bbox[2] - bbox[0]
   text_h = bbox[3] - bbox[1]

   if text_w <= 0 or text_h <= 0:
      return None

   pad_x = int(CHAR_PADDING_X * render_scale)
   pad_y = int(CHAR_PADDING_Y * render_scale)
   canvas_w = text_w + 2 * pad_x
   canvas_h = max(render_h, text_h) + 2 * pad_y

   canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
   draw   = ImageDraw.Draw(canvas)

   # Center vertically
   x = pad_x - bbox[0]
   y = pad_y + (max(render_h, text_h) - text_h) // 2 - bbox[1]
   draw.text((x, y), unicode_str, font=font, fill=(0, 0, 0, 255))

   # Downscale to target height, preserving aspect ratio
   aspect  = canvas_w / canvas_h
   out_h   = target_height + 2 * CHAR_PADDING_Y
   out_w   = max(1, round(out_h * aspect))
   canvas  = canvas.resize((out_w, out_h), Image.LANCZOS)

   return canvas

def render_word_image(
   folder_sequence: list[str],
   svg_library: dict[str, list[str]],
   target_height: int,
) -> Optional[Image.Image]:
   if not folder_sequence:
      return None

   word_stroke_jitter = random.uniform(-SVG_STROKE_JITTER, SVG_STROKE_JITTER)
   word_stroke_w = max(12.0, SVG_STROKE_NOMINAL + word_stroke_jitter)

   char_imgs: list[Image.Image] = []
   for folder in folder_sequence:
      paths = svg_library.get(folder)
      if not paths:
         log.debug("No SVGs found for folder '%s' – skipping character", folder)
         continue

      svg_path = random.choice(paths)

      scale  = random.uniform(CHAR_SCALE_MIN, CHAR_SCALE_MAX)
      char_h = max(16, int(target_height * scale))

      char_img = load_svg_character(svg_path, char_h, stroke_width=word_stroke_w)
      

      char_angle = random.uniform(-5.0, 5.0)
      char_img = char_img.rotate(char_angle, resample=Image.BICUBIC, expand=True)

      char_imgs.append(char_img)

   if not char_imgs:
      return None

   gaps = [
      random.randint(CHAR_SPACING_MIN, CHAR_SPACING_MAX)
      for _ in range(len(char_imgs) - 1)
   ]

   total_w = (
      sum(img.width for img in char_imgs)
      + sum(gaps)
      + 2 * CHAR_PADDING_X
   )
   
   max_char_h = max(img.height for img in char_imgs)
   canvas_base_h = max(target_height, max_char_h)
   canvas_h = canvas_base_h + 2 * CHAR_PADDING_Y

   canvas = Image.new("RGBA", (total_w, canvas_h), (0, 0, 0, 0))

   x = CHAR_PADDING_X
   for idx, img in enumerate(char_imgs):

      y = CHAR_PADDING_Y + (canvas_base_h - img.height) // 2
      canvas.paste(img, (x, y), mask=img)
      x += img.width
      if idx < len(gaps):
         x += gaps[idx]

   return canvas

def _pad_or_crop_to(img: Image.Image, w: int, h: int) -> Image.Image:
   canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
   x = (w - img.width)  // 2
   y = (h - img.height) // 2
   canvas.paste(img, (x, y), mask=img)
   return canvas

def apply_augmentation(img: Image.Image) -> Image.Image:
   if random.random() > AUGMENT_PROB:
      return img

   orig_w, orig_h = img.size

   angle = random.uniform(*ROTATION_RANGE)
   img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

   if random.random() < 0.6:
      blur_radius = random.uniform(0.5, 1.2)
      blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
      arr = np.array(blurred, dtype=np.float32)
      
      h, w = arr.shape[:2]
      grid_h, grid_w = max(2, h // 8), max(2, w // 8)
      small_noise = np.random.uniform(90, 160, (grid_h, grid_w)).astype(np.float32)
      noise_img = Image.fromarray(small_noise).resize((w, h), Image.BICUBIC)
      thresh = np.array(noise_img, dtype=np.float32)
      
      alpha = arr[:, :, 3]
      new_alpha = np.clip((alpha - thresh) * 5.0 + 128, 0, 255)
      
      arr[:, :, 3] = new_alpha
      arr = arr.astype(np.uint8)
      img = Image.fromarray(arr, "RGBA")

   sigma = random.uniform(*BLUR_SIGMA_RANGE)
   if sigma > 0.25:
      img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

   noise_std = random.uniform(*NOISE_STD_RANGE)
   if noise_std > 1.0:
      arr  = np.array(img, dtype=np.float32)
      alpha_norm = arr[:, :, 3] / 255.0
      noise = np.random.normal(0.0, noise_std, arr.shape[:2])
      for c in range(3):
         arr[:, :, c] += noise * alpha_norm
      arr = np.clip(arr, 0, 255).astype(np.uint8)
      img = Image.fromarray(arr, "RGBA")

   if random.random() < SCALE_DISTORT_PROB:
      sx = random.uniform(*SCALE_DISTORT_X)
      sy = random.uniform(*SCALE_DISTORT_Y)
      new_w = max(8, int(img.width * sx))
      new_h = max(8, int(img.height * sy))
      img = img.resize((new_w, new_h), Image.LANCZOS)

   return img

def load_background_paths(backgrounds_dir: str) -> list[str]:
   base = Path(backgrounds_dir)
   if not base.is_dir():
      raise FileNotFoundError(f"Backgrounds directory not found: {backgrounds_dir}")

   exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
   paths = sorted(p for p in base.iterdir() if p.suffix.lower() in exts)

   if not paths:
      raise ValueError(f"No background images found in: {backgrounds_dir}")

   log.info("Found %d background image(s) in '%s'", len(paths), backgrounds_dir)
   return [str(p) for p in paths]

def compose_with_background(
   text_img: Image.Image,
   bg_paths: list[str],
   target_height: int,
) -> Image.Image:

   bg = Image.open(random.choice(bg_paths)).convert("RGB")

   tw, th = text_img.size
   
   max_jitter_x = int(tw * BG_JITTER_FRAC)
   max_jitter_y = int(th * BG_JITTER_FRAC)
   
   patch_w = tw + 2 * max_jitter_x
   patch_h = th + 2 * max_jitter_y

   if bg.width < patch_w or bg.height < patch_h:
      scale = max(patch_w / bg.width, patch_h / bg.height) * 1.05
      bg = bg.resize(
         (math.ceil(bg.width * scale), math.ceil(bg.height * scale)),
         Image.LANCZOS,
      )

   max_x = max(0, bg.width  - patch_w)
   max_y = max(0, bg.height - patch_h)
   cx = random.randint(0, max_x)
   cy = random.randint(0, max_y)
   bg_patch = bg.crop((cx, cy, cx + patch_w, cy + patch_h))

   jitter_x = int(max_jitter_x * random.uniform(-1, 1))
   jitter_y = int(max_jitter_y * random.uniform(-1, 1))

   paste_x = max_jitter_x + jitter_x
   paste_y = max_jitter_y + jitter_y

   bg_arr = np.array(bg_patch, dtype=np.float32)
   luminance = 0.299 * bg_arr[:, :, 0] + 0.587 * bg_arr[:, :, 1] + 0.114 * bg_arr[:, :, 2]
   avg_lum = np.mean(luminance)

   opacity = random.uniform(ALPHA_MIN, ALPHA_MAX)
   

   arr = np.array(text_img, dtype=np.float32).copy()


   if avg_lum < 110:

      arr[:, :, 0] = 255
      arr[:, :, 1] = 255
      arr[:, :, 2] = 255
      
   if opacity < 0.999:
      arr[:, :, 3] = arr[:, :, 3] * opacity
      
   text_layer = Image.fromarray(arr.astype(np.uint8), "RGBA")

   composite = bg_patch.convert("RGBA")
   composite.paste(text_layer, (paste_x, paste_y), mask=text_layer)
   composite = composite.convert("RGB")

   aspect = tw / th
   out_h  = target_height
   out_w  = max(1, round(out_h * aspect))
   composite = composite.resize((out_w, out_h), Image.LANCZOS)

   return composite

def save_sample(
   img: Image.Image,
   unicode_label: str,
   idx: int,
   images_dir: Path,
   labels_fh,
   grayscale: bool = GRAYSCALE_OUTPUT,
) -> str:
   filename = f"img_{idx:05d}.png"
   out_path = images_dir / filename

   out_img = ImageOps.grayscale(img).convert("RGB") if grayscale else img
   out_img.save(str(out_path), format="PNG", optimize=False)

   labels_fh.write(f"{filename}\t{unicode_label}\n")

   return filename

def generate_dataset(
   num_samples: int      = NUM_SAMPLES,
   target_height: int    = TARGET_HEIGHT,
   baybayin_dir: str     = BAYBAYIN_DIR,
   backgrounds_dir: str  = BACKGROUNDS_DIR,
   wordlist_file: str    = WORDLIST_FILE,
   output_dir: str       = OUTPUT_DIR,
   grayscale: bool       = GRAYSCALE_OUTPUT,
   seed: Optional[int]   = RANDOM_SEED,
   font_prob: float      = FONT_RENDER_PROB,
   font_path: str        = NOTO_TAGALOG_FONT_PATH,
) -> None:
   if seed is not None:
      random.seed(seed)
      np.random.seed(seed)

   out_root   = Path(output_dir)
   images_dir = out_root / IMAGES_SUBDIR
   images_dir.mkdir(parents=True, exist_ok=True)
   labels_path = out_root / LABELS_FILE

   words    = load_wordlist(wordlist_file)
   svg_lib  = build_svg_library(baybayin_dir)
   bg_paths = load_background_paths(backgrounds_dir)

   # Validate font path when font rendering is enabled
   use_font = font_prob > 0.0
   if use_font:
      font_path_obj = Path(font_path)
      if not font_path_obj.is_file():
         log.warning(
            "Noto Sans Tagalog font not found at '%s'. "
            "Font rendering disabled (font_prob forced to 0).",
            font_path,
         )
         use_font  = False
         font_prob = 0.0
      else:
         log.info("Font rendering enabled  (prob=%.0f%%)  font: %s", font_prob * 100, font_path)

   log.info("Pre-checking %d words for translatability …", len(words))
   valid: list[tuple[str, list[str], str]] = []
   for w in words:
      folders, ustr = transliterate_to_baybayin(w)

      has_svg  = folders and ustr and any(f in svg_lib for f in folders)
      has_font = use_font and bool(ustr)
      if has_svg or has_font:
         valid.append((w, folders, ustr))

   if not valid:
      raise ValueError(
         "No words could be transliterated AND rendered.  "
         "Check your wordlist, SVG folder names, and FOLDER_TO_UNICODE mapping."
      )
   log.info("%d / %d words are renderable", len(valid), len(words))

   failures = 0
   generated = 0

   with open(labels_path, "w", encoding="utf-8") as labels_fh:
      while generated < num_samples:
         word, folders, unicode_label = random.choice(valid)

         try:
            # Choose rendering source: font or SVG
            use_font_this_sample = (
               use_font
               and random.random() < font_prob
               and bool(unicode_label)         # has a valid Unicode string
            )

            if use_font_this_sample:
               text_img = render_word_image_font(
                  unicode_label, target_height, font_path=font_path
               )
            else:
               text_img = render_word_image(folders, svg_lib, target_height)

            if text_img is None:
               failures += 1
               continue

            text_img = apply_augmentation(text_img)

            final_img = compose_with_background(text_img, bg_paths, target_height)

            save_sample(
               final_img,
               unicode_label,
               generated + 1,
               images_dir,
               labels_fh,
               grayscale=grayscale,
            )

            generated += 1

            percent = 100 * (generated / num_samples)
            filled = int(50 * generated // num_samples)
            bar = '█' * filled + '-' * (50 - filled)
            sys.stdout.write(f'\rGenerating |{bar}| {percent:.1f}%')
            sys.stdout.flush()

         except Exception as exc:
            failures += 1
            log.warning("Sample failed (word='%s'): %s", word, exc)

            if failures > max(50, num_samples * 0.15):
               raise RuntimeError(
                  f"Failure rate is too high ({failures} failures after "
                  f"{generated} successes). Check your assets."
               ) from exc
   print()

   log.info(
      "Generation complete.  Images: %d  |  Failures: %d  |  Labels: %s",
      generated, failures, labels_path,
   )

def _build_parser() -> argparse.ArgumentParser:
   p = argparse.ArgumentParser(
      prog="generate.py",
      description=(
         "Baybayin Synthetic OCR Dataset Generator\n"
         "Generates PNG images of Baybayin words with Baybayin Unicode labels."
      ),
      formatter_class=argparse.RawDescriptionHelpFormatter,
   )
   p.add_argument(
      "--num-samples", "-n",
      type=int, default=NUM_SAMPLES, metavar="N",
      help=f"Number of images to generate (default: {NUM_SAMPLES})",
   )
   p.add_argument(
      "--height", "-H",
      type=int, default=TARGET_HEIGHT, metavar="PX",
      help=f"Output image height in pixels (default: {TARGET_HEIGHT})",
   )
   p.add_argument(
      "--baybayin-dir",
      type=str, default=BAYBAYIN_DIR, metavar="DIR",
      help=f"Directory with SVG character sub-folders (default: '{BAYBAYIN_DIR}')",
   )
   p.add_argument(
      "--backgrounds-dir",
      type=str, default=BACKGROUNDS_DIR, metavar="DIR",
      help=f"Directory with background images (default: '{BACKGROUNDS_DIR}')",
   )
   p.add_argument(
      "--wordlist",
      type=str, default=WORDLIST_FILE, metavar="FILE",
      help=f"Path to word list file (default: '{WORDLIST_FILE}')",
   )
   p.add_argument(
      "--output-dir", "-o",
      type=str, default=OUTPUT_DIR, metavar="DIR",
      help=f"Root output directory (default: '{OUTPUT_DIR}')",
   )
   p.add_argument(
      "--grayscale",
      action="store_true", default=GRAYSCALE_OUTPUT,
      help="Save output images as grayscale (default: on)",
   )
   p.add_argument(
      "--color",
      dest="grayscale", action="store_false",
      help="Save output images in colour (overrides --grayscale)",
   )
   p.add_argument(
      "--seed",
      type=int, default=RANDOM_SEED, metavar="INT",
      help="Random seed for reproducible generation (default: random)",
   )
   p.add_argument(
      "--font-prob",
      type=float, default=FONT_RENDER_PROB, metavar="PROB",
      help=(
         f"Probability [0.0–1.0] of rendering each sample with the Noto Sans "
         f"Tagalog TTF font instead of SVGs (default: {FONT_RENDER_PROB}). "
         "Set to 0 to use only SVGs, 1 to use only the font."
      ),
   )
   p.add_argument(
      "--font-path",
      type=str, default=NOTO_TAGALOG_FONT_PATH, metavar="PATH",
      help=f"Path to the Noto Sans Tagalog TTF font file (default: '{NOTO_TAGALOG_FONT_PATH}')",
   )
   p.add_argument(
      "--verbose", "-v",
      action="store_true",
      help="Enable DEBUG-level logging",
   )
   return p

def main() -> None:
   args = _build_parser().parse_args()

   if args.verbose:
      logging.getLogger().setLevel(logging.DEBUG)

   log.info("=" * 60)
   log.info("  Baybayin OCR Dataset Generator")
   log.info("  Samples : %d", args.num_samples)
   log.info("  Height  : %d px", args.height)
   log.info("  Output  : %s/", args.output_dir)
   log.info("  Colour  : %s", "No (grayscale)" if args.grayscale else "Yes")
   log.info("=" * 60)

   generate_dataset(
      num_samples     = args.num_samples,
      target_height   = args.height,
      baybayin_dir    = args.baybayin_dir,
      backgrounds_dir = args.backgrounds_dir,
      wordlist_file   = args.wordlist,
      output_dir      = args.output_dir,
      grayscale       = args.grayscale,
      seed            = args.seed,
      font_prob       = args.font_prob,
      font_path       = args.font_path,
   )

if __name__ == "__main__":
   main()
