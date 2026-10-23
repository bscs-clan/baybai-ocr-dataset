import os
import re
import sys
import math
import random
import logging
import argparse
import itertools
import numpy as np
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from typing import Optional
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


BAYBAYIN_DIR    = "baybayin"
BACKGROUNDS_DIR = "backgrounds"
OUTPUT_DIR      = "dataset"
IMAGES_SUBDIR   = "images"
LABELS_FILE     = "labels.txt"

NUM_SAMPLES     = 6_000

TARGET_HEIGHT   = 48
SVG_RENDER_SIZE = 256
SVG_RENDER_CACHE_SIZE = 8192
SVG_STROKE_CACHE_STEP = 4
BACKGROUND_CACHE_SIZE = 32
FONT_WORD_CACHE_SIZE  = 8192
PROGRESS_UPDATES      = 100

NOTO_TAGALOG_FONT_PATH = "fonts/NotoSansTagalog-Regular.ttf"
FONT_RENDER_SCALE      = 4
FONT_RENDER_PROB       = 0.30

PLAIN_BACKGROUND_PROB = 0.30

CHAR_SPACING_MIN  = 0
CHAR_SPACING_MAX  = 2
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

# Quality gating defaults (for readability at small heights like 48px)
# If a sample is too low-quality, it will be rejected and another one generated.
QUALITY_MIN_FOREGROUND_RATIO = 0.010
QUALITY_MAX_COMPONENTS       = 120
QUALITY_MIN_LAPLACIAN_VAR     = 6.0
QUALITY_MIN_CONTRAST_MARGIN   = 18.0

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

_CONSONANTS = frozenset("bdghklmnprstwy")
_VOWELS     = frozenset("aeiou")

MAX_WORDS_PER_SAMPLE = 3
MIN_SYLLABLES_PER_WORD = 2
MAX_CHARACTERS_PER_WORD = 5
MAX_SYLLABLES_PER_WORD = MAX_CHARACTERS_PER_WORD
COMMA_LABEL = "\u1735"
DOT_LABEL = "\u1736"
BALANCED_CODEPOINTS = (
   *(chr(codepoint) for codepoint in range(0x1700, 0x1715)),
   COMMA_LABEL,
   DOT_LABEL,
)
_BALANCE_CONSONANTS = tuple(sorted(_CONSONANTS)) + ("ng",)

def _balanced_sample_plan(num_samples: int) -> list[tuple[list[list[str]], str, str]]:
   """Plan random text whose Baybayin Unicode codepoints have equal totals."""
   if num_samples < 2:
      raise ValueError(
         "Balanced generation needs at least 2 samples to include every "
         "Baybayin codepoint and punctuation mark equally."
      )

   # Each of the 23 Baybayin codepoints, including punctuation, occurs C times.
   # Aim for about 3.5 Baybayin glyphs per sample instead of minimizing text length.
   codepoint_count = (7 * num_samples + 35) // 36
   total_glyphs = 18 * codepoint_count
   extra_glyphs = total_glyphs - 2 * num_samples

   # Spread the extra glyphs evenly, so samples have similar text lengths.
   base_increment, remainder = divmod(extra_glyphs, num_samples)
   sample_increments = [base_increment] * (num_samples - remainder)
   sample_increments.extend([base_increment + 1] * remainder)
   random.shuffle(sample_increments)
   sample_lengths = [2 + increment for increment in sample_increments]

   counts: dict[str, int] = {"a": codepoint_count, "e_i": codepoint_count, "o_u": codepoint_count}
   marked_counts = [[0, 0, 0] for _ in _BALANCE_CONSONANTS]
   for mark_index in range(3):
      remaining = codepoint_count
      family_index = mark_index
      while remaining:
         if sum(marked_counts[family_index]) < codepoint_count:
            marked_counts[family_index][mark_index] += 1
            remaining -= 1
         family_index = (family_index + 1) % len(_BALANCE_CONSONANTS)

   for consonant, marks in zip(_BALANCE_CONSONANTS, marked_counts):
      counts[_get_folder(consonant, "a")] = codepoint_count - sum(marks)
      counts[_get_folder(consonant, "e")] = marks[0]
      counts[_get_folder(consonant, "o")] = marks[1]
      counts[consonant] = marks[2]

   glyph_pool = [folder for folder, count in counts.items() for _ in range(count)]
   random.shuffle(glyph_pool)

   punctuation_pool = [","] * codepoint_count + ["."] * codepoint_count
   punctuation_pool.extend([""] * (num_samples - len(punctuation_pool)))
   random.shuffle(punctuation_pool)

   plan: list[tuple[list[list[str]], str, str]] = []
   pool_offset = 0
   for glyph_count in sample_lengths:
      glyphs = glyph_pool[pool_offset:pool_offset + glyph_count]
      pool_offset += glyph_count
      possible_word_lengths = [
         lengths
         for word_count in range(1, MAX_WORDS_PER_SAMPLE + 1)
         for lengths in itertools.product(
            range(MIN_SYLLABLES_PER_WORD, MAX_SYLLABLES_PER_WORD + 1),
            repeat=word_count,
         )
         if sum(lengths) == glyph_count
      ]
      word_lengths = random.choice(possible_word_lengths)
      words: list[list[str]] = []
      offset = 0
      for word_length in word_lengths:
         words.append(glyphs[offset:offset + word_length])
         offset += word_length

      label = " ".join("".join(FOLDER_TO_UNICODE[folder] for folder in word) for word in words)
      punctuation = punctuation_pool[len(plan)]
      punctuation_label = {",": COMMA_LABEL, ".": DOT_LABEL, "": ""}[punctuation]
      plan.append((words, label + punctuation_label, punctuation))

   return plan

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

      return [], ""

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
      if name not in FOLDER_TO_UNICODE and name not in {"comma", "dot"}:
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

_cairosvg_module = None


def _get_cairosvg():
   global _cairosvg_module
   if _cairosvg_module is not None:
      return _cairosvg_module

   try:
      import cairosvg
   except (ImportError, OSError) as exc:
      raise RuntimeError(
         "SVG rendering requires cairosvg plus the native Cairo library. "
         "Install Cairo/GTK for Windows, or run with --font-prob 1 to render with the bundled font only."
      ) from exc

   _cairosvg_module = cairosvg
   return _cairosvg_module


def _svg_rendering_available() -> bool:
   try:
      _get_cairosvg()
   except RuntimeError:
      return False
   return True


@lru_cache(maxsize=SVG_RENDER_CACHE_SIZE)
def _read_svg_text(svg_path: str) -> str:
   return Path(svg_path).read_text(encoding="utf-8")


@lru_cache(maxsize=SVG_RENDER_CACHE_SIZE)
def _load_svg_character_cached(
   svg_path: str,
   char_height: int,
   stroke_width: int,
) -> Image.Image:
   svg_text = _patch_svg(_read_svg_text(svg_path), stroke_width)

   cairosvg = _get_cairosvg()
   png_bytes = cairosvg.svg2png(
      bytestring=svg_text.encode("utf-8"),
      output_width=SVG_RENDER_SIZE,
      output_height=SVG_RENDER_SIZE,
      background_color=None,
   )

   img = Image.open(BytesIO(png_bytes)).convert("RGBA")
   img = _make_white_transparent(img, threshold=220)
   return img.resize((char_height, char_height), Image.LANCZOS)


def load_svg_character(
   svg_path: str,
   char_height: int,
   stroke_width: float = SVG_STROKE_NOMINAL,
) -> Image.Image:
   stroke_width = max(
      12,
      round(stroke_width / SVG_STROKE_CACHE_STEP) * SVG_STROKE_CACHE_STEP,
   )
   img = _load_svg_character_cached(svg_path, char_height, stroke_width)
   return img.copy()


# ---------------------------------------------------------------------------
# Font-based rendering (Noto Sans Tagalog TTF)
# ---------------------------------------------------------------------------

_font_cache: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}

def _get_font(size: int, font_path: str = NOTO_TAGALOG_FONT_PATH) -> ImageFont.FreeTypeFont:
   """Return a cached FreeTypeFont at the requested pixel size."""
   cache_key = (font_path, size)
   if cache_key not in _font_cache:
      _font_cache[cache_key] = ImageFont.truetype(font_path, size)
   return _font_cache[cache_key]


@lru_cache(maxsize=FONT_WORD_CACHE_SIZE)
def _render_word_image_font_cached(
   unicode_str: str,
   target_height: int,
   font_path: str,
   render_scale: int,
) -> Optional[Image.Image]:
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
   img = _render_word_image_font_cached(unicode_str, target_height, font_path, render_scale)
   return img.copy() if img is not None else None

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

def render_random_text_svg(
   words: list[list[str]],
   punctuation: str,
   svg_library: dict[str, list[str]],
   target_height: int,
) -> Optional[Image.Image]:
   """Render random words and punctuation from SVG assets with explicit spaces."""
   word_images = []
   for word_folders in words:
      img = render_word_image(word_folders, svg_library, target_height)
      if img is None:
         return None
      word_images.append(img)

   if not word_images:
      return None

   space_width = target_height
   punctuation_img = None
   if punctuation:
      punctuation_folder = "comma" if punctuation == "," else "dot"
      punctuation_paths = svg_library.get(punctuation_folder, [])
      if punctuation_paths:
         punctuation_img = load_svg_character(
            random.choice(punctuation_paths), target_height
         )
         bbox = punctuation_img.getchannel("A").getbbox()
         if bbox:
            punctuation_img = punctuation_img.crop(bbox)
      if punctuation_img is None:
         return None
   punctuation_width = punctuation_img.width if punctuation_img else 0
   height = max(img.height for img in word_images)
   width = sum(img.width for img in word_images) + space_width * (len(word_images) - 1) + punctuation_width
   canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
   x = 0
   for index, img in enumerate(word_images):
      canvas.alpha_composite(img, (x, (height - img.height) // 2))
      x += img.width
      if index < len(word_images) - 1:
         x += space_width

   if punctuation_img:
      canvas.alpha_composite(punctuation_img, (x, height - punctuation_img.height))
   return canvas

def _pad_or_crop_to(img: Image.Image, w: int, h: int) -> Image.Image:
   canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
   x = (w - img.width)  // 2
   y = (h - img.height) // 2
   canvas.paste(img, (x, y), mask=img)
   return canvas

def apply_augmentation(
   img: Image.Image,
   augment_prob: float = AUGMENT_PROB,
   *,
   safe_small_height: bool = False,
) -> Image.Image:
   if random.random() > augment_prob:
      return img

   # If training is at a very small height (48px), aggressively distorted
   # / noisy / blurry samples tend to become unreadable. Apply tighter caps.
   if safe_small_height:
      rot_range = (-2, 2)
      blur_sigma_range = (0, 0.30)
      noise_std_range = (0, 7)
      scale_distort_prob = 0.22
      scale_distort_x = (0.94, 1.06)
      scale_distort_y = (0.94, 1.06)
      alpha_noise = (0.5, 0.9)
   else:
      rot_range = ROTATION_RANGE
      blur_sigma_range = BLUR_SIGMA_RANGE
      noise_std_range = NOISE_STD_RANGE
      scale_distort_prob = SCALE_DISTORT_PROB
      scale_distort_x = SCALE_DISTORT_X
      scale_distort_y = SCALE_DISTORT_Y
      alpha_noise = (0.5, 1.2)

   angle = random.uniform(*rot_range)
   img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

   if random.random() < 0.55:
      blur_radius = random.uniform(*alpha_noise)
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

   sigma = random.uniform(*blur_sigma_range)
   if sigma > 0.25:
      img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

   noise_std = random.uniform(*noise_std_range)
   if noise_std > 1.0:
      arr = np.array(img, dtype=np.float32)
      alpha_norm = arr[:, :, 3] / 255.0
      noise = np.random.normal(0.0, noise_std, arr.shape[:2])
      for c in range(3):
         arr[:, :, c] += noise * alpha_norm
      arr = np.clip(arr, 0, 255).astype(np.uint8)
      img = Image.fromarray(arr, "RGBA")

   if random.random() < scale_distort_prob:
      sx = random.uniform(*scale_distort_x)
      sy = random.uniform(*scale_distort_y)
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


@lru_cache(maxsize=BACKGROUND_CACHE_SIZE)
def _load_background_image(bg_path: str) -> Image.Image:
   return Image.open(bg_path).convert("RGB")


def _rgb_to_gray_arr(img: Image.Image) -> np.ndarray:
   if img.mode != "RGB":
      img = img.convert("RGB")
   arr = np.array(img, dtype=np.float32)
   lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
   return lum


def _estimate_contrast_from_background(bg_patch: Image.Image) -> tuple[float, float]:
   """Return mean_lum and std_lum of a background patch."""
   lum = _rgb_to_gray_arr(bg_patch)
   return float(np.mean(lum)), float(np.std(lum))


def _choose_text_color_rgba_from_background(bg_patch: Image.Image, text_img: Image.Image) -> Image.Image:
   """
   Convert the RGBA glyph layer to either pure black or pure white
   using background contrast estimation.
   """
   bg_mean, bg_std = _estimate_contrast_from_background(bg_patch)

   # "Contrast" proxy for OCR: how far foreground luminance deviates from background mean.
   # Try both and pick the higher absolute difference.
   # (bg_std affects readability but we keep this simple and robust.)
   contrast_black = abs(bg_mean - 0.0)
   contrast_white = abs(bg_mean - 255.0)

   arr = np.array(text_img, dtype=np.float32).copy()
   if contrast_white >= contrast_black:
      arr[:, :, 0] = 255
      arr[:, :, 1] = 255
      arr[:, :, 2] = 255
   else:
      arr[:, :, 0] = 0
      arr[:, :, 1] = 0
      arr[:, :, 2] = 0

   return Image.fromarray(arr.astype(np.uint8), "RGBA")


def compose_with_background(
   text_img: Image.Image,
   bg_paths: list[str],
   target_height: int,
) -> Image.Image:

   tw, th = text_img.size

   max_jitter_x = int(tw * BG_JITTER_FRAC)
   max_jitter_y = int(th * BG_JITTER_FRAC)

   patch_w = tw + 2 * max_jitter_x
   patch_h = th + 2 * max_jitter_y

   # Probability of using plain background
   PLAIN_BACKGROUND_PROB = 0.30
   # 30% plain colors
   # 70% texture images

   use_plain = random.random() < PLAIN_BACKGROUND_PROB

   # ----------------------------
   # CREATE BACKGROUND
   # ----------------------------

   if use_plain:

      # White / Black / Random Color choices
      mode = random.choice([
         "white",
         "black",
         "random"
      ])

      if mode == "white":
         bg_color = (255,255,255)

      elif mode == "black":
         bg_color = (0,0,0)

      else:
         bg_color = (
            random.randint(0,255),
            random.randint(0,255),
            random.randint(0,255)
         )

      bg_patch = Image.new(
         "RGB",
         (patch_w, patch_h),
         bg_color
      )

   else:

      bg = _load_background_image(
         random.choice(bg_paths)
      )

      if bg.width < patch_w or bg.height < patch_h:
         scale = max(
            patch_w / bg.width,
            patch_h / bg.height
         ) * 1.05

         bg = bg.resize(
            (
               math.ceil(bg.width * scale),
               math.ceil(bg.height * scale)
            ),
            Image.LANCZOS
         )

      max_x = max(0, bg.width - patch_w)
      max_y = max(0, bg.height - patch_h)

      cx = random.randint(0, max_x)
      cy = random.randint(0, max_y)

      bg_patch = bg.crop(
         (
            cx,
            cy,
            cx + patch_w,
            cy + patch_h
         )
      )

   # ----------------------------
   # TEXT PLACEMENT
   # ----------------------------

   jitter_x = int(
      max_jitter_x * random.uniform(-1,1)
   )

   jitter_y = int(
      max_jitter_y * random.uniform(-1,1)
   )

   paste_x = max_jitter_x + jitter_x
   paste_y = max_jitter_y + jitter_y

   opacity = random.uniform(
      ALPHA_MIN,
      ALPHA_MAX
   )

   text_layer = _choose_text_color_rgba_from_background(
      bg_patch,
      text_img
   )

   if opacity < 0.999:

      arr = np.array(
         text_layer,
         dtype=np.float32
      )

      arr[:,:,3] *= opacity

      text_layer = Image.fromarray(
         arr.astype(np.uint8),
         "RGBA"
      )

   composite = bg_patch.convert("RGBA")

   composite.paste(
      text_layer,
      (paste_x, paste_y),
      mask=text_layer
   )

   composite = composite.convert("RGB")

   aspect = tw / th

   out_h = target_height
   out_w = max(
      1,
      round(out_h * aspect)
   )

   composite = composite.resize(
      (out_w, out_h),
      Image.LANCZOS
   )

   return composite

def _count_connected_components(mask: np.ndarray) -> int:
   """
   Lightweight 4-connected components count for boolean mask.
   mask: HxW bool
   """
   h, w = mask.shape
   visited = np.zeros_like(mask, dtype=bool)

   def neighbors(y: int, x: int):
      if y > 0: yield y - 1, x
      if y + 1 < h: yield y + 1, x
      if x > 0: yield y, x - 1
      if x + 1 < w: yield y, x + 1

   components = 0
   for y in range(h):
      row = mask[y]
      for x in range(w):
         if not row[x] or visited[y, x]:
            continue
         components += 1
         stack = [(y, x)]
         visited[y, x] = True
         while stack:
            cy, cx = stack.pop()
            for ny, nx in neighbors(cy, cx):
               if mask[ny, nx] and not visited[ny, nx]:
                  visited[ny, nx] = True
                  stack.append((ny, nx))
   return components


def compute_sample_quality(img: Image.Image) -> dict[str, float]:
   """
   Heuristic quality scoring using foreground segmentation + sharpness + components.
   """
   gray = ImageOps.grayscale(img)
   arr = np.array(gray, dtype=np.uint8)
   h, w = arr.shape[:2]

   # Foreground segmentation by thresholding away from background mid/mean.
   # Use Otsu-like heuristic via mean and std: text is expected to differ from bg.
   mean = float(np.mean(arr))
   std = float(np.std(arr))
   # Robust threshold: if text is dark -> below threshold, if bright -> above threshold.
   t_low = mean - 0.35 * std
   t_high = mean + 0.35 * std

   fg_dark = arr < t_low
   fg_bright = arr > t_high
   fg = fg_dark | fg_bright

   fg_ratio = float(np.mean(fg)) if h * w > 0 else 0.0

   # Component count on downsampled mask to be stable at small sizes
   small = Image.fromarray(fg.astype(np.uint8) * 255).resize(
      (max(1, w // 2), max(1, h // 2)), Image.NEAREST
   )
   small_mask = np.array(small, dtype=np.uint8) > 0
   components = float(_count_connected_components(small_mask)) if small_mask.size else 0.0

   # Sharpness estimate using Laplacian variance
   # (use simple discrete laplacian)
   a = arr.astype(np.float32)
   lap = (
      -4.0 * a
      + np.roll(a, 1, axis=0)
      + np.roll(a, -1, axis=0)
      + np.roll(a, 1, axis=1)
      + np.roll(a, -1, axis=1)
   )
   lap_var = float(np.var(lap))

   # Foreground/background contrast margin
   if np.any(fg):
      fg_mean = float(np.mean(arr[fg]))
   else:
      fg_mean = mean
   bg_mask = ~fg
   if np.any(bg_mask):
      bg_mean = float(np.mean(arr[bg_mask]))
   else:
      bg_mean = mean
   contrast_margin = abs(fg_mean - bg_mean)

   return {
      "fg_ratio": fg_ratio,
      "components": components,
      "lap_var": lap_var,
      "contrast_margin": contrast_margin,
   }


def is_sample_quality_good(img: Image.Image) -> tuple[bool, dict[str, float]]:
   q = compute_sample_quality(img)

   ok = True
   if q["fg_ratio"] < QUALITY_MIN_FOREGROUND_RATIO:
      ok = False
   if q["components"] > QUALITY_MAX_COMPONENTS:
      ok = False
   if q["lap_var"] < QUALITY_MIN_LAPLACIAN_VAR:
      ok = False
   if q["contrast_margin"] < QUALITY_MIN_CONTRAST_MARGIN:
      ok = False

   return ok, q


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
   output_dir: str       = OUTPUT_DIR,
   grayscale: bool       = GRAYSCALE_OUTPUT,
   seed: Optional[int]   = RANDOM_SEED,
   font_prob: float      = FONT_RENDER_PROB,
   font_path: str        = NOTO_TAGALOG_FONT_PATH,
   augment_prob: float   = AUGMENT_PROB,
   workers: Optional[int] = None,
) -> None:
   if seed is not None:
      random.seed(seed)
      np.random.seed(seed)

   sample_plan = _balanced_sample_plan(num_samples)
   augment_prob = min(1.0, max(0.0, augment_prob))
   font_prob = min(1.0, max(0.0, font_prob))

   out_root   = Path(output_dir)
   images_dir = out_root / IMAGES_SUBDIR
   images_dir.mkdir(parents=True, exist_ok=True)
   labels_path = out_root / LABELS_FILE

   svg_lib = build_svg_library(baybayin_dir)
   bg_paths = load_background_paths(backgrounds_dir)

   if font_prob > 0:
      font_path_obj = Path(font_path)
      if not font_path_obj.is_file():
         raise FileNotFoundError(f"Baybayin font not found: {font_path}")
      log.info("Font rendering enabled  (prob=%.0f%%)  font: %s", font_prob * 100, font_path)
   if font_prob < 1 and not _svg_rendering_available():
      if Path(font_path).is_file():
         log.warning("CairoSVG/native Cairo unavailable; using font rendering for all samples")
         font_prob = 1.0
      else:
         raise RuntimeError("SVG rendering requires CairoSVG and native Cairo; no font fallback is available.")
   if font_prob < 1:
      log.info("SVG rendering enabled for Baybayin character assets")
   log.info("Augmentation enabled  (prob=%.0f%%)", augment_prob * 100)

   log.info(
      "Generating balanced text: each Baybayin codepoint and punctuation mark appears %d times",
      (num_samples + 8) // 9,
   )

   failures = 0
   generated = 0
   progress_interval = max(1, num_samples // PROGRESS_UPDATES)
   safe_small_height = (target_height <= 48)

   # Multi-threaded generation. To keep filenames/labels consistent, each worker
   # generates until it finds a *single accepted* sample, then returns it.
   # The main thread saves accepted samples in submission order.
   import concurrent.futures

   if workers is None:
      workers = max(1, (os.cpu_count() or 1))
   workers = max(1, int(workers))

   def _generate_one_until_ok(
      task_idx: int,
      retry_round: int = 0,
   ) -> tuple[int, Optional[tuple[Image.Image, str]]]:
      try:
         base_local = 0 if seed is None else (seed + 10_000_000 + task_idx + retry_round * 200)

         # Try multiple times inside the worker to reduce wasted capacity.
         # (Acceptance rate can be low due to quality gating.)
         for attempt in range(200):
            if seed is not None:
               local_seed = base_local + attempt
               random.seed(local_seed)
               np.random.seed(local_seed % (2**32 - 1))

            word_folders, unicode_label, punctuation = sample_plan[task_idx]
            if font_prob > 0 and random.random() < font_prob:
               text_img = render_word_image_font(
                  unicode_label, target_height, font_path=font_path
               )
            else:
               text_img = render_random_text_svg(
                  word_folders, punctuation, svg_lib, target_height
               )

            if text_img is None:
               continue

            text_img = apply_augmentation(
               text_img,
               augment_prob=augment_prob,
               safe_small_height=safe_small_height,
            )

            final_img = compose_with_background(text_img, bg_paths, target_height)

            ok, _q = is_sample_quality_good(final_img)
            if not ok:
               continue

            return task_idx, (final_img, unicode_label)

         return task_idx, None

      except Exception as exc:
         log.warning("Worker generation failed (task_idx=%d): %s", task_idx, exc)
         return task_idx, None

   failures = 0
   generated = 0
   progress_interval = max(1, num_samples // PROGRESS_UPDATES)

   with open(labels_path, "w", encoding="utf-8") as labels_fh:
      with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
         futures: list[concurrent.futures.Future] = []
         task_counter = 0

         def _submit_one() -> None:
            nonlocal task_counter
            idx = task_counter
            task_counter += 1
            futures.append(ex.submit(_generate_one_until_ok, idx))

         # Keep a sliding window of in-flight tasks.
         in_flight_target = min(num_samples, workers * 4)
         for _ in range(in_flight_target):
            _submit_one()

         # Save accepted samples in submission order by tracking the next task index.
         accepted_buffer: dict[int, tuple[Image.Image, str]] = {}
         retry_rounds: dict[int, int] = {}
         next_to_save = 0

         while generated < num_samples:
            done, not_done = concurrent.futures.wait(
               futures,
               return_when=concurrent.futures.FIRST_COMPLETED,
            )
            futures = list(not_done)

            for fut in done:
               result_task_idx, result = fut.result()
               if result is None:
                  failures += 1
                  # Retry the same planned sample so the balanced label plan
                  # remains aligned with task indices.
                  retry_rounds[result_task_idx] = retry_rounds.get(result_task_idx, 0) + 1
                  futures.append(ex.submit(
                     _generate_one_until_ok,
                     result_task_idx,
                     retry_rounds[result_task_idx],
                  ))
               else:
                  accepted_buffer[result_task_idx] = result

               # Attempt to flush in-order accepted results.
               while next_to_save in accepted_buffer and generated < num_samples:
                  final_img, unicode_label = accepted_buffer.pop(next_to_save)
                  generated += 1
                  save_sample(
                     final_img,
                     unicode_label,
                     generated,
                     images_dir,
                     labels_fh,
                     grayscale=grayscale,
                  )

                  if generated == 1 or generated == num_samples or generated % progress_interval == 0:
                     percent = 100 * (generated / num_samples)
                     filled = int(50 * generated // num_samples)
                     bar = "#" * filled + "-" * (50 - filled)
                     sys.stdout.write(f"\rGenerating |{bar}| {percent:.1f}%")
                     sys.stdout.flush()

                  next_to_save += 1

               if failures > max(50, num_samples * 0.15):
                  raise RuntimeError(
                     f"Failure rate is too high ({failures} failures after "
                     f"{generated} successes). Check your assets."
                  )

               # Maintain in-flight window.
               while len(futures) < in_flight_target and task_counter < num_samples:
                  _submit_one()

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
         "Generates PNG images of random Baybayin text with Unicode labels."
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
      help=f"Probability [0.0-1.0] of using the Noto Tagalog font instead of SVGs (default: {FONT_RENDER_PROB}).",
   )
   p.add_argument(
      "--font-path",
      type=str, default=NOTO_TAGALOG_FONT_PATH, metavar="PATH",
      help=f"Path to the Noto Sans Tagalog TTF font file (default: '{NOTO_TAGALOG_FONT_PATH}')",
   )
   p.add_argument(
      "--augment-prob",
      type=float, default=AUGMENT_PROB, metavar="PROB",
      help=f"Probability [0.0-1.0] of applying augmentation (default: {AUGMENT_PROB}).",
   )
   p.add_argument(
      "--no-augment",
      dest="augment_prob", action="store_const", const=0.0,
      help="Disable augmentation for faster generation.",
   )
   p.add_argument(
      "--workers",
      type=int,
      default=None,
      metavar="N",
      help="Number of worker threads for image generation (default: CPU count).",
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
      output_dir      = args.output_dir,
      grayscale       = args.grayscale,
      seed            = args.seed,
      font_prob       = args.font_prob,
      font_path       = args.font_path,
      augment_prob    = args.augment_prob,
      workers         = args.workers,
   )

if __name__ == "__main__":
   main()
