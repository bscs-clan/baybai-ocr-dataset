import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from generate import FOLDER_TO_UNICODE, transliterate_to_baybayin


SUPPORTED_LABEL_CHARS = {chr(codepoint) for codepoint in range(0x1700, 0x1715)}


def _chars_in_mapping() -> set[str]:
   return {char for label in FOLDER_TO_UNICODE.values() for char in label}


def test_folder_mapping_uses_supported_baybayin_unicode():
   assert _chars_in_mapping() == SUPPORTED_LABEL_CHARS
   assert FOLDER_TO_UNICODE["da"] == "\u1707"
   assert FOLDER_TO_UNICODE["ra"] == "\u170d"
   assert FOLDER_TO_UNICODE["r"] == "\u170d\u1714"


def test_asset_folders_match_unicode_mapping():
   asset_folders = {path.name for path in (ROOT / "baybayin").iterdir() if path.is_dir()}
   mapped_folders = set(FOLDER_TO_UNICODE)

   assert mapped_folders - asset_folders == set()
   assert asset_folders - mapped_folders == {"comma", "dot"}


def test_transliteration_preserves_ra_distinct_from_da():
   assert transliterate_to_baybayin("dara") == (
      ["da", "ra"],
      "\u1707\u170d",
   )
   assert transliterate_to_baybayin("araw") == (
      ["a", "ra", "w"],
      "\u1700\u170d\u170f\u1714",
   )


def test_unsupported_latin_letters_are_rejected_not_dropped():
   for word in ["casa", "filipino", "quiz", "zara"]:
      assert transliterate_to_baybayin(word) == ([], "")

