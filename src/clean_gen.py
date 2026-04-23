import re
import sys

with open('generate.py', 'r', encoding='utf-8') as f:
   text = f.read()

# Replace the logging inside the loop with a progress bar
loop_search = """                if generated == 1 or generated % 500 == 0:
               log.info(
                  "  [%5d / %d]  word='%s'  label='%s'",
                  generated, num_samples, word, unicode_label,
               )"""

loop_replace = """                percent = 100 * (generated / num_samples)
            filled = int(50 * generated // num_samples)
            bar = '█' * filled + '-' * (50 - filled)
            sys.stdout.write(f'\\rGenerating |{bar}| {percent:.1f}%')
            sys.stdout.flush()"""

text = text.replace(loop_search, loop_replace)

# Add a print() after the while loop finishes
end_loop_search = """                if failures > max(50, num_samples * 0.15):
               raise RuntimeError(
                  f"Failure rate is too high ({failures} failures after "
                  f"{generated} successes). Check your assets."
               ) from exc

   log.info("""

end_loop_replace = """                if failures > max(50, num_samples * 0.15):
               raise RuntimeError(
                  f"Failure rate is too high ({failures} failures after "
                  f"{generated} successes). Check your assets."
               ) from exc
   print()

   log.info("""

text = text.replace(end_loop_search, end_loop_replace)

# Remove huge horizontal rules
text = re.sub(r'# [─]+[^\n]*\n', '', text)
text = re.sub(r'# ---[^\n]*\n', '', text)

# Remove docstrings
text = re.sub(r'^[ \t]*"""[\s\S]*?"""\n', '', text, flags=re.MULTILINE)

# Remove some specific comment noise
text = re.sub(r'[ \t]*# [^\n]*\n', '\n', text) # Remove all full line comments!

# Collapse multiple newlines into two max
text = re.sub(r'\n{3,}', '\n\n', text)

with open('generate.py', 'w', encoding='utf-8') as f:
   f.write(text)

