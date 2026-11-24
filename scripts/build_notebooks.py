"""Build every course notebook and chapter README in notebooks/.

Run from the project folder:

    uv run python scripts/build_notebooks.py

The content lives in the scripts/course/ package, one module per chapter
(ch00_start_here.py ... ch10_your_domain.py, appendix_a_toolbox.py). This file only
calls them, in order. It rewrites every notebook and every chapter README, so
edit the content there, then run this.

Optional: pass a folder to build into it instead of notebooks/.
"""

from course import (
    appendix_a_toolbox,
    ch00_start_here,
    ch01_foundations,
    ch02_text_to_numbers,
    ch03_attention,
    ch04_building_the_model,
    ch05_training,
    ch06_making_it_talk,
    ch07_iliad,
    ch08_fine_tuning,
    ch09_grounded_answers,
    ch10_your_domain,
    readmes,
    template,
)

# The order matters: a word is explained only in the first notebook that uses it.
STEPS = [
    template,
    ch00_start_here,
    ch01_foundations,
    ch02_text_to_numbers,
    ch03_attention,
    ch04_building_the_model,
    ch05_training,
    ch06_making_it_talk,
    appendix_a_toolbox,
    ch07_iliad,
    ch08_fine_tuning,
    ch09_grounded_answers,
    ch10_your_domain,
    readmes,
]

for step in STEPS:
    step.build()

print("built")
