# ruff: noqa: E501
"""The notebook template that course authors copy (scripts/templates/00_template.ipynb)."""

from pathlib import Path

from .cells import (
    JOURNEY,
    code,
    exercise,
    learn,
    md,
    q,
    questions,
    recap,
    save,
    saw,
    step_intro,
    title,
    words,
)


def build():
    # =====================================================================
    # scripts/templates/00_template.ipynb  (for course authors, not learners)
    # =====================================================================
    save(
        "00_template.ipynb",
        [
            title("<NN_NN> <Topic name>", "section <x.y>"),
            md(
                "> **Where this fits**\n>\n> ```\n> "
                + JOURNEY
                + "\n> ```\n>\n> You are here: **<stage>**\n>\n> Project files this connects to:\n>\n> - `<src/homer_ai/...>`"
                + "\n>\n> **This notebook:** <one sentence: the problem this notebook solves>\n>\n> The big picture is in the [course map](../../notebooks/00_start_here/00_00_course_map.ipynb)."
            ),
            learn(["<one line per learning goal>"]),
            words(
                [
                    (
                        "<word>",
                        "<only words NEW to this notebook; older ones are linked automatically>",
                        "<tiny example>",
                    )
                ]
            ),
            md(
                "## 3. The idea\n\n*(Say it once: the opening box above already says where we are and what this notebook solves, "
                "via SOLVES in the build script. Don't repeat the big picture here; it lives in the course map.)*"
            ),
            q(
                "<the next question a beginner would ask>",
                "<a short answer with an example; one idea per question>",
            ),
            step_intro(),
            q(
                "<Step 1: a question, e.g. 'How do I ... in Python?'>",
                "<short explanation of the one new idea>\n\n👉 **Predict first:** <question>",
            ),
            code("# Step 1 code"),
            saw("<one or two lines explaining the result>"),
            *exercise(
                "<a question: one small thing the learner now does alone>",
                "<the task, on variables the notebook already has>\n\n👉 **Predict first:** <question>",
                starter="# 👉 do: <what to write, in one to three lines>\n# 💡 hint: <optional>\nmy_answer = None",
                solution="my_answer = 1 + 1",
                check='check(\n    "<what is checked>",\n    my_answer,\n    2,\n    coach="<the likely mistake, and what to look at>",\n)',
                hint="<a nudge towards the cell that shows how>",
                answer="<one or two lines on why>",
            ),
            md(
                "## 5. Full example\n\nAll the steps together, on the real Odyssey text."
            ),
            code("# Full example code"),
            *questions(
                [
                    (
                        "<A real learner question NOT already answered in sections 3-4>",
                        "<A short answer.>",
                    )
                ]
            ),
            recap(
                ["<at most 3 one-line takeaways>"],
                "<the next notebook>",
                can=[
                    "<2 to 4 things the learner can now do, each starting with a verb>"
                ],
            ),
        ],
        root=Path(__file__).parent.parent / "templates",
    )
