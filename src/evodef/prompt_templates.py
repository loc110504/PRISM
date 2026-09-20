"""Loads and renders the `prompts/*.txt` templates (04_PROMPT_ENGINEERING_SPEC.md).

Each template file has a `SYSTEM` block and a `USER` block separated by a
line containing exactly `USER`. Placeholders are `{{name}}` and are
substituted with plain string replacement (no Jinja dependency needed for
this small, fixed set of templates).

Prompt files are frozen artifacts: `03_run_evolution.py`/`05_run_test.py`
record `sha256(system + user_template)` per template so every run logs which
exact prompt version produced its predictions (03_IMPLEMENTATION_SPEC.md #7).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .utils import sha256_text


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    system: str
    user_template: str

    @property
    def hash(self) -> str:
        return sha256_text(self.system + "\n---\n" + self.user_template)

    def render_user(self, **kwargs: str) -> str:
        text = self.user_template
        for key, value in kwargs.items():
            text = text.replace("{{" + key + "}}", str(value))
        return text


def load_prompt_template(path: str | Path) -> PromptTemplate:
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if "\nUSER\n" not in raw:
        raise ValueError(f"Prompt template {path} is missing a 'USER' section marker")
    system_part, user_part = raw.split("\nUSER\n", 1)
    system_part = system_part.removeprefix("SYSTEM\n").strip()
    return PromptTemplate(name=path.stem, system=system_part, user_template=user_part.strip())


def load_prompt_dir(prompts_dir: str | Path) -> dict[str, PromptTemplate]:
    prompts_dir = Path(prompts_dir)
    templates = {}
    for path in sorted(prompts_dir.glob("*.txt")):
        templates[path.stem] = load_prompt_template(path)
    return templates
