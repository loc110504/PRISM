"""Parses SARA's `statutes/source/sectionNNN` files into subsection-level
`StatuteChunk`s (02_DATA_AND_SPLITS_SPEC.md #5).

Observed SARA statute format (verified against the official archive,
`data/raw/sara_extracted/sara/statutes/source/*`):

    §151. Allowance of deductions for personal exemptions

    (a) Allowance of deductions

    In the case of an individual, the exemptions ...

    (d) Exemption amount

    For purposes of this section-

        (1) In general

        Except as otherwise provided ...

            (A) In general

            ...

A labeled unit `(x) ...` starts a new node at a depth given by its
indentation (4 spaces per nesting level: (a)=0, (1)=1, (A)=2, (i)=3, (I)=4).
An unlabeled paragraph is appended to the text of whichever node is
currently deepest open on the stack (or treated as the whole section's own
text when there are no headers at all, e.g. `section3301`).

Every node - whether or not it has children - becomes its own retrievable
`StatuteChunk`, holding only its own text (not its descendants'), so
retrieval can find the specific subsection a case turns on
(01_METHOD_SPEC.md #2.1: "Chunk at subsection level whenever possible.").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..formalization.validators import find_referenced_sections
from ..schemas import StatuteChunk

_TITLE_RE = re.compile(r"^§\s*(\d+[A-Za-z]?)\.\s*(.*)$")
_HEADER_RE = re.compile(r"^(?P<indent>[ ]*)\((?P<label>[A-Za-z0-9]+)\)(?:\s+(?P<rest>.*))?$")
_SUBSECTION_SELF_REF_RE = re.compile(
    r"\b(?:subsection|paragraph|subparagraph|clause|subclause)\s+\(([A-Za-z0-9]+)\)",
    re.IGNORECASE,
)


@dataclass
class _Node:
    label: str
    depth: int
    parent: "_Node | None"
    text_lines: list[str] = field(default_factory=list)
    children: list["_Node"] = field(default_factory=list)
    char_start: int = 0
    char_end: int = 0

    @property
    def section_id_path(self) -> list[str]:
        path = []
        node: "_Node | None" = self
        while node is not None and node.label != "__root__":
            path.append(node.label)
            node = node.parent
        return list(reversed(path))


def _iter_lines_with_offsets(body: str) -> list[tuple[int, str]]:
    """Returns (char_start_of_line, line_text_without_newline) for every line."""
    result = []
    offset = 0
    for line in body.splitlines(keepends=True):
        result.append((offset, line.rstrip("\n")))
        offset += len(line)
    return result


def parse_statute_file(path: str | Path) -> tuple[str, str, list[StatuteChunk]]:
    """Returns (section_number, title, chunks) for one `statutes/source/sectionNNN` file.

    Parses line-by-line (not paragraph-by-paragraph): SARA keeps each
    subsection/list-item on its own physical line without necessarily
    separating siblings by a blank line (e.g. section1(a)'s rate brackets
    `(i)`..`(v)` are consecutive lines with no blank line between them), so
    every line - not just the first line of a blank-line-delimited block -
    must be checked for a new header.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    title_match = _TITLE_RE.match(lines[0].strip()) if lines else None
    section_number = title_match.group(1) if title_match else path.stem.replace("section", "")
    title = title_match.group(2) if title_match else lines[0].strip() if lines else ""

    body_start = raw.index("\n", raw.index(lines[0])) if lines else 0
    body = raw[body_start:]

    root = _Node(label="__root__", depth=-1, parent=None)
    stack: list[_Node] = [root]
    nodes_in_order: list[_Node] = []

    for start, line in _iter_lines_with_offsets(body):
        if not line.strip():
            continue
        header_match = _HEADER_RE.match(line)
        if header_match:
            indent = len(header_match.group("indent"))
            depth = indent // 4
            label = header_match.group("label")
            rest = header_match.group("rest") or ""

            while stack and stack[-1].depth >= depth:
                stack.pop()
            if not stack:
                stack = [root]
            parent = stack[-1]

            node = _Node(label=label, depth=depth, parent=parent, char_start=start)
            if rest.strip():
                node.text_lines.append(rest.strip())
            parent.children.append(node)
            stack.append(node)
            nodes_in_order.append(node)
        else:
            target = stack[-1] if stack else root
            target.text_lines.append(line.strip())

    chunks: list[StatuteChunk] = []

    def _finalize(node: _Node, char_end: int) -> None:
        node.char_end = char_end

    if root.children:
        for i, node in enumerate(nodes_in_order):
            next_start = nodes_in_order[i + 1].char_start if i + 1 < len(nodes_in_order) else len(body)
            _finalize(node, body_start + next_start)
    else:
        root.char_end = len(raw)

    def _emit(node: _Node, section_path: list[str]) -> None:
        text = "\n\n".join(node.text_lines).strip()
        section_id = section_number + "".join(f"({p})" for p in section_path)
        chunk_id = "sec_" + re.sub(r"[^0-9A-Za-z]+", "_", section_id).strip("_")
        parent_id = None
        if node.parent is not None and node.parent.label != "__root__":
            parent_path = section_path[:-1]
            parent_section_id = section_number + "".join(f"({p})" for p in parent_path)
            parent_id = "sec_" + re.sub(r"[^0-9A-Za-z]+", "_", parent_section_id).strip("_")
        references = find_referenced_sections(text)
        for m in _SUBSECTION_SELF_REF_RE.finditer(text):
            references.append(f"{section_number}({m.group(1)})")
        chunks.append(
            StatuteChunk(
                chunk_id=chunk_id,
                section_id=section_id,
                text=text if text else title,
                parent_id=parent_id,
                references=sorted(set(references)),
                source_file=str(path.name),
                char_start=body_start + node.char_start,
                char_end=node.char_end,
            )
        )
        for child in node.children:
            _emit(child, section_path + [child.label])

    if root.children:
        for child in root.children:
            _emit(child, [child.label])
    else:
        # No subsections at all: the whole section is one chunk (e.g. section3301).
        text = "\n\n".join(root.text_lines).strip() or title
        references = sorted(set(find_referenced_sections(text)))
        chunks.append(
            StatuteChunk(
                chunk_id=f"sec_{section_number}",
                section_id=section_number,
                text=text,
                parent_id=None,
                references=references,
                source_file=str(path.name),
                char_start=body_start,
                char_end=len(raw),
            )
        )

    return section_number, title, chunks


def build_corpus(statutes_dir: str | Path) -> list[StatuteChunk]:
    statutes_dir = Path(statutes_dir)
    all_chunks: list[StatuteChunk] = []
    for path in sorted(statutes_dir.glob("section*")):
        if path.is_file():
            _, _, chunks = parse_statute_file(path)
            all_chunks.extend(chunks)
    return all_chunks
