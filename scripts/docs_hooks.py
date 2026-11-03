"""MkDocs hook: fill the command-line page from the real argument parser,
so the reference can never drift from the code."""

from __future__ import annotations

import argparse

MARKER = "<!-- cli-reference -->"


def _reference() -> str:
    from fastroute.__main__ import build_parser

    parser = build_parser()
    out = [f"```text\n{parser.format_help()}```\n"]
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                out.append(f"## `{name}`\n\n```text\n{sub.format_help()}```\n")
    return "\n".join(out)


def on_page_markdown(markdown: str, page, config, files) -> str:
    if MARKER in markdown:
        return markdown.replace(MARKER, _reference())
    return markdown
