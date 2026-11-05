# Contributing to Ridegraph

Thanks for taking an interest. Bug reports, questions, small fixes and new
cities are all welcome. This page explains how to get set up and what a
change needs before it can be merged.

## Reporting a problem

Open an issue on GitHub. The most useful reports say:

- what you ran (the command, or what you clicked on the map),
- what you expected, and what happened instead,
- the city, and the two points or the stops involved if it is about a route.

If a route looks wrong, a link to the same trip on another map, or the OSM
way ids involved, makes it much quicker to track down.

## Setting up

You need Python 3.10 or newer, and Node.js 18 or newer if you touch the map.

```bash
git clone https://github.com/ayqon/ridegraph
cd ridegraph
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ml]"
pytest
```

The tests build small synthetic road networks, so they do not need any map
data. To work on a real city:

```bash
python -m ridegraph fetch --city london
python -m ridegraph build --city london
python -m ridegraph export-web --city london
python -m http.server -d web 8000
```

## Making a change

1. Open an issue first for anything larger than a small fix, so we can agree
   on the approach before you spend time on it.
2. Work on a branch and keep each commit to one change.
3. Add or update tests. Every search method is checked against Dijkstra; new
   routing behaviour should be too. If you change the map's JavaScript
   engine, the Node tests in `web/test/` must still agree with Python.
4. Run the same checks CI runs:

   ```bash
   ruff check . && ruff format --check .
   mypy ridegraph
   pytest
   ```

5. Add a line to the `Unreleased` section of `CHANGELOG.md`.
6. Open a pull request describing what changed and why.

## Adding a city

A city is one entry in `ridegraph/cities.py`: its bounding box, a few
landmarks people would recognise, where ride demand concentrates, and which
side of the road traffic drives on. Keep the area small enough that the map
loads quickly in a browser; a few kilometres across is plenty.

## Style

- Plain, readable Python with type hints; numpy where it matters for speed.
- Docstrings say what a function is for, not how it does it.
- Comments explain why, not what.
- Plain English in docs and messages, British spelling.

## Use of AI tools

You may use AI tools when contributing, but you are responsible for every
line you submit: you must understand it, have tested it, and be able to
explain it in review. Please say in the pull request if a substantial part
was generated.

## Code of conduct

Be respectful and constructive. Disagree with ideas, not people.
Harassment of any kind is not tolerated and will lead to removal from the
project.
