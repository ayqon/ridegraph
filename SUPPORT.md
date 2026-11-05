# Support and governance

## Getting help

- **Questions and bug reports:** open an issue on GitHub. Issues are read
  and usually answered within a week.
- **How things work:** the [documentation](https://ridegraph.dev/docs/)
  explains each part of the system, its design decisions and what it does
  not model yet.

## What is supported

The latest release on the `main` branch, on Python 3.10, 3.12 and 3.13 and
current versions of Chrome, Firefox and Safari. Older releases do not get
fixes; please upgrade before reporting a problem.

Map data comes from OpenStreetMap and changes over time. A route that looks
wrong may be a mapping issue rather than a Ridegraph bug; if so, the fix
belongs in OpenStreetMap itself.

## Governance

Ridegraph is maintained by Ioannis Konstantinou
([ORCID 0009-0000-2007-0581](https://orcid.org/0009-0000-2007-0581)), who
reviews and merges changes and makes releases. Decisions about direction
are discussed in public issues, and anyone may propose a change there.
Contributors who make sustained, substantial contributions may be invited
to become maintainers.

## Releases

Releases follow semantic versioning and are listed in `CHANGELOG.md`. Each
release is archived on Zenodo with its own DOI; the DOI
[10.5281/zenodo.23966468](https://doi.org/10.5281/zenodo.23966468) always
points to the latest version.

## Security

If you find a security problem in the HTTP API or the Docker image, please
do not open a public issue. Use GitHub's private vulnerability reporting
on the repository (Security, then Report a vulnerability) instead.
