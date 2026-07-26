# Summary

<!-- What changes, and why. One paragraph is plenty. -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] New selector rule / chart type
- [ ] Refactor (no behaviour change)
- [ ] Docs / packaging / CI

## Checklist

- [ ] `make lint` passes (ruff check, ruff format, mypy on `plotaviz/core`)
- [ ] `make test` passes
- [ ] `plotaviz/core/` still contains **no Qt imports** — it stays headless
- [ ] Changes to preprocessing are modelled as replayable **steps**, not in-place mutation
- [ ] New chart capability goes through `ChartSpec`, not side-channel state
- [ ] Generated code still runs standalone (no `plotaviz` import) if `codegen.py` was touched
- [ ] `CHANGELOG.md` updated under Unreleased
- [ ] No credentials, keys, or private datasets in the diff

## Testing

<!-- How you verified this. For chart/selector changes, name the dataset shape you tested. -->

## Screenshots

<!-- Required for UI changes. Before/after if you're altering existing layout. -->
