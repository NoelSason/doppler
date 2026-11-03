# The virtualenv lives outside ~/Desktop: macOS keeps setting the "hidden"
# flag on dotfolders there, and Python 3.12+ skips hidden .pth files, which
# silently breaks the editable install.
export UV_PROJECT_ENVIRONMENT := env_var("HOME") / ".venvs" / "dopplergeo"

default:
    @just --list

# Install the package with the plotting and DEM extras
sync:
    uv sync --extra plots --extra dem

# Run the test suite
test *args:
    uv run --no-sync pytest -q {{args}}

# Synthetic end-to-end run with figures
demo out="demo" *args:
    uv run --no-sync dopplergeo demo --out {{out}} {{args}}
