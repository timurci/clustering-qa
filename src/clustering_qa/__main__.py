"""clustering-qa file for ensuring the package is executable.

Allows the project to be run as `clustering-qa` and `python -m clustering_qa`.
"""

import sys
from pathlib import Path
from typing import Any

from kedro.framework.cli.utils import find_run_command
from kedro.framework.project import configure_project


def main(*args, **kwargs) -> Any:  # noqa: ANN002,ANN003,ANN401 - passthrough to Click's `find_run_command`; signature must stay dynamic to forward arbitrary CLI options
    """Entry point that defers to Kedro's CLI run command.

    The signature is intentionally untyped: it forwards every argument
    Click receives (positional or keyword) to the resolved run command.
    """
    package_name = Path(__file__).parent.name
    configure_project(package_name)

    interactive = hasattr(sys, "ps1")
    kwargs["standalone_mode"] = not interactive

    run = find_run_command(package_name)
    return run(*args, **kwargs)


if __name__ == "__main__":
    main()
