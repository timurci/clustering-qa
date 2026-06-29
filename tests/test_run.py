"""This module contains example tests for a Kedro project.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py.
"""

from clustering_qa.pipeline_registry import register_pipelines


class TestKedroRun:
    """Smoke tests that exercise the Kedro run entry point."""

    def test_kedro_run_default_pipeline_is_not_empty(self):
        pipelines = register_pipelines()
        assert len(pipelines["__default__"].nodes) > 0
