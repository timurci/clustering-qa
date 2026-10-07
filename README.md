# clustering-qa

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 1.4.0`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Pipelines

* `clustering_stability`: clustering stability metrics per partition.
* `clustering_agreement`: per-cohort agreement between cluster labels and
  features, plus the report and the pairwise Jaccard summary. A DESeq2
  likelihood ratio test scores each gene per cohort; selection is a joint
  filter on the adjusted p-value and the pairwise fold-change gap (see
  `docs/clustering-agreement-context.md`).

Run one with `kedro run --pipelines=<name>`, or all of them with `kedro run`.
Every run writes under `data/08_reporting/clustering_agreement/`:

```
feature_significance/{pid}.csv           per-cohort test results (one row per gene)
clustering_agreement_features_global.json  consensus: every cluster pair separated in all cohorts
clustering_agreement_features_local.json   consensus: at least one pair separated in all cohorts
clustering_agreement_report.md           per-cohort counts, gap statistics, consensus sizes
jaccard_summary_report.md                pairwise cohort overlap of significant gene sets
jaccard_by_fdr.png                       per-FDR heatmaps: Jaccard index over raw intersection counts
jaccard_by_fdr_and_min_gap.png           the same heatmaps requiring the global gap (min_gap)
jaccard_by_fdr_and_max_gap.png           the same heatmaps requiring the local gap (max_gap)
```

> **Note:** these paths moved under `clustering_agreement/`. Outputs from
> earlier runs sit in the old location and are left untouched — delete
> `data/08_reporting/clustering_agreement_features.json`,
> `data/08_reporting/clustering_agreement_report.md` and
> `data/08_reporting/feature_significance/` once you have updated anything
> that reads them.

See `docs/clustering-agreement-context.md` for the method.

### DESeq2 R environment

A plain `kedro run` needs this setup, because the clustering agreement
pipeline shells out to `run_deseq2_lrt.R` in the standalone
[uvr](https://github.com/nbafrank/uvr) project `r/deseq2_clustering_agreement`
(R 4.5.3 + Bioconductor DESeq2 + CRAN ashr, pinned in `uvr.lock`). One-time
setup from the repository root:

```
cd r/deseq2_clustering_agreement && uvr sync
```

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/deploy/package_a_project/#package-an-entire-kedro-project)
