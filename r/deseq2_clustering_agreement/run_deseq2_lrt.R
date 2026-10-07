#!/usr/bin/env Rscript
# Per-feature differential expression with a DESeq2 likelihood ratio test.
#
# Usage:
#   uvr run run_deseq2_lrt.R -- --counts counts.csv --labels labels.csv \
#     --out results.csv [--alpha 0.05] [--z 1.96] [--nproc 8]
#
# Inputs:
#   counts.csv : samples x genes raw count matrix with a `sample_id` column.
#                Values must be raw non-negative integers (DESeq2 requirement).
#   labels.csv : `sample_id`, `label` columns assigning each sample to a
#                cluster. Samples present in only one file are ignored.
#   alpha      : FDR target given to DESeq2::results() for independent
#                filtering optimization.
#   z          : normal quantile of the lower confidence bound on the pairwise
#                log2 fold changes (default 1.96, a 95% one-sided bound).
#   nproc      : number of workers for the per-gene model fits (default: one
#                fewer than the detected cores, capped at 8). The fits
#                dominate the runtime, so this scales nearly linearly.
#
# Output (one row per gene, written as CSV):
#   feature : gene name.
#   p_value : LRT p-value for `~ label` vs `~ 1`; NaN where untestable.
#   p_adj   : Benjamini-Hochberg adjusted p-value from DESeq2::results().
#   min_gap : smallest pairwise effect size across every pair of label
#             levels: the lower confidence bound of the shrunken log2 fold
#             change, |lfc| - z * posterior SD, minimized over the pairs.
#             Large values mark global differentiators (every cluster pair
#             separated). NaN when no pair is testable.
#   max_gap : largest such bound over the pairs; large values mark local
#             differentiators (at least one pair separated).
#
# The effect size is the pairwise log2 fold change from ashr shrinkage, so
# the gaps live on the same log2 scale as the fold changes and a fixed
# threshold is meaningful. NaN is written literally as "NaN" so that float
# columns round-trip without being coerced to strings by readers.

suppressPackageStartupMessages(library(DESeq2))

usage <- paste(
  "usage: run_deseq2_lrt.R --counts <counts.csv> --labels <labels.csv>",
  "--out <results.csv> [--alpha <0.05>] [--z <1.96>] [--nproc <n>]"
)

parse_args <- function(args) {
  opts <- list(
    counts = NULL, labels = NULL, out = NULL,
    alpha = "0.05", z = "1.96", nproc = ""
  )
  i <- 1
  while (i <= length(args)) {
    key <- sub("^--", "", args[[i]])
    if (!key %in% names(opts)) {
      stop("unknown argument: ", args[[i]], "\n", usage, call. = FALSE)
    }
    if (i == length(args)) {
      stop("missing value for ", args[[i]], "\n", usage, call. = FALSE)
    }
    opts[[key]] <- args[[i + 1]]
    i <- i + 2
  }
  if (is.null(opts$counts) || is.null(opts$labels) || is.null(opts$out)) {
    stop(usage, call. = FALSE)
  }
  opts
}

# Worker count for the per-gene fits; empty means: one fewer than the
# detected cores, capped at 8 (each worker forks a copy of the count matrix).
resolve_nproc <- function(argument) {
  if (nzchar(argument)) {
    nproc <- suppressWarnings(as.integer(argument))
    if (is.na(nproc) || nproc < 1L) {
      stop("--nproc must be a positive integer, got ", argument, call. = FALSE)
    }
    return(nproc)
  }
  detected <- parallel::detectCores()
  if (is.na(detected)) {
    return(1L)
  }
  max(1L, min(detected - 1L, 8L))
}

args <- parse_args(commandArgs(trailingOnly = TRUE))

counts_df <- read.csv(args$counts, check.names = FALSE, stringsAsFactors = FALSE)
labels_df <- read.csv(args$labels, check.names = FALSE, stringsAsFactors = FALSE)
if (!"sample_id" %in% names(counts_df)) {
  stop("counts file must have a `sample_id` column", call. = FALSE)
}
if (!all(c("sample_id", "label") %in% names(labels_df))) {
  stop("labels file must have `sample_id` and `label` columns", call. = FALSE)
}

common <- intersect(
  as.character(counts_df$sample_id), as.character(labels_df$sample_id)
)
if (length(common) == 0) {
  stop("inner join of counts and labels produced no rows", call. = FALSE)
}

count_mat <- as.matrix(
  counts_df[match(common, as.character(counts_df$sample_id)), -1, drop = FALSE]
)
rownames(count_mat) <- common
if (!is.numeric(count_mat)) {
  stop("DESeq2 requires raw non-negative integer counts", call. = FALSE)
}
if (anyNA(count_mat) || any(count_mat < 0) || any(count_mat != round(count_mat))) {
  stop("DESeq2 requires raw non-negative integer counts", call. = FALSE)
}
storage.mode(count_mat) <- "integer"
# DESeq2 expects genes x samples; the input file is samples x genes.
count_mat <- t(count_mat)

label <- labels_df$label[match(common, as.character(labels_df$sample_id))]
groups <- factor(label)
if (nlevels(groups) < 2) {
  stop("need at least two label groups for a likelihood ratio test", call. = FALSE)
}

dds <- DESeqDataSetFromMatrix(
  countData = count_mat,
  colData = data.frame(label = groups, row.names = common),
  design = ~ label
)
# The per-gene dispersion and GLM fits dominate the runtime (67k features), so
# they run on `nproc` workers when more than one is available. Forked workers
# share no state, so results match a single-worker run up to floating-point
# summation order (observed: 3 of 67007 p-values differ by <1e-14). The caller
# must start R with single-threaded math libraries, because forking an R
# process whose BLAS/OpenMP thread pools are live deadlocks the children.
nproc <- resolve_nproc(args$nproc)
if (nproc > 1L) {
  backend <- if (.Platform$OS.type == "unix") {
    BiocParallel::MulticoreParam(workers = nproc, progressbar = FALSE)
  } else {
    BiocParallel::SnowParam(workers = nproc, progressbar = FALSE)
  }
  BiocParallel::register(backend)
  message("DESeq2: fitting genes on ", nproc, " workers")
}
dds <- DESeq(dds, test = "LRT", reduced = ~ 1, quiet = TRUE, parallel = nproc > 1L)
res <- results(dds, alpha = as.numeric(args$alpha))

# Effect size: for every pair of label levels, the shrunken pairwise log2
# fold change from ashr, as a lower confidence bound |lfc| - z * posterior SD.
# min_gap is the smallest bound over the pairs (a gene separated in every
# pair, i.e. a global differentiator); max_gap is the largest (a gene
# separated in at least one pair, i.e. a local differentiator).
z <- as.numeric(args$z)
if (is.na(z) || z < 0) {
  stop("--z must be a non-negative number, got ", args$z, call. = FALSE)
}
level_pairs <- combn(levels(groups), 2L)
lower_bounds <- vapply(
  seq_len(ncol(level_pairs)),
  function(i) {
    shrunk <- lfcShrink(
      dds,
      contrast = c("label", level_pairs[1L, i], level_pairs[2L, i]),
      type = "ashr",
      quiet = TRUE
    )
    abs(shrunk$log2FoldChange) - z * shrunk$lfcSE
  },
  numeric(nrow(count_mat))
)
if (is.null(dim(lower_bounds))) {
  lower_bounds <- matrix(lower_bounds, ncol = 1L)
}
testable <- rowSums(is.finite(lower_bounds)) > 0L
min_gap <- rep(NA_real_, nrow(lower_bounds))
max_gap <- rep(NA_real_, nrow(lower_bounds))
min_gap[testable] <- apply(
  lower_bounds[testable, , drop = FALSE], 1L, min, na.rm = TRUE
)
max_gap[testable] <- apply(
  lower_bounds[testable, , drop = FALSE], 1L, max, na.rm = TRUE
)

out <- data.frame(
  feature = rownames(count_mat),
  p_value = res$pvalue,
  p_adj = res$padj,
  min_gap = min_gap,
  max_gap = max_gap,
  stringsAsFactors = FALSE
)
write.csv(out, args$out, row.names = FALSE, na = "NaN")
