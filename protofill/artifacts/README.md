# Released result summaries

`final_config.json` records the validation-locked ProtoFill configuration.
`results_summary.csv` is a compact, human-readable extract of matched-run
results used in the manuscript.

The BioDCASE rows are derived from:

```text
analysis_output/paper_experiments/inference_method_matrix/results.json
```

The CREMA-D rows are derived from:

```text
analysis_output/paper_experiments/protofill_final_cremad/results.json
```

The full files are retained locally as research evidence but are not included
in the compact Git release because the paper experiment directory is close to
1 GB and contains machine-specific checkpoint paths. The released CSV keeps
unrounded means, paired gains, run identifiers, and the number of positive
paired runs. It does not replace the full per-sample audit archive.
