# CON²PHYS solution

Code that produces my answers to the [CON²PHYS](https://mchini.github.io/con2phys/index.html)
questionnaire: 18 mice, three anonymised brain areas, spikes and LFP.

## Run

```bash
conda env create -f environment.yml
conda activate con2phys
bash reproduce.sh              # or N_JOBS=8 bash reproduce.sh to run 8 animals in parallel
```

The data are expected in `data/<mouse>/` (one folder per animal, as distributed:
`spikes.mat`, `clusters.mat`, `brain_area.mat`, `lfp_1.mat`–`lfp_3.mat`,
`trial_data.xlsx`). With `N_JOBS=8` the full run took 61 min and at most 26 GB of
memory in total; with one job expect about 4 h and up to 7 GB.

## Results

Each question writes `outputs/<question>/final_report.json` (and a readable
`final_report.md`). These hold the answer, the mean and 95% CI per area (or pair or
segment), the units and the statistical tests. Figures go to `outputs/<question>/figures/`.

| Question | Script |
|---|---|
| Q1 lowest firing rate | `questions/q01_firing_rate.py` |
| Q2 highest 1–100 Hz LFP power | `questions/q02_lfp_power.py` |
| Q3 highest ripple density | `questions/q03_ripples.py` |
| Q4 strongest pairwise spike interactions (100 ms) | `questions/q04_spike_correlation.py` |
| Q5 strongest undirected functional connectivity | `questions/q05_undirected_fc.py` |
| Q6 strongest directed functional connectivity | `questions/q06_granger.py` |
| Q8 strongest theta phase locking | `questions/q08_theta_locking.py` |
| Q11 most information about variable A | `questions/q11_information.py` |
| Q12 segment where variable C is best decoded | `questions/q12_decoding.py` |
| Q13 highest dimensionality | `questions/q13_dimensionality.py` |
| Q15 most complex signal | `questions/q15_entropy.py` |

Questions 7, 9, 10 and 14 were not answered.

## How it is organised

- `questions/`: one script per question. It analyses one animal (`--animal N`),
  then combines all animals (`--aggregate`).
- `con2phys/`: shared code for data loading (`io.py`), statistics (`stats.py`),
  LFP processing (`signal.py`), spike-train helpers (`spikes.py`), plotting and the
  command line.
- `config.yaml`: every parameter.

All questions use the same group statistics. There is one value per mouse and
condition, and the group mean has a percentile-bootstrap 95% CI over mice. A Friedman
test is followed by Holm-corrected paired Wilcoxon tests. A condition is the answer
only if it differs from both others in the expected direction; otherwise the answer
is "Not enough data / no differences".
