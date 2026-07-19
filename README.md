# CARE-MM

CARE-MM is a tri-modal clinical decision-support training and evaluation package for gastric-cancer diagnosis and management triage across tertiary, provincial, and community care settings. It combines frozen endoscopy, pathology, and structured EHR encoders with a 512-dimensional, four-layer, eight-head fusion network. Missing inputs are handled by masked-softmax renormalization over modalities that are present. Training draws availability masks from the empirical distribution within each care setting. Temperature calibration, mask-conditional conformal sets, abstention, and expected-cost routing bind diagnostic probabilities to biopsy, endoscopic resection, or surgical referral.

## Scope

The package contains the training, calibration, evaluation, statistical analysis, data preparation, and audit components needed for the reported prospective and retrospective analyses. Raw clinical images and records are private and are not included. The public datasets listed in `datasets.txt` support encoder refinement and domain preparation only.

## Environment

Python 3.10, PyTorch 2.3.1, and CUDA 12.1 are pinned. Create an environment with either:

```bash
conda env create -f environment.yml
conda activate care-mm
```

or:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The container image can be built with:

```bash
docker build -t care-mm:1.0 .
```

## Data layout

Each row of the cohort CSV represents one patient and contains:

```text
case_id,site,region,setting,label,triage,endoscopy_path,pathology_path,ehr_path
```

`label` is binary. `triage` uses `0` for biopsy, `1` for endoscopic resection, and `2` for surgical referral. A missing modality path is empty. At least one modality must be available. Splits are made at patient level.

Endoscopy and pathology paths point to tensors shaped `[frames, feature_width]` and `[tiles, feature_width]`. EHR paths point to tensor dictionaries containing `numerical`, `categorical`, and `missing`. Encoder features should be extracted with the study backbone families: Endo-FM for endoscopy and CONCH, Virchow2, or UNI2 for pathology. Backbone weights are obtained from their respective maintainers under their terms.

Build the empirical care-setting availability table with:

```bash
care-mm-prepare --manifest data/cohort.csv --availability-output data/availability.csv
```

The private clinical cohort contains 9,529 patients: 2,098 in the prospective arm and 7,431 in the retrospective support arm. It cannot be distributed. Public data access points and licenses are collected in `datasets.txt`. Verify local downloads against a locally generated SHA-256 manifest before feature extraction.

## Configuration

`configuration/main.yaml` preserves every reported primary value:

- fusion width 512
- four fusion layers
- eight attention heads
- AdamW
- learning rate `1e-4`
- 50 epochs
- conformal miscoverage `0.10`
- false-negative diagnosis cost ten times the false-positive cost

Batch size, weight decay, warmup, precision, and gradient clipping were not reported. They remain `null`; training refuses to start until they are explicitly supplied. This keeps local runtime choices separate from reported methodology.

## Training

After filling the unreported runtime fields in a local configuration copy, launch one process per GPU with `torchrun`:

```bash
torchrun --standalone --nproc-per-node=4 -m care_mm.commands.train \
  --config configuration/local.yaml \
  --endoscopy-width 768 \
  --pathology-width 768 \
  --ehr-numerical 32 \
  --ehr-cardinalities 4 8 5 3
```

The modality encoders are frozen and the fusion head is optimized for 50 epochs. Availability masks are sampled from `π(setting) P(mask|setting)`. Random state, seed, optimizer state, scaler state, configuration identity, and cohort-manifest digest are written atomically at each epoch.

The paper does not report GPU model, GPU count, VRAM, storage, batch size, precision, or wall-clock time. No hardware claim can therefore be made from the manuscript. A local budget should be measured after fixing the encoder feature widths and batch size.

## Evaluation

Evaluation accepts an `.npz` archive containing patient-level arrays named `case_id`, `label`, `score`, `site`, `setting`, and `mask`. Optional `baseline_score` enables the paired DeLong comparison.

```bash
care-mm-evaluate \
  --predictions results/prospective_predictions.npz \
  --output results/primary \
  --resamples 10000 \
  --seed 2026
```

The evaluator writes CSV and JSON results for AUROC, sensitivity, specificity, calibration bins, Brier score, site and setting subgroups, maximum subgroup gaps, paired DeLong testing, and Holm adjustment. Confidence intervals use 10,000 patient-level bootstrap resamples.

Reference values for release verification are:

| Analysis | Value |
|---|---:|
| Prospective diagnostic AUROC | 0.938 |
| Prospective AUROC 95% CI | 0.923–0.951 |
| Sensitivity | 0.913 |
| Specificity | 0.902 |
| Overall triage concordance | 0.883 |
| Early/operable triage concordance | 0.941 |
| Expected calibration error | 0.024 |
| Brier score | 0.082 |
| Conformal target coverage | 0.90 |
| Abstention rate | 0.100 |

The community endoscopy-only analysis uses 790 patients. CARE-MM reaches AUROC 0.872 and ECE 0.031; zero imputation reaches AUROC 0.806 and ECE 0.071. Component removals are represented by the imputation and fusion ablation modules.

## Statistical analysis

`care_mm.analysis` provides patient bootstrap intervals, paired DeLong covariance, Holm adjustment, subgroup estimates, cross-vendor shifts, selective risk-coverage curves, safety and harm summaries, marginal gap-closed estimates, permutation tests, and inverse-probability, self-normalized, direct, and doubly robust policy estimates.

The deployment audit summarizes turnaround, view rate, recommendation concordance, abstention, and mask-specific coverage tripwires. A tripwire activates when observed miscoverage exceeds the target by more than two standard errors.

## Clinical boundary

This research package does not provide autonomous clinical care. Recommendations are non-binding and must remain in a passive workflow until local validation, governance, security review, and regulatory requirements are satisfied. Ambiguous conformal sets are deferred to a clinician or multidisciplinary team.

