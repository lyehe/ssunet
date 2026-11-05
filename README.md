# SSUnet

SSUnet implements self-supervised photon prediction with a configurable 3D U-Net.
Training uses observed photons and their sampled splits; a clean image reference
is optional. The methods follow [Bit2Bit](https://arxiv.org/abs/2410.23247) for binary
photon detections and [GAP](https://arxiv.org/abs/2307.06607) for photon-count data.
Choose the loss and masking policy for the acquisition model: sums of binary
detections and correlated SPAD frames do not automatically satisfy Poisson assumptions.

Install the package and development tools, configure the data paths in a YAML
configuration, and run the pipeline:

```sh
python -m pip install -e ../torch-convnd
python -m pip install -e ".[dev]"
python run.py --config config.yml
python -m pytest tests
```

Partial convolutions and the 4D models use
[torch-convnd](https://github.com/lyehe/torch-convnd), which is not on PyPI. `uv sync`
installs it from a sibling checkout at `../torch-convnd`. Without one, install it with
`python -m pip install "torch-convnd @ git+https://github.com/lyehe/torch-convnd"`.

The pipeline tools need optional extras: `confocal` (LIF reading, dask and the napari
viewers), `neutrino` (plotly figures) and `spad` (visionsim colour conversion, the
napari viewer and Triton for `torch.compile` on Windows). `uv sync --all-extras`
installs them with the development tools; with pip, name them, as in
`python -m pip install -e ".[dev,confocal,neutrino,spad]"`. `uv sync` removes packages
that the lockfile does not list, so add `--inexact` when the environment also holds
another project's packages.

`forward(batch)` returns unrestricted photon-location logits in `(B, C, Z, H, W)`
layout. Use the explicit distribution API when normalized probabilities are needed:

```python
import torch

from ssunet.configs import ModelConfig
from ssunet.models import Bit2Bit

model = Bit2Bit(ModelConfig(architecture_version=2, partial_conv=False))
with torch.inference_mode():
    probabilities = model.predict_distribution(batch)
```

For `loss_function="photon"`, each sample sums to one over channels and spacetime.
For `"photon_2d"`, each temporal frame sums to one over channels and spatial pixels.
The inference tools separately multiply these distributions by the observed photon
count to produce expected-count images in the input's units. Such image exports
are distinct from normalized probabilities. Temporal offset averaging aligns each
prediction to its recorded frame before combining estimates. PNG color export in
the SPAD scripts additionally requires `visionsim`; the inference helpers do not.

The photon losses use stable log-probability reductions. Bit2Bit excludes input-hit
locations from both target scoring and the exponential sum in the normalization
offset: `sum(mask * exp(logits))`. An excluded logit contributes zero mass;
replacing it with zero would leave `exp(0)=1` in that sum. Fully masked samples
contribute zero loss, while inference uses the full live support. Corrected masked
loss/diagnostic values differ from earlier logits-zeroing implementations.
`val_loss` retains the training loss scale. Validation uses reproducible per-sample
photon splits without changing
the training RNG state. `val_photon_nll` is an event-normalized cross-entropy score;
`val_masked_photon_nll` names the corresponding Bit2Bit masked diagnostic and does
not claim independence for correlated data. These scores require no image reference.

Image PSNR/SSIM diagnostics are disabled by default. Enable
`MODEL.log_reference_metrics: true` only for a separately supplied, aligned reference
in matching intensity/color units. Integer references use their original dtype range;
normalized floating references are preserved. Floating references in other units
must declare `PATH.ground_truth_scale`. Pixel PSNR against individual photon events
does not establish the quality of the predicted distribution.

Architecture options use an explicit compatibility version:

- `MODEL.architecture_version: 1` is the default for historical and versionless
  checkpoints. It retains their ordinary convolutions and concatenation, even when
  the saved configuration contained previously ignored flags.
- `MODEL.architecture_version: 2` honors `partial_conv` and `merge_mode` for a new
  model. Checkpoints record these effective options and reject incompatible overrides.
  Partial convolutions use torch-convnd's `PartialConv3d` (NVIDIA's border rescaling).

The checkpoint loader handles both installed `ssunet` and historical `src.ssunet`
configuration paths. End-of-training saving is controlled by the training callbacks;
the model does not write an additional anonymous checkpoint. On Windows, generated
experiment names are shortened when necessary to leave room for TensorBoard files;
the full configuration remains in the experiment directory.

### Controlled neutrino photon reference

The experimental event-based loader preserves integer photon counts, fixed time
windows, empty observations and configuration-level train/validation/test splits.
It predicts a distribution over live sensors and time bins. Simulation truth is
saved separately and excluded from the model inputs and supervision. Its raw
count likelihood is separate from the historical Bit2Bit image loss.

From the repository root:

```sh
python -m src.tools.neutrino_reference audit data/neutrino/controlled_reference/poisson_audit.json
python -m src.tools.neutrino_reference analytic data/neutrino/controlled_reference/analytic_reference_1000.h5
python -m src.tools.neutrino_reference --help
```

Shard creation refuses to overwrite an existing file. The `import-prometheus`
command accepts a photon-level SQLite file and a matching geometry Parquet file;
Parquet import additionally requires `pyarrow`. The `profile` command runs a bounded
training smoke test on an explicitly selected device, with frozen validation and
an untouched test partition. Installed packages can use
`python -m tools.neutrino_reference` instead.

The initial physical sample uses the public GraphNeT/Prometheus P-ONE geometry in
water (60 sensors), while the exact Poisson toy uses the planned 256-sensor layout.
The Poisson statistical audit applies to the constructed reference; it does not
certify the selection or counting statistics of the curated physical simulation.
See [the plan](plans/neutrino_photon_reference.txt) and
[the execution record](plans/neutrino_photon_reference_status.txt).

The longer pilot keeps the same dense architecture and raw-count objective,
with fixed training budgets and frozen validation/control settings:

```sh
python -m src.tools.neutrino_pilot --config config/neutrino_pilot.yml --output data/neutrino/controlled_reference/longer_pilot_v1 --device cuda:0
```

The output directory must be new. It contains data/code fingerprints, model and
optimizer checkpoints, learning curves, per-event likelihoods and count-matched
donor controls. Model seeds vary while the validation photon splits stay fixed.
Final-step results and checkpoints selected on low-count validation are reported
separately. These pilot comparisons use validation only; the test partition is
reserved for the later fixed comparison. Installed packages can use
`python -m tools.neutrino_pilot`.
See [the longer-pilot results](plans/neutrino_longer_pilot.txt) for the fixed
three-seed comparison and its validation-only limitations.

[The phased study plan](plans/neutrino_phased_plan.txt) specifies the source
audit, bounded-memory execution, 100k-event cohort, locked evaluation and later
real-data work, with deliverables and exit criteria for each phase.

Export pilot score comparisons, event maps, simulator-truth context, analytic
expected-distribution GT and an offline interactive viewer with:

```sh
python -m src.tools.neutrino_visualize --output data/neutrino/controlled_reference/visualizations_reproduction
```

The interactive export additionally requires `plotly`. Physical-event maps show
held-out detections separately from probability predictions; clean expected-PDF
GT is available only in the constructed analytic reference. Truth is used only
for display, after inference, and stays outside training.

The phased study has now been implemented and evaluated. See the
[complete results and limitations](plans/neutrino_study_results.txt),
[follow-up protocol](plans/neutrino_followup_protocol.txt), and
[real marked-pulse contract](plans/neutrino_real_data_contract.txt).
The independent low-count test gives 4.093 versus 4.339 nats/photon for the
validation-selected models and simple control. The all-photon frozen control
and the separate analytic benchmark retain important negative comparisons.

The local offline viewers are
[main test results](data/neutrino/study/visualizations_test_v1/index.html),
[known analytic distribution GT](data/neutrino/analytic_study/visualizations_v2/index.html),
and [mechanism/real-data/probe results](data/neutrino/study/followup_visualizations/index.html).
PNG/SVG and ZIP exports accompany the viewers. To reproduce figures from the
completed cached studies, run these research commands from the repository root
with new output directories:

```sh
python -m src.tools.neutrino_study_visualize --partition test --output data/neutrino/study/test_figures_reproduction
python -m src.tools.neutrino_analytic_visualize --output data/neutrino/analytic_study/gt_figures_reproduction
python -m src.tools.neutrino_followup_visualize --output data/neutrino/study/followup_figures_reproduction
```

These figure commands preserve the sealed test decisions. The test viewer
requires completed test evidence and cannot open an unscored reserved test.

A separate [subthreshold detection protocol](plans/neutrino_detection_protocol.txt)
tests whether the frozen analytic-trained models distinguish faint injections
from background at a declared false-positive rate. It compares count, static
prior, geometry/time kernel and physics-template controls on the same recorded
photons, with independent background calibration and source-group uncertainty.
This exact Poisson toy does not establish real neutrino detection or isolate a
causal benefit of self-splitting. Run it once into a fresh output directory with:

```sh
python -m src.tools.neutrino_detection --config config/neutrino_detection.yml --device cuda:0
```

The command requires the completed analytic study and rejects an existing output
directory. For reproduction, copy the YAML and change its `output` path. The
[detection viewer](data/neutrino/detection_study_v1/index.html) includes efficiency,
false alarms, ROC curves and expected-distribution GT examples; its JSON/CSV
exports retain both recoveries and newly introduced misses.

The [frozen-feature detector probe](plans/neutrino_detection_probe_protocol.txt)
compares matched linear heads using raw/prior features, added self-splitting
features, or added untrained-network features. It fits only the small event-labelled
heads on fresh sources, then locks them and independent background calibration
before opening a new test. Encoder weights remain frozen; this is not fully
self-supervised detection or a real-detector validation.

```sh
python -m src.tools.neutrino_detection_probe --config config/neutrino_detection_probe.yml --device cuda:0
```

Its output directory must be new. The
[probe viewer](data/neutrino/detection_probe_v1/index.html) contains all controls,
conditional uncertainty, measured false alarms and validation-only selection.

The first probe does not establish added detection value: at the registered
two-photon/1% operating point, raw/prior detects 50.24%, added self-splitting
features 49.12%, and added untrained features 48.49%. See the
[full probe evaluation](plans/neutrino_detection_probe_results.txt), including
the secondary operating point, conditional uncertainty and next diagnostic.
Replay the completed study's saved scores, training scalers and feature cases:

```sh
python -m src.tools.neutrino_detection_probe_audit --study data/neutrino/detection_probe_v1 --device cuda:0
```

The [development diagnosis](plans/neutrino_development_diagnostics_results.txt)
uses a strict train/validation accessor to inspect selection, synthetic timing
retention and sensor-time pairing sensitivity. Learned PDF summaries retain
timing information, while the pooled hidden features are near the constant
timing baseline. This is development evidence; completed tests remain unchanged.

```sh
python -m src.tools.neutrino_development --config config/neutrino_development_diagnostics.yml --device cuda:0
```

The [diagnostic viewer](data/neutrino/development_diagnostics_v2/index.html)
includes the input-access log and numerical comparisons. The
[next fix plan](plans/neutrino_next_fix_plan.txt) first isolates PDF features in
the detector head, then considers dense context that retains the time axis.

For observation-only inspection, the
[3D data viewer](data/neutrino/data_3d_v1/index.html) shows real IceCube calibrated
pulses and simulated P-ONE photon detections. Rotate/zoom, choose a count example,
and play the original arrival times. It displays no model predictions or inferred
particle tracks. Export or refresh the self-contained viewer with:

```sh
python -m src.tools.neutrino_data_3d --refresh
```

The optional [native ConvNd addon](src/ssunet/ops/README.md) provides unrestricted
dense convolution over N spatial axes, including a `Conv4d` wrapper, with C++/CUDA
patch kernels, tiled workspace and first-order gradients. It is independent of
the existing models; the documentation includes build, validation and benchmark
commands and the current precision/performance limits.

The corrected neutrino 4D pipeline selects supervision from the acquisition format:
`observation_kind: one_bit` uses Bit2Bit; `observation_kind: photon_counts`
uses GAP. A sparse count batch containing only zeros and ones is still count
data. Conflicting method overrides and nonbinary Bit2Bit observations are
rejected. The P-ONE reference contains multi-photon bins and therefore uses GAP.
The [normalization audit](plans/photon_method_normalization_audit.txt) records
the pinned paper code and the difference from the archived count-weighted run.

The current [4D configuration](config/neutrino_gap4d_retry_v2.yml) uses raw-count
sine features, per-event unit-mean targets, and a joint probability distribution
over live sensor/time bins. It scores noisy held-out photons without clean
photon ground truth. See the [fixed retry protocol](plans/neutrino_gap4d_retry_v2_protocol.txt)
and [run instructions](plans/neutrino_gap4d_retry_v2_usage.txt).

```sh
python -m src.tools.neutrino_gap4d_retry --config config/neutrino_gap4d_retry_v2.yml --device cuda:0
python -m src.tools.neutrino_gap4d_report --study data/neutrino/gap4d_retry_v2 --device cuda:0
```

The report command requires completed training, replays all saved checkpoints,
checks their data/source identity and output normalization, and exports both
photon-weighted and equal-event comparisons.

The [published-method comparison](plans/neutrino_published_gap_protocol.txt)
uses the unchanged [official GAP network](https://github.com/krulllab/GAP)
at the paper's six-level, 28-channel configuration. It compares the original
Adam/batch-32 setup and a setup matched to our AdamW/batch-16 training, with
80,000 training examples per seed. Mapping detector observations to sensor/time
matrices, padding/cropping and fixed thinning are explicit domain adaptations.
This transfers the released method to P-ONE; it does not reproduce its microscopy
benchmark. See the [commands and source provenance](plans/neutrino_published_gap_usage.txt).

```sh
python -m src.tools.neutrino_published_gap --config config/neutrino_published_gap.yml --device cuda:0
python -m src.tools.neutrino_published_gap_report --study data/neutrino/published_gap_comparison_v1 --device cuda:0
```

The [traditional-method comparison](plans/neutrino_classical_protocol.txt) adds
Dirichlet Bayesian shrinkage, temporal and geometry-aware Gaussian KDE, and
Poisson-TV estimation. Priors are fitted on 640 training observations and
parameters selected on 160 separate training observations, with configuration
groups kept disjoint. Parameters are locked before the original validation
photons are opened; the selected prior is then refitted on all 800 training
observations. All estimators return normalized live sensor/time probabilities.

```sh
python -m src.tools.neutrino_classical --config config/neutrino_classical.yml --device cuda:0
python -m src.tools.neutrino_classical_report --study data/neutrino/classical_comparison_v1 --device cuda:0
```

The [results and limits](plans/neutrino_classical_results.txt) distinguish the
low-count neural advantage from the stronger pooled-photon point score of
temporal KDE. See the [run instructions](plans/neutrino_classical_usage.txt)
and [comparison viewer](data/neutrino/classical_comparison_v1/index.html).

The [confocal training test](plans/confocal_gap4d_batch4_usage.txt) provides a
four-level full XYZT U-Net (32/64/128/256 channels, 17,973,665 parameters) and
an XYZ-only control with a similar parameter budget. The
[default configuration](config/confocal_gap4d_features32_batch4.yml) uses one HyD photon-counting
channel, separate training/validation acquisitions, local count-preserving
caches, 16x16x128x128 TZYX patches, and the same GAP supervision for both modes.
Both use effective batch four: the 4D model has physical batch two and two
accumulation steps; the 3D control accumulates four single-patch microbatches.
Canonical example indices preserve identical photon draws across grouping.
This comparison uses [Barron et al.'s fixed log-softmax](https://jonbarron.info/softmaxfix/)
for its loss and scoring while retaining GAP's count supervision.
The [original small preset](config/confocal_gap4d_test.yml) remains available
through an explicit `--config` argument.

```sh
python -m src.tools.confocal_train --prepare-only
python -m src.tools.confocal_train --smoke-test
python -m src.tools.confocal_train --train
python -m src.tools.confocal_train --train --resume
python -m src.tools.confocal_report
python -m src.tools.confocal_evaluate
```

The smoke test takes three optimizer updates per model. The full pilot takes
500 updates per model and scores held-out photon-arrival distributions. It
does not require clean images. These commands use physical GPU 0; the usage
document describes data dependencies, safeguards and the acquisition holdout.
The checkpoint replay command requires completed training and generates
probability predictions, a source/data audit and a standalone local HTML report.

The completed [500-update comparison](plans/confocal_gap4d_batch4_results.txt)
scores 15.10873 nats/photon for 4D and 15.12148 for 3D (lower is better).
Both checkpoint scores replay exactly, and 53 relevant tests pass. Open the
[local report](data/confocal_gap4d_features32_batch4_v1/index.html) for curves,
observations and predicted probability maps. This is a single-seed pilot with
overlapping validation patches; the displayed predictions also show banding.

For [whole-acquisition reconstruction and napari playback](plans/confocal_full_inference_usage.txt):

```sh
python -m src.tools.confocal_napari
python -m src.tools.confocal_infer --precision fp16
python -m src.tools.confocal_napari --inference data/confocal_full_series013_v1
```

The full inference command processes all 1,024 time points and all 50 Z planes
of Series013 on the trained HyD4 channel. It blends overlapping 4D tiles,
averages complementary photon-half predictions, and saves normalized arrival
probabilities plus expected counts for display. A memory-mapped napari viewer
shows completed volumes as the run progresses. Inference can also use training
acquisitions; `--source train --output data/confocal_full_series001_v1` selects
the full training acquisition. See the usage document for resume, units and
precision checks.

The completed [whole-Series013 reconstruction](plans/confocal_full_series013_results.txt)
covers all 13.4 billion voxels. Its independent source/data/normalization audit
passes; the arrival-probability sum is 1.0000000391. See the
[first/middle/last examples](data/confocal_full_series013_v1/whole_acquisition_examples.png)
and [photon-count time course](data/confocal_full_series013_v1/photon_time_course.png).
Large-scale structures are visible, with residual smoothing and banding.

The [all-channel survey](plans/confocal_all_channels_survey_results.txt) checks
ten real 4D acquisitions and all 16 recorded detector channels. See the
[channel overview](data/confocal_all_channels_survey_v2/all_channels_overview.png),
[time courses](data/confocal_all_channels_survey_v2/all_channels_time_profiles.png)
and [Cell cluster early/middle/late views](data/confocal_all_channels_survey_v2/lif_00_00_detail.png).
The new joint model uses every retained detector as input context and predicts
a separately normalized TZYX photon distribution for each detector. Its
[two-channel configuration](config/confocal_all_channels_test.yml) passed a
three-update readiness check at physical batch two (effective four), using
18.79 GiB of allocated GPU memory. This is separate from the completed
HyD4-only checkpoint; joint-channel quality needs a new training run.

```sh
python -m src.tools.confocal_survey_napari
python -m src.tools.confocal_channels_test
python -m pytest -q tests/test_confocal_multichannel.py
```

The survey viewer displays all detectors side by side and offers acquisition,
time-range and raw-movie/noisy-mean selectors. It loads bounded local clips.

The [microtubule 3 trial](plans/confocal_microtubule3_joint4d_protocol.txt) trains
one joint-channel 4D model over all 1,024 times and 64 Z planes of the central
256x256 XY crop. A permanent 20% photon reserve provides evaluation targets
that never enter training; no separate acquisition or clean reference is
required. The fixed 500-update endpoint is compared with uniform and predeclared
3D/4D Gaussian controls, with detector and early/middle/late scores reported
separately. Training uses raw counts and per-detector Barron normalization.

```sh
python -m src.tools.confocal_joint_train --smoke-test
python -m src.tools.confocal_joint_train --train
python -m src.tools.confocal_joint_train --train --resume
python -m src.tools.confocal_joint_report
python -m src.tools.confocal_joint_visualize
python -m src.tools.confocal_joint_napari
```

The report command requires completed training, verifies data/source/checkpoint
identity, replays the reserved-photon likelihoods and exports comparisons.
The supplemental visualization shows all twelve fixed cases. The prediction
viewer shows both detectors with a shared count scale, patch selector and
movie/noisy-mean controls. HyD 2 is blue and HyD 4 is green in the viewers
and all-case visual report. Choose **Channels → Overlay channels** for additive
blue/green overlays; observations and predictions occupy separate panels.
**Separate channels** displays each detector individually.

The completed [microtubule 3 results](plans/confocal_microtubule3_joint4d_results.txt)
score **14.871 nats/photon**, versus 15.106 for the fixed 4D Gaussian control
and 15.249 for uniform. Both detectors improve over those controls overall.
The benefit is concentrated in early/middle data; late near-dark patches show
no clear gain over uniform. Broad structures remain visible, with strong
smoothing and limited fine detail. See the
[visual review of every scored patch](data/confocal_microtubule3_joint4d_v1/gallery.html)
and [learning curves](data/confocal_microtubule3_joint4d_v1/heldout_comparison.png).
The saved final checkpoint replays exactly. This is one acquisition and seed,
without a clean reference or evidence of newly detectable microtubules.

The [two-channel motion review](plans/confocal_twochannel_motion_results.txt)
applies that same checkpoint to 128-time clips from microtubule 3 and 4,
with real temporal halos and independent photon observations for comparison.
Microtubule 4 is a transfer test. Broad patterns change, but the current
network often attenuates the HyD4 bulk displacement estimates; individual
microtubule motion is not validated. See the
[movies and motion measurements](data/confocal_twochannel_motion_v1/index.html),
or open both recordings in one viewer:

```sh
python -m src.tools.confocal_motion_napari --clip 2
```

The [motion-control follow-up](plans/confocal_twochannel_motion_comparison_results.txt)
compares that checkpoint with fixed spatial and temporal Gaussian controls
using the same photon inputs, both detectors and reserved-photon likelihoods.
GAP scores slightly better on microtubule 3; the temporal Gaussian scores better
on microtubule 4 transfer and more closely preserves several HyD4 bulk shifts.
The checkpoint remains unvalidated for measuring individual microtubule motion.
See the [comparison movies and measurements](data/confocal_twochannel_motion_comparison_v1/index.html).
Run `python -m src.tools.confocal_motion_compare` to reproduce this follow-up.

The [capacity measurements](plans/confocal_gap4d_capacity.txt) also verify wider
4D models and a larger Z window. The optional
[32-channel preset](config/confocal_gap4d_features32.yml) provides a 17.97-million
parameter 4D model and a similarly sized 3D control; both fit on GPU 0 in FP32.
These sizing checks establish memory requirements, without claiming an improvement
in denoising quality.
