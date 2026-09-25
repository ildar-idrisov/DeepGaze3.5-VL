# [ECCV 2026] InternVL3.5-8B Combined Scanpath Model

📄 **Paper:** https://arxiv.org/abs/2607.02083

A rank-32 LoRA adapter on top of `OpenGVLab/InternVL3_5-8B-HF`, fine-tuned to
predict human free-viewing scanpaths as coordinate sequences on a 100x100 grid.
Given an image, the model emits a sequence of fixation coordinates that
approximate where a human observer would look. This is the **combined** model:
it was trained jointly on five eye-tracking datasets (MIT, CAT, COCO, Daemons,
Figrim). The weights bundled here are the best checkpoint.

## Models

Two LoRA adapters are bundled, both on top of `OpenGVLab/InternVL3_5-8B-HF`:

- **`model/combined_adapter/`** — free-viewing scanpath prediction (rank 32),
  trained jointly on MIT, CAT, COCO, Daemons and Figrim. Used by `run_eval.sh`.
- **`model/visual_search_adapter/`** — goal-directed **visual search** (rank 8),
  trained on COCO-Search18 (target-present and target-absent trials); given a
  search target it predicts the search scanpath.

Choose which one to load with `--adapter-path`. The bundled 5-image sample and
`run_eval.sh` target the free-viewing model; the visual-search model expects
COCO-Search18-style inputs (a target category in the prompt), which are not
bundled here.

## Directory layout

```
internvl3_5_8b_combined_release/
├── README.md
├── LICENSE
├── requirements.txt
├── run_eval.sh
├── predict_scanpath.py                    # image -> scanpath inference
├── evaluate_vllm_unified.py               # evaluation / scoring script
├── tests/                                 # unit tests (pytest)
├── configs/
│   ├── internvl3_5_8b_combined.yaml        # free-viewing LoRA SFT config
│   └── internvl3_5_8b_visual_search.yaml   # visual-search LoRA SFT config
├── model/
│   ├── combined_adapter/          # free-viewing scanpath (rank 32)
│   └── visual_search_adapter/     # COCO-Search18 visual search (rank 8)
└── data/
    ├── sample_MIT.json            # 75 entries for the 5 sample MIT images
    ├── images/                    # MIT_0985.jpg .. MIT_0989.jpg
    └── centerbias/
        └── MIT/                   # per-image center-bias priors (IG baseline)
```

## Quick start

1. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the evaluation (needs 1 GPU). The first run downloads the base model
   from HuggingFace and merges the LoRA adapter into
   `model/combined_adapter_merged/`:

   ```bash
   bash run_eval.sh
   ```

   Results are written to `eval_output/`.

## Tests

`python -m pytest tests` checks the prompt format, center-bias loading and metric
aggregation (requires pytest; no GPU or model download).

## Predict a scanpath

`predict_scanpath.py` runs a single image through the model and prints the
predicted scanpath — no ground truth or metrics needed.

Free viewing (8 fixations by default, uses `model/combined_adapter`):

```bash
python predict_scanpath.py --image path/to/image.jpg
```

Visual search (3 fixations by default, uses `model/visual_search_adapter`) —
pass the target object:

```bash
python predict_scanpath.py --image path/to/office.jpg --mode search --target laptop
```

The scanpath is printed both on the model's 0–100 grid and in pixel
coordinates. Use `--num-fixations N` to change the length, `--output out.json`
to save the result, and `--save-overlay overlay.png` to render the scanpath on
the image.

The scanpath is **sampled** from the model (`--temperature 1.0` by default, fixed
`--seed`), i.e. it is one draw from the predicted distribution over human
scanpaths; run several seeds to see the variability. `--temperature 0` gives the
deterministic greedy path, which is not a typical human scanpath. In the
free-viewing training data the first fixation is the initial central fixation of
the eye-tracking setup, so the first predicted point is (close to) the image
center. The visual-search model was trained on the 18 COCO-Search18 targets:
bottle, bowl, car, chair, clock, cup, fork, keyboard, knife, laptop, microwave,
mouse, oven, potted plant, sink, stop sign, toilet, tv.

## Running evaluations

`run_eval.sh` is a thin wrapper around `evaluate_vllm_unified.py`. To customise
a run, call the script directly:

```bash
python evaluate_vllm_unified.py \
    --base-model OpenGVLab/InternVL3_5-8B-HF \
    --adapter-path model/combined_adapter \
    --val-json data/sample_MIT.json \
    --images-dir data \
    --pkl-dir data/centerbias \
    --output-dir eval_output \
    --metric-mode fast \
    --batch-size 64 --max-num-seqs 32 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --skip-viz
```

On the first run the base model is downloaded from HuggingFace and the LoRA
adapter is merged into `model/combined_adapter_merged/` (reused on later runs).
The merge is done in bfloat16, the dtype the adapters were trained in, and vLLM
runs the merged model in bfloat16. On GPUs without bf16 support (before Ampere)
pass `--dtype half`. Merged directories created by earlier versions of this
script are float16; the script refuses to reuse them — delete them to re-merge.
A GPU is required.

### Metric modes

Two scoring modes are available via `--metric-mode`:

- **`fast`** (default) — scores the ground-truth coordinate of each fixation by
  probing its digits with per-digit normalisation. Reports per-fixation
  Information Gain (IG) and log-likelihood (LL). Recommended.
- **`grid`** — builds the full 100×100 next-fixation probability grid for every
  transition and normalises over it. Slower (many more forward passes) but also
  yields AUC and NSS alongside IG/LL, and can dump the grids with `--save-grids`.

Both modes probe the fixation coordinates digit-by-digit; the digit distribution
is renormalised over the ten digit tokens (0–9) so that probability mass on
non-digit tokens does not distort the score.

Every metric is reported as three averages: **per fixation** (every scored
fixation weighted equally; the pysaliency / DeepGaze convention), **per image**
and **per scanpath** (mean of per-scanpath means, the only average earlier
versions reported; a scanpath with one transition weighs as much as one with
twelve). The summary is also stored under `summary` in the results JSON. IG is
in bits per fixation relative to the center bias, LL is the natural-log
probability of the ground-truth cell of the 100×100 grid. Both are computed on
that grid, not per pixel, so they are not directly comparable to pixel-level
numbers of models such as DeepGaze III.

### Protocol notes

- **Prompt format.** Prompts reproduce the LlamaFactory `intern_vl` template the
  adapters were trained with, including its default system prompt and the image
  placeholder directly followed by the text. Earlier versions built prompts with
  the Hugging Face chat template, which has no system prompt and adds a newline
  after the image, so the model was evaluated on prompts it had not been trained
  on. Only InternVL base models are supported.
- **Scanpath length in the prompt.** The prompt asks for "exactly N fixation
  points", where N is the length of the ground-truth scanpath, because the model
  was trained this way. The scores are therefore conditioned on the true number of
  fixations, which models such as DeepGaze III do not get. Removing this requires
  retraining without N in the prompt.
- **First fixation.** The first ground-truth fixation (the initial central
  fixation) is only used as history and is not scored.
- **Checkpoint selection.** The bundled checkpoints were selected by eval loss on
  the `scanpath_val_*` splits (see `configs/`). Report results on data that was
  not used for this selection.
- **Failures are not skipped.** A missing image, center-bias prior or any other
  error stops the run. Results are saved after every image, so a run can be
  continued with `--resume-json`.

### Key options

| Flag | Meaning |
|------|---------|
| `--metric-mode {fast,grid}` | Scoring mode (default `fast`). |
| `--base-model` | HuggingFace base model (`OpenGVLab/InternVL3_5-8B-HF`). |
| `--adapter-path` | LoRA adapter directory (merged into `*_merged/` on first use). |
| `--val-json` | Evaluation set in LlamaFactory format (see below). |
| `--images-dir` | Base directory that image paths in the JSON resolve against (required). |
| `--pkl-dir` | Center-bias priors (the IG baseline); every image needs one. Without `--pkl-dir` a synthetic Gaussian center bias is used for all images. |
| `--dtype` | vLLM dtype (default `auto` = bf16; `half` for GPUs without bf16). |
| `--output-dir` | Where the results JSON is written. |
| `--max-samples N` | Evaluate only the first N entries (quick checks). |
| `--batch-size` / `--max-num-seqs` / `--max-model-len` | vLLM throughput / context knobs. |
| `--gpu-memory-utilization` | vLLM GPU memory fraction. |
| `--skip-viz` | Skip per-sample visualisation output. |
| `--seed` | RNG seed. |

Run `python evaluate_vllm_unified.py --help` for the complete list.

### Evaluating on your own data

Pass a `--val-json` in LlamaFactory format — a list of entries, each with an
image and a human/assistant turn pair:

```json
[
  {
    "images": ["images/MIT_0987.jpg"],
    "conversations": [
      {"from": "human", "value": "<image>Analyze this image and predict a human eye movement scanpath ..."},
      {"from": "gpt",   "value": "[(53, 48), (72, 32), (38, 54)]"}
    ]
  }
]
```

- Coordinates are integers on a **0–99 grid** (the image is treated as 100×100),
  written as `(x, y)` with `x` = column, `y` = row. The `gpt` turn holds the
  ground-truth scanpath the model is scored against.
- Image paths resolve relative to `--images-dir` (so `images/MIT_0987.jpg` with
  `--images-dir data` reads `data/images/MIT_0987.jpg`).
- Center-bias priors are looked up at `<pkl-dir>/<DATASET>/<index>.pkl` derived
  from the image name `DATASET_index.jpg` (e.g. `MIT_0987.jpg` →
  `data/centerbias/MIT/0987.pkl`). Each pickle is a dict
  `{"centerbias": <2-D log-density array>}`. A missing prior is an error, so that
  one run never mixes data-driven and synthetic baselines.
