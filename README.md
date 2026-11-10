# BLIP-2 Q-Former Query Token Study

**Zero-shot Visual Question Answering (VQAv2) · Research project**

This repository investigates how changing the number of Q-Former query tokens **at inference time** affects the zero-shot visual question-answering (VQA) behavior of a pretrained BLIP-2 model. The project uses **Jupyter notebooks only** for Python code, with Anaconda for the environment and Hugging Face Transformers for the pretrained model. There is **no need to clone Salesforce LAVIS**.

> **Project status:** The VQAv2 validation subset has been prepared; the pretrained BLIP-2 model has been loaded and verified; and a single-question zero-shot baseline smoke test has succeeded. Resumable query-ablation and official VQA scoring notebooks are implemented. The main experiment remains to be run.

## 1. Background: the original BLIP-2 paper

**Paper:** Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. *BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.* ICML 2023, PMLR 202:19730–19742.\
**Read:** [https://proceedings.mlr.press/v202/li23q.html](https://proceedings.mlr.press/v202/li23q.html)

BLIP-2 connects a **frozen pretrained vision encoder** and a **frozen large language model** through a relatively lightweight **Querying Transformer (Q-Former)**. Instead of training the full vision and language models end to end, the Q-Former learns a set of visual representations suitable for the language model.

The original BLIP-2 pretraining procedure has two stages:

1. **Vision-language representation learning:** With the image encoder frozen, the Q-Former learns to extract visual information through image-text training objectives.
2. **Vision-to-language generative learning:** The Q-Former's visual representations are projected into the input space of a frozen language model to enable image-conditioned text generation.

In the pretrained configuration used here, the Q-Former has **32 learned query tokens**, each with a hidden dimension of **768**. The paper includes **zero-shot VQAv2 evaluation**: the pretrained model answers image questions *without being fine-tuned on VQAv2 question-answer training examples*. The paper also reports separate supervised VQA experiments; **this project focuses on zero-shot inference**.

Original implementation (reference only): [https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS)\
Hugging Face checkpoint used in this project: [https://huggingface.co/Salesforce/blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl)

## 2. Research question and experimental design

**Research question:** *How does changing the number of Q-Former query tokens available at inference affect BLIP-2's zero-shot VQA accuracy and computational efficiency?*

We plan to compare these conditions against the original **K = 32** checkpoint:

| Condition | Planned modification                            | Interpretation                         |
| --------- | ----------------------------------------------- | -------------------------------------- |
| K = 8     | Retain 8 of the original 32 pretrained queries  | Query-pruning ablation                 |
| K = 16    | Retain 16 of the original 32 pretrained queries | Query-pruning ablation                 |
| K = 32    | Use the unchanged pretrained query set          | Original baseline                      |
| K = 64    | Repeat the original 32 queries twice            | **Exploratory repeated-query control** |
| K = 128   | Repeat the original 32 queries four times       | **Exploratory repeated-query control** |

**Important methodological limitation:** K = 64 and K = 128 do **not** represent models with 64 or 128 independently *trained* queries. Repeating pretrained queries only tests the effect of a longer, repeated visual prefix. Valid claims about genuinely larger learned bottlenecks would require further image-text pretraining. We will analyze these controls separately from the K = 8/16 pruning experiments.

The main evaluation will use the **same 2,000 VQAv2 validation questions** (sampling seed **42**) across conditions. No VQAv2 training images, training questions, or training annotations are needed for the present zero-shot work. Planned measurements include standard VQAv2 answer accuracy, inference latency/throughput, and memory where relevant. A single-answer smoke test is **not** an accuracy estimate.

## 3. Project structure

```
BLIP2_STATS425_Final/
├── README.md
├── requirements.txt
├── .gitignore                         # Ignores data/val2014.zip only
├── scripts/
│   ├── data_extraction.ipynb
│   ├── load_blip2.ipynb
│   ├── baseline_test.ipynb
│   ├── query_ablation.ipynb
│   └── score_results.ipynb
├── data/                              # Other files may be committed (see size warning)
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   ├── v2_mscoco_val2014_annotations.json
│   ├── val2014.zip
│   └── validation_dataset/
│       ├── images/val2014/
│       ├── v2_OpenEnded_mscoco_val2014_questions_subset.json
│       ├── v2_mscoco_val2014_annotations_subset.json
│       └── subset_manifest.csv
├── hf_cache/                          # Keep local; do NOT commit large model files
└── offload/                           # Keep local; do NOT commit large offload files
```

**Repository convention:** All Python experiments and utilities live in `scripts/*.ipynb`. The repository does **not** require custom `.py` scripts. `README.md`, `requirements.txt`, and `.gitignore` are configuration/documentation files, not executable Python scripts.

## 4. Setup guide (Windows + Anaconda/Miniconda + NVIDIA GPU)

The commands below target the Windows/NVIDIA setup used to validate the current notebooks. Install [**Anaconda**](https://www.anaconda.com/download) or [**Miniconda**](https://www.anaconda.com/docs/getting-started/miniconda/install) first, then open **Anaconda Prompt**.

### Step 1 — Clone or download this GitHub repository

```
git clone https://github.com/<insert-your-github-username>/BLIP2_STATS425_Final
cd BLIP2_STATS425_Final
```

Alternatively, download the repository ZIP using GitHub's **Code → Download ZIP**, extract it, and open Anaconda Prompt in the extracted project folder. The project structure above assumes that the current working directory is the repository root.

### Step 2 — Create and activate the Conda environment

```
conda create -n blip2 python=3.11 pip -y
conda activate blip2
python -m pip install --upgrade pip setuptools wheel
```

The environment name is **`blip2`**. Each teammate can use another name, but must use that same name in the kernel-registration command below.

### Step 3 — Install CUDA-enabled PyTorch

**First**, make sure the NVIDIA driver is installed and visible:

```
nvidia-smi
```

**Then**, install the PyTorch build with the **CUDA 12.8 runtime**. This is the tested installation path for the RTX 5070 Ti setup and is published in the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/):

```
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

**Note:** Installing this CUDA-enabled PyTorch wheel is what we mean by setting up CUDA for this project. You generally **do not need to separately install the full NVIDIA CUDA Toolkit** to run these pretrained Transformers models. A compatible NVIDIA **driver is still required**. Do not install a CPU-only PyTorch build by mistake. If your system needs a different wheel, use the [official PyTorch installer](https://pytorch.org/get-started/locally/) and record the versions used.

Quick check:

```
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'n/a')"
```

On the project's GPU, `CUDA` and `BF16` should both be `True` and the GPU should be the **GeForce RTX 5070 Ti**. Windows Task Manager GPU numbering can differ from PyTorch CUDA numbering; PyTorch's `cuda:0` can correctly identify a card labeled **GPU 1** in Task Manager.

### Step 4 — Install `requirements.txt`

From the repository root with the `blip2` environment active:

```
python -m pip install -r requirements.txt
python -m pip check
```

The file covers Hugging Face Transformers, Accelerate, tokenizers, image processing, Jupyter, and plotting. **PyTorch is installed separately in Step 3** so it uses the proper CUDA wheel. You do not need to install or clone the full Salesforce LAVIS library.

### Step 5 — Register the Jupyter kernel

```
python -m ipykernel install --user --name blip2 --display-name "Python (blip2)"
```

If you used a different environment name, replace `blip2` in `--name` and in the display name accordingly.

Start JupyterLab **from the project root**:

```
jupyter lab
```

In every notebook, select **Kernel → Change Kernel → Python (blip2)**. Start Jupyter from the repository root (or `scripts/`) so the notebooks can resolve their relative paths.

## 5. Download the VQAv2 data (validation only)

**Official download page:** [https://visualqa.org/download.html](https://visualqa.org/download.html)

Download **three archives**, all for **VQAv2 / COCO real images / validation**:

| Required download on official page                                            | What it provides            | File to place in `data/`                                           |
| ----------------------------------------------------------------------------- | --------------------------- | ------------------------------------------------------------------ |
| **VQA Input Questions → Validation questions 2017 v2.0**                      | Questions and question IDs  | Extract ZIP to obtain `v2_OpenEnded_mscoco_val2014_questions.json` |
| **VQA Annotations → Balanced Real Images → Validation annotations 2017 v2.0** | Human answers for scoring   | Extract ZIP to obtain `v2_mscoco_val2014_annotations.json`         |
| **VQA Input Images → COCO → Validation images**                               | COCO validation photographs | Keep archive **`val2014.zip`** intact                              |

The two VQA question/annotation ZIPs should be **unzipped** so the JSON files sit directly in `data/`. Only the COCO image file needs to remain a ZIP for the image-extraction step: `data_extraction.ipynb` opens it directly and extracts only the images associated with our selected questions. **Do not extract all of `val2014.zip` manually.**

The expected inputs are:

```
data/
├── v2_OpenEnded_mscoco_val2014_questions.json
├── v2_mscoco_val2014_annotations.json
└── val2014.zip
```

### Step 6 — Run `scripts/data_extraction.ipynb`

With the **Python (blip2)** kernel selected, run the notebook from top to bottom. It:

- Matches VQAv2 validation questions with their annotations.
- Samples **2,000** annotated questions using a fixed random seed of **42**.
- Identifies the distinct COCO image filenames needed by those questions.
- Streams only the corresponding image files out of `val2014.zip`.
- Writes the question/annotation subsets and `subset_manifest.csv` into `data/validation_dataset/`.

The notebook safely reuses extracted images whose file sizes match the source archive, so reruns should not require extracting everything again. Retain `val2014.zip` until you have confirmed the subset is complete. Teammates can download the image ZIP separately from the official COCO link instead of storing it in GitHub. If you remove the ZIP locally later, the extraction cell cannot run again without restoring it.

## 6. Load BLIP-2 from Hugging Face

### `scripts/load_blip2.ipynb`

The loader uses the **official Hugging Face Transformers implementation and Salesforce checkpoint**, not a local clone of LAVIS:

- Model ID: **`Salesforce/blip2-flan-t5-xl`**
- Vision encoder: ViT-G-sized configuration in this checkpoint (39 layers; hidden size 1,408)
- Language model: Flan-T5-XL
- Original Q-Former query count: **32** (query embeddings shape **`[1, 32, 768]`**)
- Precision: **BF16**
- Model placement: `device_map="auto"` with an initial GPU placement budget of **`12GiB`**
- Shared downloads: `hf_cache/` (created automatically)
- Potential offload directory: `offload/`

Run `load_blip2.ipynb` directly to verify that the model loads, or simply proceed to `baseline_test.ipynb`, which calls the loader automatically.

**Important:** The first load downloads large pretrained weight files. Future runs reuse the **disk cache**, so the weights are not normally downloaded again. Restarting the Jupyter kernel still requires reloading the model into CPU/GPU memory.

### Sharing the loader between notebooks

`baseline_test.ipynb` calls `load_blip2.ipynb` using IPython's **`%run -i`**, rather than duplicating its loader code. The loader checks whether `model` and `processor` already exist **in the current kernel** and reuses them.

This reuse does **not** automatically work between two independent kernels. To avoid accidentally holding two copies of the large model in VRAM, run experiments in one shared kernel or stop other model-holding kernels first. Other teammates can use the notebook loader the same way in future experiment notebooks.

## 7. Run the zero-shot baseline smoke test

### `scripts/baseline_test.ipynb`

Run the cells in order. The notebook:

1. Invokes the shared BLIP-2 loader.
2. Reads the first question in our extracted validation subset.
3. Loads the corresponding COCO image.
4. Prompts BLIP-2 with **`Question: {question} Short answer:`**.
5. Generates one answer using greedy decoding (`num_beams=1`, `max_new_tokens=10`).
6. Displays the generated answer and most common human answers for inspection.

The notebook does **not** run fine-tuning, download a second model copy, or report VQA benchmark accuracy from this single example.

### First observed baseline result (K = 32)

| Field                         | Result                                                                                       |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| Question                      | What is plastered all over the big vehicle?                                                  |
| Model answer                  | **feel good bus**                                                                            |
| Human answers (most frequent) | **feel good bus** (3), people (2), advertisement (2), artwork (1), feel good bus graphic (1) |

The model's answer matches the most frequently reported human answer for this image. This is a **successful qualitative smoke test**, not evidence of overall VQAv2 accuracy. The existing notebook intentionally avoids verbose GPU diagnostics; any later performance profiling should be collected consistently in the experiment notebooks.

## 8. Run the query-count experiment

Use **Python (blip2 AMD)** on the local RX 9070 setup. The experiment reuses the
existing loader, BF16 precision, automatic placement with the 12 GiB GPU budget,
and shared Hugging Face cache. The NVIDIA setup above remains relevant to a
teammate using that hardware. Each run records its actual GPU, ROCm version,
package versions, checkpoint revision, device map, query indices, and data hashes.

1. Open `scripts/query_ablation.ipynb` and choose `STAGE = "smoke"` in its first code cell.
2. Run all cells. This evaluates the same 10 questions in all 13 conditions.
3. Open `scripts/score_results.ipynb`, select the same stage, and run all cells.
4. When ready for a larger experiment, set the stage to `"pilot"` for 500 questions
   or `"main"` for the complete existing 2,000-question subset in both notebooks.

K=8 and K=16 each use five fixed seeds, 0-4. The eight retained queries are a
subset of the sixteen retained queries within each seed. K=32 uses the original
queries. K=64 and K=128 repeat all 32 pretrained query embeddings two and four
times before the Q-Former. They are exploratory repeated-query controls. The
13-condition order is shuffled reproducibly using seed 2026.

Generation uses the prompt `Question: {question} Short answer:`, five beams,
length penalty -1, at most 10 new tokens, and batch size one. Every condition
receives two unmeasured warmups. The earlier baseline notebook uses greedy
decoding for its single-question functionality check.

Results are stored in `results/query_ablation/<stage>-<signature>/`. Each condition
has durable `records.jsonl`, official-format `predictions.json`, and `status.json`.
Rerun identical settings to resume. A lock prevents concurrent writers and a
manifest prevents combining incompatible settings. A changed protocol, checkpoint,
environment, or dataset gets a different experiment directory. Interrupted final
record writes are recovered. The original K=32 queries are restored on exit.

The scoring notebook verifies the pinned official VQA evaluator source at its
first download, translates its two Python 2 print statements, and preserves its
answer normalization and consensus scoring. It checks all question IDs and scores
only the evaluated subset. It selects the most recently completed experiment for
the chosen stage, or an explicitly supplied `EXPERIMENT_DIR`.

The `analysis/` directory contains per-run and grouped accuracy tables, per-question
scores, accuracy by answer type, paired differences against K=32, and figures for
accuracy, latency, throughput, peak allocated memory, and query-selection variability.
95% paired image-cluster bootstrap intervals use 2,000 replicates and seed 425.
Grouped intervals condition on the selected query subsets. Standard deviations
across the five subsets are reported separately.

Timing covers synchronized `model.generate()` only. Image reading, preprocessing,
transfers, decoding, and saving are excluded. Throughput uses summed generation
time. Peak allocated GPU memory includes the model and current inputs. Keep each
hardware setup's speed and memory comparisons within that setup. Smoke results
validate execution and are too small for project conclusions.

## 9. GitHub: ignoring only `val2014.zip`

To ignore **only the COCO validation ZIP archive** (but not automatically ignore the rest of `data/`), the root-level `.gitignore` should contain exactly:

```
# Ignore only the large original COCO validation archive
/data/val2014.zip
```

Delete any earlier `data/` rule if you want the remaining files in `data/` to be eligible for commits. The leading `/` means the rule applies to `data/val2014.zip` from the repository root. GitHub will not automatically upload this file when you run `git add .`.

If the ZIP was **already tracked** before this rule was added, remove it from Git's index **without deleting your local copy**:

```
git rm --cached data/val2014.zip
```

If Git reports that the file was never tracked, no action is needed. Inspect your staging area before pushing:

```
git status
```

**Size and privacy warning:** This requested one-line `.gitignore` also means that extracted JPEGs, VQAv2 JSONs, `hf_cache/`, `offload/`, and Jupyter checkpoint files are *not* automatically ignored. This is intentional for the "ignore only `val2014.zip`" setting, but **do not run `git add .` blindly** if those directories exist. Hugging Face model weights and offload artifacts can exceed GitHub's normal file limits, and copied dataset files may make the repository unnecessarily large. Inspect all files being staged, and do not commit secrets, private local paths, or unneeded model weights. If GitHub rejects a large file, exclude that file or directory from tracking rather than uploading the weights.

For a simple notebook/code-only repository, an alternative *more protective* `.gitignore` can additionally exclude `hf_cache/`, `offload/`, `.ipynb_checkpoints/`, and extracted images. That is **not** the default in the accompanying one-line `.gitignore` file, because this project currently requests that only the COCO ZIP be excluded.

For reproducibility, record the **model ID**, package versions, query-selection seed(s), **validation sampling seed 42**, prompt, decoding settings, and selected checkpoint. For the final benchmark, use the **official VQAv2 evaluation protocol** rather than treating a single matching answer as an accuracy score.

## 10. References

- Li et al. (2023), **BLIP-2**, ICML: [https://proceedings.mlr.press/v202/li23q.html](https://proceedings.mlr.press/v202/li23q.html)
- Salesforce LAVIS (original code): [https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS)
- Salesforce Hugging Face checkpoint: [https://huggingface.co/Salesforce/blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
- Hugging Face Transformers: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
- VQAv2 data downloads: [https://visualqa.org/download.html](https://visualqa.org/download.html)
- VQAv2 evaluation protocol: [https://visualqa.org/evaluation.html](https://visualqa.org/evaluation.html)
