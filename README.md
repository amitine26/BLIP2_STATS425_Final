# BLIP-2 Q-Former Query Token Study

Zero-shot visual question answering on VQAv2 | Inference-time query ablation and leave-one-query-out analysis

This repository investigates how the number and identity of pretrained Q-Former query tokens affect BLIP-2's zero-shot visual question answering (VQA) accuracy and inference efficiency. The experiments use the Hugging Face Transformers implementation of `Salesforce/blip2-flan-t5-xl` and contain no model fine-tuning. All executable project code is kept in Jupyter notebooks under `scripts/`.

## 1. Background and research questions

The main reference is:

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML, Proceedings of Machine Learning Research 202:19730-19742.

- Paper: https://proceedings.mlr.press/v202/li23q.html
- Original Salesforce LAVIS implementation, for reference: https://github.com/salesforce/LAVIS
- Model checkpoint used here: https://huggingface.co/Salesforce/blip2-flan-t5-xl

BLIP-2 connects a frozen image encoder with a frozen language model using a Querying Transformer (Q-Former). During pretraining, the Q-Former learns to extract visual information that can be passed to the language model. The checkpoint used here has 32 pretrained query embeddings of width 768. Its visual encoder is EVA-CLIP ViT-g/14, and its language model is Flan-T5-XL.

The original BLIP-2 pretraining procedure includes vision-language representation learning and vision-to-language generative learning. This project does not repeat that pretraining. It evaluates inference-time changes to the pretrained queries with the rest of the model unchanged.

The main research questions are:

1. How does changing the number of available pretrained Q-Former queries affect zero-shot VQA accuracy and computational efficiency?
2. How sensitive is VQA performance to the identities of queries retained during pruning?
3. What is the marginal effect of removing each individual pretrained query while keeping the remaining 31?

This is zero-shot VQAv2 evaluation, not supervised VQA fine-tuning. The original paper reports those as separate evaluation settings.

## 2. Experimental design

### 2.1 Query-count ablation

The original K = 32 configuration is the reference. The 13 conditions are:

| K | Method | Runs | Interpretation |
| ---: | --- | ---: | --- |
| 8 | Retain a random subset of 8 original queries | 5 seeds | Pretrained-query pruning |
| 16 | Retain a random subset of 16 original queries | 5 seeds | Pretrained-query pruning |
| 32 | Keep all original queries | 1 | Unmodified reference |
| 64 | Repeat the 32 original query embeddings twice | 1 | Repeated-query control |
| 128 | Repeat the 32 original query embeddings four times | 1 | Repeated-query control |

Query subset seeds are 0, 1, 2, 3, and 4. Within a seed, the K = 8 selection is contained in the corresponding K = 16 selection. The 13-condition order is shuffled using seed 2026.

The K = 64 and K = 128 cases repeat pretrained embeddings before Q-Former processing. They do not add 32 or 96 new independent learned queries, and they do not test a separately pretrained Q-Former of greater capacity. Any claim about genuinely larger learned bottlenecks would require additional training.

### 2.2 Leave-one-query-out ablation (LOQOA)

LOQOA evaluates the marginal contribution of each of the original 32 pretrained queries. It uses 33 conditions:

- One original K = 32 reference with all pretrained queries.
- Thirty-two K = 31 conditions, each omitting exactly one different query with zero-based index 0 through 31.
- The other 31 pretrained queries remain in their original order; no query is reinitialized or retrained.
- Every condition receives the same selected validation questions and generation settings.

Define the marginal importance of original query `i`, measured in percentage points, as:

`I_i = Accuracy(K=32) - Accuracy(with query i removed)`

Positive importance means removal reduced observed VQA accuracy. Negative importance means removal improved observed accuracy. These are marginal effects conditional on retaining the other 31 queries, not independent or additive query contributions.

LOQOA question counts are controlled by `NUM_QUESTIONS` in `scripts/leave_one_query_out.ipynb`:

| `NUM_QUESTIONS` | Stage | Conditions | Predictions |
| ---: | --- | ---: | ---: |
| 10 | `smoke` | 33 | 330 |
| 200 | `pilot` | 33 | 6,600 |
| 2000 | `main` | 33 | 66,000 |

Its 10-question sample is nested in its 200-question sample, which is nested in the 2,000-question sample. All are selected reproducibly from the prepared validation subset. The notebook also reports observed run times and projects the 2,000-question runtime using either pilot stage and a fixed-overhead plus linear-cost estimate.

### 2.3 Shared inference protocol

Both experiment families use:

- Pretrained checkpoint: `Salesforce/blip2-flan-t5-xl`.
- Precision: BF16, single GPU, batch size 1.
- Prompt: `Question: {question} Short answer:`.
- Deterministic beam search with `num_beams=5`, `length_penalty=-1.0`, and `max_new_tokens=10`.
- Two warmup generations per unfinished condition, excluded from scored predictions and timed generation summaries.
- No parameter updates, gradient calculations, or optimizer steps.
- Saved predictions, per-question performance measurements, experiment manifests, and resumable condition outputs.

The separate one-question `baseline_test.ipynb` uses greedy decoding as a functionality check only. It should not be compared with the beam-search benchmark results as though it were the same experimental condition.

## 3. Repository structure

```text
BLIP2_STATS425_Final/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── data_extraction.ipynb
│   ├── load_blip2.ipynb
│   ├── baseline_test.ipynb
│   ├── query_ablation.ipynb
│   ├── score_results.ipynb
│   ├── leave_one_query_out.ipynb
│   └── score_loqoa.ipynb
├── data/
│   ├── v2_OpenEnded_mscoco_val2014_questions.json
│   ├── v2_mscoco_val2014_annotations.json
│   ├── val2014.zip
│   └── validation_dataset/
│       ├── images/val2014/
│       ├── v2_OpenEnded_mscoco_val2014_questions_subset.json
│       ├── v2_mscoco_val2014_annotations_subset.json
│       └── subset_manifest.csv
├── results/
│   ├── query_ablation/
│   │   └── <stage>-<experiment_signature>/
│   │       └── analysis/
│   └── leave_one_query_out/
│       └── <stage>-<experiment_signature>/
│           └── analysis/
├── hf_cache/
└── offload/
```

Some files shown under `data/` and the model-cache directories are generated or downloaded locally and should not necessarily be committed. In particular, the original validation annotations JSON exceeds GitHub's standard 100 MB per-file limit. The paths under `results/` are generated by the completed experiment and scoring notebooks; stage signatures depend on the exact environment and protocol.

Repository convention: Python experiments and supporting logic live in `scripts/*.ipynb`. No custom `.py` script is required to run the current pipeline.

## 4. Installation and environment setup

### 4.1 Clone the repository and prepare Python

Install Anaconda or Miniconda and open Anaconda Prompt. The GitHub repository is:

https://github.com/amitine26/BLIP2_STATS425_Final

```bash
git clone https://github.com/amitine26/BLIP2_STATS425_Final.git
cd BLIP2_STATS425_Final
conda create -n blip2 python=3.11 pip -y
conda activate blip2
python -m pip install --upgrade pip setuptools wheel
```

The `main` branch contains team-approved work. Development for the NVIDIA query ablation and LOQOA updates has been carried out on `query-ablation-cuda`. Teammates who need those branch-specific notebooks or results should switch explicitly:

```bash
git switch query-ablation-cuda
```

### 4.2 Install GPU-enabled PyTorch

The completed NVIDIA experiments were run on an RTX 5070 Ti. A compatible NVIDIA driver is required. Check it with:

```bash
nvidia-smi
```

Install a CUDA-enabled PyTorch build suitable for the system. The original Windows setup used a CUDA 12.8 wheel command:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

If those exact versions are unavailable for the machine, select a compatible CUDA-enabled wheel using the official PyTorch instructions at https://pytorch.org/get-started/locally/ and record the installed version in the experiment manifest. The full NVIDIA CUDA Toolkit is usually not required for the pip wheels, but the NVIDIA driver is required.

For AMD ROCm, use a supported ROCm-enabled PyTorch installation for the teammate's operating system and GPU. Do not install NVIDIA CUDA wheels on an AMD machine. PyTorch commonly exposes ROCm GPUs through the `torch.cuda` API.

Verify GPU support:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('GPU available:', torch.cuda.is_available()); print('BF16:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else 'n/a')"
```

### 4.3 Install project dependencies

With the intended Conda environment activated:

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The repository requirements cover Transformers, Accelerate, image processing, numerical analysis, plotting, and Jupyter. GPU-enabled PyTorch is installed separately to avoid accidentally choosing a CPU-only build. Salesforce LAVIS does not need to be cloned or installed.

```bash
python -c "import sentencepiece, pandas, numpy, matplotlib, torch; print('SentencePiece:', sentencepiece.__version__); print('pandas:', pandas.__version__); print('CUDA:', torch.cuda.is_available())"
```
### 4.4 Register and start Jupyter

```bash
python -m ipykernel install --user --name blip2 --display-name "Python (blip2)"
jupyter lab
```

Start Jupyter from the repository root or `scripts/`, and select the `Python (blip2)` kernel. The loaders and experiment notebooks infer the repository root from the working directory.

## 5. Dataset preparation

Official download page: https://visualqa.org/download.html

The dataset is VQAv2 validation on COCO val2014 images. The original preparation process used:

| Source | File under `data/` |
| --- | --- |
| VQAv2 validation questions | `v2_OpenEnded_mscoco_val2014_questions.json` |
| VQAv2 validation annotations | `v2_mscoco_val2014_annotations.json` |
| COCO val2014 images | `val2014.zip` |

Unzip the questions and annotations archives into `data/`. Keep `val2014.zip` intact. Then run `scripts/data_extraction.ipynb` to:

1. Select 2,000 annotated questions reproducibly, using seed 42.
2. Find the corresponding COCO image filenames.
3. Extract only the necessary images from the COCO ZIP.
4. Write the subset question and annotation JSON files and `subset_manifest.csv`.

The completed 2,000-question LOQOA main stage contains questions from 1,931 distinct images. The same prepared 2,000-question subset is used in the query-count experiment.

Some teammates may already receive extracted subset images and annotations from the repository if those files have been committed. If files are missing, run the extraction notebook after obtaining the official downloads. The original 163.81 MB annotations JSON is too large to commit directly to standard GitHub Git storage.

No VQAv2 training dataset or supervised fine-tuning data is used by the present experiments.

## 6. Running the notebooks

### 6.1 Model loading and smoke testing

`scripts/load_blip2.ipynb` loads the shared Hugging Face checkpoint in BF16 and configures the original 32 queries. Its image model uses a ViT-g-sized representation with 1,408 hidden features, and its query parameter has shape `[1, 32, 768]`. The loader uses a shared `hf_cache/` disk cache and a `12GiB` model-placement budget in the original single-GPU setup.

The model downloads only on the first cache miss. Restarting the Jupyter kernel reloads weights into memory, but does not normally download them again. Avoid having multiple model-holding Jupyter kernels competing for VRAM.

`scripts/baseline_test.ipynb` calls the shared loader through `%run -i` and performs one greedy-decoding example. The observed question was "What is plastered all over the big vehicle?" and the model answered "feel good bus," matching the most frequent human answer for that example. This verifies inference functionality but is not a benchmark accuracy estimate.

### 6.2 Query-count ablation and scoring

1. Open `scripts/query_ablation.ipynb` with the GPU-enabled environment.
2. Use `STAGE = "smoke"` for 10 questions per condition, `"pilot"` for 500, or `"main"` for 2,000. The notebook defaults to the smoke stage.
3. Run the notebook. A fresh main stage evaluates 13 conditions and saves 26,000 predictions.
4. Open `scripts/score_results.ipynb`. Set the matching `STAGE`; optionally specify `GPU_BACKEND = "cuda"` or `"rocm"`, or set an exact `EXPERIMENT_DIR`.
5. Run all scoring cells to produce official VQAv2 scores, paired comparisons, tables, and plots. No GPU inference is rerun during scoring.

Prediction and measurement files are stored under `results/query_ablation/<stage>-<signature>/`. The scorer writes `analysis/` inside that experiment directory. Its outputs include:

- `run_summary.csv`, `per_question_scores.csv`, `paired_comparisons.csv`, and `group_summary.csv`.
- Accuracy by answer type and question-level changes, including grouped tables.
- `accuracy_vs_k.png`, `efficiency_vs_k.png`, and `query_selection_variability.png`.
- `paired_accuracy_differences.png`, `accuracy_by_answer_type.png`, and `question_level_changes.png`.
- `scoring_protocol.json` with evaluation settings and provenance.

The later optional computational efficiency cell can additionally write `detailed_computational_efficiency.csv`, `efficiency_by_condition.csv`, and `detailed_computational_efficiency.png` from existing saved measurement data. It does not require new model inference.

### 6.3 Leave-one-query-out ablation and scoring

1. Open `scripts/leave_one_query_out.ipynb` with the GPU-enabled environment.
2. Set `NUM_QUESTIONS = 10` for a smoke test, `200` for the pilot, or `2000` for the full study. The notebook defaults to 10.
3. Run all cells to evaluate the original K = 32 reference and every individual K = 31 omission.
4. Review `runtime.json` and `runtime_estimates.csv` to estimate the full experiment's inference-loop time from fresh 10- and 200-question runs on the same GPU. Runtime projections are estimates, not guarantees.
5. Open `scripts/score_loqoa.ipynb`, choose `STAGE = "smoke"`, `"pilot"`, or `"main"` to match the completed run, and select the exact experiment if needed.
6. Run all scoring cells to produce official VQAv2 accuracy, marginal importance, confidence intervals, answer-type analysis, question-level comparisons, and figures.

LOQOA results are separate from the query-count results, under `results/leave_one_query_out/<stage>-<signature>/`. The LOQOA scoring `analysis/` folder includes:

- `run_summary.csv`, `per_question_scores.csv`, and `query_importance.csv`.
- `ranked_query_importance.csv`, `accuracy_by_answer_type.csv`, and `question_level_changes.csv`.
- `query_importance.png` and `ranked_query_importance.png`.
- `query_importance_by_answer_type.png`, `question_level_changes.png`, and `prediction_agreement.png`.
- `loqoa_efficiency.png` and `scoring_protocol.json`.

The scoring notebooks operate on completed saved predictions using the pinned official VQAv2 evaluation implementation. They do not load BLIP-2 or require CUDA/ROCm to score previous runs. The experiments are independently runnable; neither scoring notebook is a prerequisite for the other experiment.

### 6.4 Reproducibility and measurement notes

Experimental manifests store the checkpoint revision, hardware backend, GPU name, software versions, selected query indices, dataset hashes, generation settings, and code identifiers. The inference notebooks support interrupted-run recovery, file locking, and restoration of the original 32-query configuration.

The statistical analyses use 2,000 paired image-cluster bootstrap replicates to account for multiple questions referring to the same image. For query pruning, variation across the five query subsets is separate from evaluation-sample uncertainty. For LOQOA, pointwise intervals and an approximate simultaneous confidence band across the 32 omissions are both reported.

Generation latency measures synchronized `model.generate()` time. It excludes image reading, preprocessing, some transfers, answer decoding, and saving results. The VRAM measurements are PyTorch GPU-memory measurements, not system RAM utilization. System RAM usage was not logged and cannot be reconstructed retrospectively from these files. Cross-GPU latency and VRAM results should not be compared as though the systems were identical.

## 7. Query-count ablation results

The following table summarizes the completed NVIDIA RTX 5070 Ti main-stage results. K = 8 and K = 16 accuracies are averages across five subsets; other K values are single runs.

| Query count | Method | VQAv2 accuracy (%) | Difference from K = 32 (pp) | Paired 95% CI of difference (pp) |
| ---: | --- | ---: | ---: | --- |
| 8 | Random pruning | 53.325 | -6.165 | [-7.515, -4.804] |
| 16 | Random pruning | 56.359 | -3.131 | [-4.210, -2.086] |
| 32 | Original baseline | 59.490 | 0.000 | Reference |
| 64 | Repeated-query control | 58.650 | -0.840 | [-1.596, -0.060] |
| 128 | Repeated-query control | 57.740 | -1.750 | [-2.667, -0.744] |

The paired confidence intervals above are not simultaneous intervals adjusted for all comparisons. For K = 8 and K = 16, these intervals are conditional on the five selected query subsets.

### 7.1 Query-subset variability

| Seed | K = 8 accuracy (%) | K = 16 accuracy (%) |
| ---: | ---: | ---: |
| 0 | 50.640 | 56.735 |
| 1 | 52.940 | 50.445 |
| 2 | 54.005 | 57.350 |
| 3 | 54.295 | 58.120 |
| 4 | 54.745 | 59.145 |

The best observed K = 16 subset achieved 59.145% accuracy, just 0.345 percentage points below the K = 32 reference. The worst K = 16 subset achieved 50.445%. Selection sensitivity is substantial: adding eight queries to the K = 8 subset did not improve performance for every seed.

The best subset was identified on the same evaluation questions, so its generalization cannot be inferred without an independent evaluation set.

### 7.2 VQAv2 answer type

| K | Yes/no (%) | Number (%) | Other (%) |
| ---: | ---: | ---: | ---: |
| 8 | 77.092 | 21.304 | 42.789 |
| 16 | 77.964 | 24.713 | 47.419 |
| 32 | 82.952 | 25.061 | 49.797 |
| 64 | 82.536 | 24.980 | 48.435 |
| 128 | 82.783 | 24.372 | 46.545 |

The evaluated subset contains 769 yes/no, 247 number, and 984 other questions. Reductions differ by answer category; the subgroup results are descriptive rather than evidence of a causal query specialization.

### 7.3 Computational efficiency

The following values summarize the previously recorded NVIDIA measurements. For K = 8 and K = 16, the latency values are means of per-run medians. They are not whole-program wall-clock timings.

| K | Median generation latency (s) | Throughput (questions/s) | Peak allocated VRAM (GiB) |
| ---: | ---: | ---: | ---: |
| 8 | 0.1199 | 7.9479 | 7.4085 |
| 16 | 0.1239 | 7.6848 | 7.4163 |
| 32 | 0.1304 | 7.2961 | 7.4338 |
| 64 | 0.1278 | 7.5900 | 7.4637 |
| 128 | 0.1305 | 7.4900 | 7.5263 |

Reducing K from 32 to 8 generally shortened generation latency and reduced measured peak GPU allocation by approximately 26 MiB, but accuracy declined by 6.165 percentage points. Timing was not strictly monotonic across all K values, and the majority of model weight memory belongs to the unchanged vision encoder and language model.

A prior AMD RX 9070 main-stage replication showed the same overall accuracy pattern: 53.345% for K = 8, 56.446% for K = 16, 59.455% for K = 32, 58.670% for K = 64, and 57.745% for K = 128. That replication should be reported separately from NVIDIA timing and memory measurements.

## 8. Leave-one-query-out results

The completed NVIDIA LOQOA main stage evaluated 32 different K = 31 omissions plus the original K = 32 baseline on 2,000 questions from 1,931 images.

| Measure | Result |
| --- | ---: |
| K = 32 reference accuracy | 59.490% |
| Mean accuracy across 32 K = 31 omissions | 59.316% |
| Mean marginal importance | +0.174 percentage points |
| Lowest K = 31 accuracy | 58.925% |
| Highest K = 31 accuracy | 60.030% |
| Queries with positive observed marginal importance | 26 |
| Queries with negative observed marginal importance | 6 |

### 8.1 Selected query importance estimates

| Removed query index | K = 31 VQA accuracy (%) | Marginal importance (pp) | Interpretation |
| ---: | ---: | ---: | --- |
| 0 | 58.925 | +0.565 | Largest observed loss after removal |
| 31 | 59.025 | +0.465 | Second-largest observed loss |
| 3 | 59.060 | +0.430 | Positive observed marginal importance |
| 19 | 59.090 | +0.400 | Positive observed marginal importance |
| 15 | 59.705 | -0.215 | Higher observed accuracy without query |
| 2 | 60.030 | -0.540 | Largest observed improvement after removal |

The remaining query-specific values are available in `query_importance.csv` and `ranked_query_importance.csv`, along with confidence intervals and the complete ranking.

### 8.2 Statistical interpretation

The LOQOA analysis uses paired image-cluster bootstrap confidence intervals. The pointwise 95% intervals for queries 0 and 31 narrowly excluded zero, but all 32 approximate simultaneous 95% confidence bands included zero. Therefore, no individual query can be declared to have statistically distinguishable importance from zero under that multiple-comparison-aware analysis.

Removing query 2 increased observed accuracy by 0.540 percentage points, but its confidence interval included zero. This does not establish that dropping query 2 reliably improves accuracy on new questions.

### 8.3 Answer-type and question-level findings

Individual omissions did not have identical observed effects across yes/no, number, and other questions. For example, removing query 8 reduced observed numerical-question accuracy by 2.186 percentage points, while its yes/no accuracy difference was near zero. These are exploratory category-level comparisons and do not establish functional specialization.

The vast majority of question-level official VQA scores remained unchanged after removing any single query. Exact generated-answer agreement with K = 32 ranged from approximately 90.00% to 97.90% across omissions, showing that answers can change without changing their official VQA score.

Taken together with the K = 8 and K = 16 experiment, LOQOA suggests local robustness to losing a single query but greater sensitivity to removing groups of queries. Query interactions and redundancy are plausible explanations, but the LOQOA design alone cannot isolate or prove specific interactions.

## 9. Discussion, limitations, and next steps

The combined experiments support the following conclusions:

1. The original K = 32 checkpoint achieved the highest mean accuracy among the query-count conditions tested.
2. Query identity matters: performance varies substantially across randomly chosen K = 8 and K = 16 subsets.
3. Removing a single pretrained query usually causes a small observed accuracy change, while removing many queries can cause a much larger decrease.
4. Pruning provided modest GPU latency and memory benefits compared with the observed losses in VQA accuracy.
5. Repeating pretrained queries did not improve accuracy, but these controls do not test independently trained larger Q-Formers.

Potential next steps are to compare importance-ranked K = 8 and K = 16 subsets against random pruning, study pairwise or higher-order query interactions, evaluate robustness to visual occlusion, and test selected subsets on independent questions. Rankings derived from the current validation questions should not be evaluated on those same questions as though the evaluation were independent.

The 2,000-question subset is not the full VQAv2 validation benchmark. The experiments modify only inference-time query inputs; they do not fine-tune Q-Former parameters or retrain a model with a different number of learned queries. Grouped and per-query comparisons involve multiple hypotheses, so confidence intervals and selected best-performing subsets require cautious interpretation.

## 10. References

- Li et al. (2023), **BLIP-2**, ICML: [https://proceedings.mlr.press/v202/li23q.html](https://proceedings.mlr.press/v202/li23q.html)
- Salesforce LAVIS (original code): [https://github.com/salesforce/LAVIS](https://github.com/salesforce/LAVIS)
- Salesforce Hugging Face checkpoint: [https://huggingface.co/Salesforce/blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
- Hugging Face Transformers: [https://huggingface.co/docs/transformers/](https://huggingface.co/docs/transformers/)
- VQAv2 data downloads: [https://visualqa.org/download.html](https://visualqa.org/download.html)
- VQAv2 evaluation protocol: [https://visualqa.org/evaluation.html](https://visualqa.org/evaluation.html)
