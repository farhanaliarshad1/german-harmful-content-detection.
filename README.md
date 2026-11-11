# German Harmful Content Detection

**Fake news, hate speech and toxicity detection in German social media.** A comparison of single-task transformers, a unified multi-task BERT, T5 and LLM prompting on three German benchmarks, with SHAP explainability and leakage-safe evaluation.

![Python](https://img.shields.io/badge/Python-3.9%2B-informational)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-informational)
![Transformers](https://img.shields.io/badge/Hugging%20Face%20Transformers-4.46%2B-informational)
![License](https://img.shields.io/badge/License-MIT-informational)

**Paper:** [Review paper (PDF, 8 pages)](docs/review_paper.pdf) · [Markdown](docs/review_paper.md) · [Word](docs/review_paper.docx)

---

## Overview

Disinformation, hate speech and toxicity rarely appear alone. A false claim about a minority comes wrapped in hateful framing and draws toxic replies. Moderation systems, however, usually run a separate model for each harm.

This project builds and compares four modelling paradigms for detecting all three harms in German social media text:

| # | Paradigm | Models | Idea |
|---|---|---|---|
| 1 | Single-task fine-tuning | DistilBERT-German, XLM-RoBERTa, mBERT | one specialist encoder per harm |
| 2 | Unified multi-task BERT | bert-base-german-cased with 3 heads | one shared encoder, masked multi-task loss |
| 3 | Unified text-to-text | T5 / mT5 | all tasks as generation |
| 4 | LLM prompting | Llama 3.1 / 3.3, Mistral, Phi-4, GPT-4o-mini | zero- and few-shot, no training |

**Research questions**

- **RQ1.** Do separate models per task outperform one unified multi-task model?
- **RQ2.** What does combined detection teach us about healthier online discourse?
- **RQ3.** How do encoder, encoder-decoder and decoder-only (LLM) architectures compare?

---

## Datasets

All three corpora are German. Details and download instructions: [`data/README.md`](data/README.md).

| Task | Dataset | Size | Class balance | Main difficulty |
|---|---|---|---|---|
| Fake news / disinformation | [DeFaktS](https://aclanthology.org/2024.lrec-main.409/): German tweets annotated for disinformation framing | 5,000 | 50 / 50 | labels a framing style, not factual falsity |
| Hate speech | [GAHD](https://aclanthology.org/2024.naacl-long.248): German Adversarial Hate speech Dataset | 10,996 | 58 / 42 | adversarially written, often implicit hate |
| Toxicity | DeTox and GermEval 2018 / 2021 comments | 5,000 | 50 / 50 | rudeness vs. heated but legitimate debate |
| Unified | all three merged; each row labelled for one task, `-1` elsewhere | 15,000 | 5,000 per task | partially labelled multi-task data |

<details>
<summary>Example posts</summary>

| Task | Example (from the data) | Label |
|---|---|---|
| Fake news | „Bekämpfung der Erderwärmung: G7-Staaten gründen ‚Klimaclub‘ …“ | neutral |
| Hate speech | „Ich mag Sterne. Morgensterne, Abendsterne, Filmsternchen … Nur Judensterne nicht“ | hate (implicit) |
| Hate speech | „Ich liebe Inder“ | not hate |
| Toxicity | „Was ist nur mit diesem Scheißland passiert? Alle werden kollektiv bekloppt …“ | toxic |

</details>

---

## Architecture

### End-to-end system

<p align="center">
  <img src="assets/architecture_end_to_end.png" alt="End-to-end architecture: DeFaktS, GAHD and DeTox feed individual models and a unified multi-task model; prompting with any transformer model; evaluation with accuracy, precision, recall and F1; predictions served in a Streamlit UI" width="900">
</p>

The three German corpora (DeFaktS, GAHD, DeTox) go through tokenization and data preparation. Three paths are then trained and evaluated side by side:

- **Individual models:** one transformer per dataset.
- **Unified multi-task model:** trained on the combined dataset.
- **Prompting:** any transformer LLM, no training needed.

All paths are scored with the same metrics (accuracy, precision, recall, F1). In the system design, the multi-task model's predictions are shown in a locally hosted Streamlit UI.

### Individual (single-task) models

<p align="center">
  <img src="assets/architecture_individual_models.png" alt="Individual model workflow: raw data sources, preprocessing pipeline, model architecture (XLM-RoBERTa, DistilBERT-German-cased, multilingual BERT), training configuration and model saving" width="850">
</p>

### Unified multi-task model

<p align="center">
  <img src="assets/architecture_unified_model.png" alt="Unified model workflow: data preparation, text processing, multi-task model with a shared BERT encoder and task-specific heads for fake news, hate speech and toxicity, training loop, evaluation and model saving" width="900">
</p>

---

## Results

Accuracy and F1 on held-out test data (see the [paper](docs/review_paper.pdf) for precision and recall):

| Paradigm | Model | Fake news | Hate speech | Toxicity | Avg. accuracy |
|---|---|---|---|---|---|
| Single-task | DistilBERT-German / XLM-R / mBERT | **91.0%** (F1 91.0) | 78.1% (F1 77.9) | 88.7% (F1 88.7) | **85.9%** |
| Unified multi-task | bert-base-german-cased, 3 heads | 85.0% | 76.0% (F1 74.0) | **91.0%** (F1 90.0) | 84.0% |
| Unified text-to-text | t5-base (course version) | 73.0% (F1 51.0) | 68.0% (F1 42.0) | 51.0% (F1 40.0) | 64.0% |
| LLM, few-shot | GPT-4o-mini | 59% | **79%** | 78% | 72.0% |
| LLM, few-shot | Llama-3.3-70B (Q6) | 60% | 72% | 77% | 69.7% |
| LLM, zero-shot | GPT-4o-mini | 40% | 46% | 46% | 44.0% |

<details>
<summary>All LLM results (accuracy)</summary>

| Model | Fake news | Hate speech | Toxicity |
|---|---|---|---|
| llama3.1:8b-instruct-q6_K | 51% | 57% | 58% |
| llama3.3:70b-instruct-q6_K | 60% | 72% | 77% |
| mistral:7b-instruct-v0.2-q6_K | 57% | 64% | 70% |
| phi4 (14B) | 58% | 71% | 68% |
| gpt-4o-mini, zero-shot | 40% | 46% | 46% |
| gpt-4o-mini, few-shot | 59% | 79% | 78% |

</details>

---

## Key findings

1. **No single winner.** Specialist models are best on fake news, the unified model is best on toxicity, and few-shot LLMs are narrowly best on hate speech.
2. **Fake news needs a specialist.** The unified model loses 6 points (91% vs. 85%). Disinformation cues (framing, sensationalism, missing sources) differ from abuse cues, and a shared encoder dilutes them (task interference).
3. **Toxicity benefits from sharing.** Toxicity and hate speech share surface features such as insults and slurs, so multi-task training transfers positively (91% vs. 88.7%) with one encoder instead of three.
4. **Hate speech is the hardest task for fine-tuned models.** GAHD is adversarial, and much of its hate is implicit („Nur Judensterne nicht“). LLMs with broad world knowledge match the encoders here (79% vs. 78%).
5. **LLMs need demonstrations.** A few German examples add 19 to 33 points to GPT-4o-mini. LLMs still trail fine-tuned encoders by about 31 points on fake news, because DeFaktS's "disinformation" is an annotation-defined framing style rather than a general concept.
6. **SHAP reveals shortcut learning.** The fake-news model relies on framing words („trigger“, „liberal“). The toxicity model relies more on the topic word „Querdenker“ than on the insult „Scheiß“. The models learn dataset-specific cues, not the harm itself.
7. **Adversarial probing exposes failure modes.** The unified model misses calm conspiracy claims („Die Erde ist tatsächlich flach …“) and flags hostile political statements as fake news.

### Methodological lessons, fixed in this repository

| Pitfall in the original experiments | Effect | Fix |
|---|---|---|
| Minority-class oversampling before the train/test split (fake news, hate speech) | duplicated posts in train and test, inflated scores | split first; balance the training split only; remove exact duplicates |
| Test set used for early stopping and checkpoint selection (all single-task models) | optimistic test scores | separate validation split; test set evaluated once |
| Aggressive cleaning for GAHD (lowercasing; deleting digits, punctuation and emojis) | loses signals such as sarcasm markers and noun capitalisation | light, case- and emoji-preserving cleaning for all tasks |
| T5 trained to generate `-1` for missing labels; padding counted in the loss; LR schedule for 7 epochs, training for 4 | T5 collapsed (toxicity F1 40) | task-prefixed examples only for labelled tasks; `-100` label padding; schedule matches epochs; multilingual mT5 |
| Unparseable LLM answers mapped to class 0 | hidden format failures, biased accuracy | invalid answers counted and reported separately |
| Hard-coded paths and GPU ids, inconsistent column names, API key in source | not reproducible | YAML configs, one schema, key from environment, resumable JSONL cache |

---

## Quickstart

```bash
git clone https://github.com/farhanaliarshad1/german-harmful-content-detection.git
cd german-harmful-content-detection
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[llm,explain,dev]"
```

Put the CSV files in `data/` (see [`data/README.md`](data/README.md)), then:

```bash
# 1. Single-task models: one config per task (GPU used automatically)
python -m harmdetect.train --config configs/fake_news_distilbert.yaml
python -m harmdetect.train --config configs/hate_speech_xlmr.yaml
python -m harmdetect.train --config configs/toxicity_mbert.yaml
#    any value can be overridden from the command line
python -m harmdetect.train --config configs/fake_news_distilbert.yaml --model_name deepset/gbert-base --balance oversample

# 2. Unified multi-task BERT (all three tasks, masked loss)
python -m harmdetect.multitask --data data/merged_dataset.csv --epochs 3

# 3. Unified task-prefixed mT5
python -m harmdetect.t5_unified --data data/merged_dataset.csv --model_name google/mt5-small

# 4. LLM baseline: OpenAI, or any OpenAI-compatible server such as Ollama
export OPENAI_API_KEY=...
python -m harmdetect.llm_baseline --model gpt-4o-mini --few_shot --n 600
python -m harmdetect.llm_baseline --model llama3.1:8b --base_url http://localhost:11434/v1 --few_shot

# Prediction and explanation
python -m harmdetect.predict --model outputs/hate_speech-xlmr/model --text "Diese Minderheit zerstört unser Land."
python -m harmdetect.explain --model outputs/toxicity-mbert/model --data outputs/toxicity-mbert/test_predictions.csv

# Tests
pytest -q
```

Every run writes `metrics.json` (configuration, split sizes, test metrics, confusion matrix, per-class report), `test_predictions.csv` and the trained model to `outputs/<run>/`.

---

## Future work

- Re-run all single-task models with the leakage-safe pipeline and report mean and standard deviation over several seeds.
- Predict DeFaktS's fine-grained framing indicators (polarisation, vagueness, missing sources) instead of a binary label.
- Reduce task interference with adapters, loss re-weighting or gradient surgery.
- Fine-tune an open LLM with LoRA on the merged corpus, and add HateCheck-style functional tests for German.

## Acknowledgements

Developed as a Research Case Study in the M.Sc. Natural Language Processing programme at Universität Trier (TWON Metrics, WS 2024/25), together with fellow students Aastha Shreya, Shivang Sinha, Syed Huzaifa Ali, Donya Sheikhi and Muhammad Ahmed. Open-model inference used the university's LLM service.

## References

- Ashraf et al. (2024). *DeFaktS: A German Dataset for Fine-Grained Disinformation Detection through Social Media Framing.* LREC-COLING.
- Goldzycher, Röttger and Schneider (2024). *Improving Adversarial Data Collection by Supporting Annotators: Lessons from GAHD, a German Hate Speech Dataset.* NAACL.
- Wiegand, Siegel and Ruppenhofer (2018). *Overview of the GermEval 2018 Shared Task on the Identification of Offensive Language.*
- Risch, Stoll, Wilms and Wiegand (2021). *Overview of the GermEval 2021 Shared Task on the Identification of Toxic, Engaging, and Fact-Claiming Comments.*
- Devlin et al. (2019), *BERT*; Conneau et al. (2020), *XLM-R*; Raffel et al. (2020), *T5*; Lundberg and Lee (2017), *SHAP*.

The full bibliography is in the [review paper](docs/review_paper.md#references).

## License

Code: [MIT](LICENSE). The datasets keep their original licences and are not redistributed here.
