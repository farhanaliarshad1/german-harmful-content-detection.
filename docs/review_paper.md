# One Model or Many? A Review of Transformer and LLM Approaches to Detecting Fake News, Hate Speech and Toxicity in German Social Media

**Farhan Ali Arshad**
M.Sc. Natural Language Processing, Universität Trier
*Research Case Study: TWON Metrics, Group B1 (WS 2024/25)*

---

## Abstract

Online disinformation, hate speech and toxicity are usually tackled by separate classifiers, even though the three phenomena co-occur and share linguistic signals. This paper reviews the main modelling paradigms for these tasks: fine-tuned encoders, multi-task learning, text-to-text models and prompted large language models (LLMs). It grounds the review in a case study on German social media text. We compare (i) three single-task encoders (DistilBERT-German, XLM-RoBERTa and multilingual BERT), each fine-tuned on one dataset, (ii) a unified multi-task BERT with a shared encoder and task-specific heads, (iii) a unified T5 model that generates all three labels as text, and (iv) zero- and few-shot prompting of open and commercial LLMs. The datasets are DeFaktS (disinformation), GAHD (adversarial hate speech) and DeTox/GermEval (toxicity). Single-task models gave the best fake-news result (91% accuracy). The unified model was best on toxicity (91%) and close on hate speech (76% vs. 78%) while needing one encoder instead of three. Few-shot GPT-4o-mini matched the encoders on hate speech (79%) but lagged far behind on fake news (59%). SHAP attributions and adversarial probing reveal dataset-specific shortcuts in all three tasks. The fake-news model keys on political framing words, because DeFaktS encodes *framing* rather than factual falsity. The toxicity model keys on topic words such as *„Querdenker“*. The hate-speech model spreads weak evidence over many tokens when faced with GAHD's implicit, adversarial hate. We also document methodological pitfalls found when re-implementing the pipeline: oversampling before splitting, model selection on the test set, destructive text cleaning, and naive text-to-text handling of missing labels. We argue that they matter as much as the choice of architecture.

**Keywords:** fake news detection, hate speech, toxicity, multi-task learning, BERT, T5, large language models, German NLP, explainability, data leakage

---

## 1. Introduction

Social media now plays a central role in how people get news and debate politics. The same platforms carry content that degrades public discourse. False stories spread "farther, faster, deeper and more broadly than the truth" (Vosoughi et al., 2018). Hate speech targets people for protected characteristics, and toxic language drives users out of conversations. These harms are usually studied in isolation, but they interact. A fabricated story about a minority invites hateful comments, toxic replies escalate misinformation threads, and hateful posts often carry false claims about the groups they target.

Automated moderation is the only way to deal with this content at platform scale. The dominant engineering pattern is one classifier per harm. This is simple but costly: every post passes through several large models, and any knowledge shared across tasks is wasted. Multi-task learning (MTL) promises to share representations across related tasks (Caruana, 1997; Ruder, 2017). Instruction-tuned LLMs promise to drop task-specific training altogether. Which of these paradigms works best for *combined* harmful-content detection, especially outside English, has received little systematic study.

This paper combines a focused literature review with a case study carried out in the *TWON Metrics* research project at Universität Trier. It is organised around three research questions:

- **RQ1.** How do separate transformer models, each trained on one task, compare with a single unified model trained on all three tasks?
- **RQ2.** What can be concluded from combined detection, and how can it contribute to healthier online discourse?
- **RQ3.** How do encoder, encoder-decoder and decoder-only architectures compare on these tasks?

Our contributions are: (1) a structured review of the four paradigms; (2) an empirical comparison on three German datasets; (3) an analysis of *what* the fake-news models learn, using SHAP and adversarial examples; and (4) a set of methodological lessons from re-implementing the pipeline, with a corrected, open-source implementation.

---

## 2. Background and Related Work

### 2.1 Fake news and disinformation detection

Early fake-news detection relied on hand-crafted lexical, syntactic and stylistic features or on propagation patterns (Shu et al., 2017). Surveys separate *knowledge-based* approaches, which check claims against evidence, from *style-based* ones, which detect deceptive writing, and from *propagation-* and *source-based* approaches (Zhou & Zafarani, 2020). Transformer encoders now dominate style-based detection.

This distinction matters for interpreting results. A style-based classifier can only detect *how* something is written, not *whether* it is true. **DeFaktS** (Ashraf et al., 2024), the dataset used here, makes this explicit. It annotates German tweets and Telegram posts for disinformation through the lens of *social media framing* (cf. Entman, 1993), with fine-grained labels for indicators such as polarisation, emotional appeals and vagueness. Its authors found that German BERT performed best on both binary and fine-grained classification. DeFaktS therefore operationalises disinformation as a recognisable *communicative style*, an important point for Section 6.

### 2.2 Hate speech and toxicity detection

Hate-speech detection has followed the same path from lexicons and n-gram classifiers to neural models (Fortuna & Nunes, 2018). Two persistent problems are *spurious lexical cues*, where models over-rely on slurs or identity terms, and *bias*, such as over-flagging dialects associated with minority groups (Sap et al., 2019). Functional test suites such as HateCheck (Röttger et al., 2021) showed that high benchmark scores can hide brittle behaviour.

**GAHD** (Goldzycher et al., 2024) responds to this for German. It is built through *dynamic adversarial data collection*: annotators write examples that fool the current model. This yields about 11k hard, often implicit, examples on which even strong models struggle.

For toxicity and offensive language in German, the **GermEval** shared tasks are the standard benchmarks. GermEval 2018 covered offensive language (Wiegand et al., 2018), and GermEval 2021 covered toxic, engaging and fact-claiming comments (Risch et al., 2021). Our toxicity data (*DeTox*) draws on these resources. Toxicity and hate speech overlap, but they are not the same: toxicity covers rude or hostile language in general, while hate speech requires a target defined by a protected characteristic.

### 2.3 Transformer architectures

Since Vaswani et al. (2017), three transformer families have emerged:

- **Encoders** such as BERT (Devlin et al., 2019) build bidirectional contextual representations and are fine-tuned with a classification head. Relevant variants are **DistilBERT** (Sanh et al., 2019), a 40% smaller distilled model that keeps most of BERT's accuracy; the German **GBERT/German BERT** models (Chan et al., 2020); and the multilingual **mBERT** (Pires et al., 2019) and **XLM-RoBERTa** (Conneau et al., 2020).
- **Encoder-decoders** such as **T5** (Raffel et al., 2020) frame every task as text-to-text, so the same model can output "fake_news: 1 hate_speech: 0 toxicity: 1".
- **Decoder-only LLMs** (GPT-style; Brown et al., 2020) perform tasks from instructions and a few in-context examples, without any gradient updates.

### 2.4 Multi-task learning

MTL trains one model on several tasks so that shared structure acts as an inductive bias (Caruana, 1997). In NLP, the most common form is *hard parameter sharing*: one encoder with a separate head per task (Ruder, 2017). The benefits are positive transfer between related tasks and lower serving cost. The main risk is *negative transfer* or *task interference*, where tasks with conflicting gradients hurt each other. Which tasks benefit from being trained together is an empirical question (Standley et al., 2020).

A practical complication arises when tasks come from different datasets. Each example then has a label for only one task, so the loss must be *masked* to ignore missing labels.

### 2.5 Explainability

Moderation decisions affect speech, so explanations matter for trust and debugging. **SHAP** (Lundberg & Lee, 2017) attributes a prediction to input features using Shapley values. For text, it shows which words push a post towards a label, making it possible to check whether a model relies on meaningful cues or on artefacts.

---

## 3. Data

| Task | Dataset | Domain | Used size | Balance |
|---|---|---|---|---|
| Fake news | DeFaktS (Ashraf et al., 2024) | German tweets, political/climate/crisis topics | 5,000 (unified); full Twitter split for single-task | 50/50 |
| Hate speech | GAHD (Goldzycher et al., 2024) | Adversarially written German texts | 10,996 | 58/42 |
| Toxicity | DeTox / GermEval 2018, 2021 | German social media comments | 5,000 | 50/50 |
| **Unified** | Concatenation of the above | — | **15,000** | 5,000 per task |

*Table 1: Datasets. In the unified corpus each row carries one real label; the other two are set to −1 ("not annotated").*

**The three corpora.** *DeFaktS* tweets are short, topical (politics, climate, the 2021 Ahr flood, COVID-19) and full of hashtags and links; a "fake" label marks a post that uses disinformation framing. *GAHD* was collected over four rounds of human-versus-model adversarial writing. Its hateful examples are often implicit, for example *„Ich mag Sterne. Morgensterne, Abendsterne, Filmsternchen … Nur Judensterne nicht“*, and its non-hateful examples deliberately mention identity groups (*„Ich liebe Inder“*), so identity terms alone are not predictive. The *DeTox/GermEval* comments are long, heated replies from news and political discussions. Rude language (*„Scheißland“*, *„bekloppt“*) appears in both classes, which makes the toxic/non-toxic boundary fuzzy.

**Preprocessing.** In the original experiments, DeFaktS URLs were replaced by a `[URL]` token, and DeTox rows with missing text or labels were dropped. GAHD was lowercased and stripped of all digits, punctuation and emojis. The unified corpus additionally had e-mail addresses removed and emojis replaced. Minority classes were upsampled for DeFaktS and GAHD. The corrected pipeline applies the same *light* cleaning to every task: HTML unescaping, `[URL]` and `@user` placeholders, and whitespace normalisation. It keeps case (German capitalises all nouns), punctuation and emojis, because these carry signal such as sarcasm markers (🤥) in hate speech.

**Heterogeneity.** The three sources differ in genre, time period, annotation guidelines and length. GAHD's adversarial examples are deliberately subtle, while DeFaktS tweets are short, topical and full of hashtags and links. In the merged corpus, a model can partly infer *which dataset* a post comes from by its style, a shortcut we return to in Section 6.

---

## 4. Methods

### 4.1 Single-task fine-tuning

One encoder was fine-tuned per task, each with a linear classification head on the `[CLS]` representation:

- **Fake news:** `distilbert-base-german-cased` with class-weighted cross-entropy, dropout regularisation, learning rate 3e-5, weight decay, a cosine learning-rate schedule with warm-up, up to 5 epochs, and early stopping on F1.
- **Hate speech:** `xlm-roberta-base` (SentencePiece vocabulary covering 100 languages), learning rate 3e-5, weight decay 0.01, batch size 16, 5 epochs, with minority-class upsampling.
- **Toxicity:** `bert-base-multilingual-cased` with dropout raised to 0.2 (hidden and attention layers), a small learning rate (5e-6) with 500 warm-up steps, batch size 32, up to 10 epochs, and early stopping. The low learning rate proved the most stable setting for mBERT on noisy, long comments.

All inputs were truncated to 128 tokens.

### 4.2 Unified multi-task BERT

A single `bert-base-german-cased` encoder feeds three sigmoid heads (fake news, hate speech, toxicity) after dropout of 0.3. For each head, binary cross-entropy is computed only over rows whose label is not −1, and the task losses are summed. Training ran for 3 epochs with batch size 32 and Adam (learning rate 2e-5) on an 80/20 split.

### 4.3 Unified T5

`t5-base` was fine-tuned to generate the label string "fake_news: *a* hate_speech: *b* toxicity: *c*". At inference, the labels are parsed back out of the generated text with a regular expression.

### 4.4 LLM prompting

A system prompt defined the three tasks with explicit criteria. For fake news these were sensational claims, missing specifics or broken logic, emotionally viral wording, unsupported or contradictory claims, and fear-inducing language. For hate speech the key criterion was targeting by protected characteristic; for toxicity it was general hostility or vulgarity. The prompt also gave rules for hate/toxicity overlap. The model had to answer in the strict format `a,b,c`. The prompt was refined over six iterations and translated to target German input.

We evaluated Llama-3.1-8B and Llama-3.3-70B, Mistral-7B and Phi-4 (6-bit quantised, via the university's inference server), and GPT-4o-mini via API. GPT-4o-mini was run both zero-shot and few-shot, with two German examples per task. The temperature was 0.3.

### 4.5 Explainability and error analysis

We applied SHAP with a text masker to all three single-task classifiers (fake news, hate speech, toxicity) and to the unified model, on hand-written and test-set posts, and aggregated token attributions into feature-importance rankings. We also tested the unified model on hand-written adversarial German posts that separate factual falsity, hostility and group-targeting.

### 4.6 Metrics

We report accuracy, precision, recall and F1. For the binary tasks, F1 refers to the positive (harmful) class or to the weighted average, as reported per experiment.

---

## 5. Results

### 5.1 Single-task vs. unified models (RQ1)

| Model | Task | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| DistilBERT-German (single) | Fake news | 90% | 91% | 91% | **91.0%** |
| XLM-R (single) | Hate speech | 79% | 78% | 77.9% | 78.1% |
| mBERT (single) | Toxicity | 88.7% | 88.7% | 88.7% | 88.7% |
| Unified BERT | Fake news | 81% | 84% | —¹ | 85.0% |
| Unified BERT | Hate speech | 76% | 73% | 74% | 76.0% |
| Unified BERT | Toxicity | 90% | 89% | 90% | **91.0%** |
| Unified T5 | Fake news | 71% | 50% | 51% | 73.0% |
| Unified T5 | Hate speech | 54% | 48% | 42% | 68.0% |
| Unified T5 | Toxicity | 45% | 49% | 40% | 51.0% |

*Table 2: Fine-tuned models on held-out test data. ¹The originally reported F1 of 85% cannot be correct, since F1 is the harmonic mean of precision and recall and cannot exceed both. We therefore report accuracy for this cell.*

The answer to RQ1 depends on the task. **Fake news** clearly favours the specialist (91% vs. 85%). **Toxicity** favours the unified model (91% vs. 88.7%). **Hate speech** is nearly tied (78% vs. 76%) and is the hardest task for every fine-tuned model. Averaged over the three tasks, the single-task models score 85.9% and the unified BERT 84.0%: a 1.9-point gap, in exchange for running one encoder instead of three.

### 5.2 Prompted LLMs (RQ3)

| Model | Fake news | Hate speech | Toxicity |
|---|---|---|---|
| Llama-3.1-8B | 51% | 57% | 58% |
| Llama-3.3-70B | **60%** | 72% | 77% |
| Mistral-7B | 57% | 64% | 70% |
| Phi-4 (14B) | 58% | 71% | 68% |
| GPT-4o-mini, zero-shot | 40% | 46% | 46% |
| GPT-4o-mini, few-shot | 59% | **79%** | **78%** |

*Table 3: Accuracy of prompted LLMs.*

Three patterns stand out. First, **scale helps**: Llama 70B beats Llama 8B by 9 to 19 points. Second, **demonstrations help even more**: two examples per task raise GPT-4o-mini by 19 to 33 points. Its zero-shot run was the worst of all, below chance on fake news, probably because its outputs were systematically biased towards one class. Third, the best LLMs **match the fine-tuned encoders on hate speech** (79% vs. 78%) but **fall 31 points short on fake news**.

### 5.3 Summary across paradigms (RQ3)

| Paradigm | Fake news | Hate speech | Toxicity | Training data | Inference cost |
|---|---|---|---|---|---|
| Best single-task encoder | **91.0%** | 78.1% | 88.7% | per task | 3 models |
| Unified multi-task BERT | 85% | 76% | **91%** | merged | 1 model |
| Unified T5 | 73% | 68% | 51% | merged | 1 model, generative |
| Best prompted LLM | 60% | **79%** | 78% | none / few-shot | large model per post |

*Table 4: Best accuracy per paradigm.*

### 5.4 What do the models learn? SHAP and adversarial analysis

SHAP was applied to each single-task model on random test posts and to the unified model on one post per task.

**Fake news.** For the DistilBERT model, the strongest contributors to the "fake" class are tokens such as *„trigger“* and *„liberal“* (plus several sub-word fragments). These are words of **political framing**, not markers of factual error. In the unified model, the fake-news output reacts most to polarising compounds and hashtags in a post about migration policy: *„linksgrünen“*, *„Gutmenschen“*, *#Deutschland*, *„zerstören“*.

**Hate speech.** The XLM-R attributions on GAHD are an order of magnitude smaller (mean |SHAP| ≈ 0.002, vs. about 0.05–0.10 for fake news). The top-ranked pieces are function words, punctuation and harmless content words (*„Ich“*, *„schön“*, *„Morgen“*). For the implicit antisemitic post *„Ich mag Sterne … Nur Judensterne nicht“*, the evidence is spread over the whole sentence; no single slur exists to attach it to. This is exactly what GAHD's adversarial design aims for: the model cannot rely on lexical shortcuts, and when it is right, it is right because of composition and context. The lack of a dominant cue also explains why hate speech is the hardest task for every fine-tuned model.

**Toxicity.** The mBERT model attributes the largest positive value in a vulgar COVID-protest rant to *„Querdenker“* (the name of the anti-lockdown movement), more than to *„bekloppt“*. The explicit insult *„Scheiß(land)“* even pushes *against* toxicity. The model has learned a **topic shortcut**: in GermEval-style data, posts about the Querdenker debate are disproportionately toxic. Such cues hold within the dataset but fail on calm posts about the same topic, and they risk penalising a political topic instead of an abusive tone.

**Adversarial probing.** The unified model was also tested on hand-written German posts that separate factual falsity, hostility and group-targeting:

| Input (German) | Gold intuition | Prediction (F / H / T) |
|---|---|---|
| *Die Erde ist tatsächlich flach, alles andere ist eine Lüge der Wissenschaft.* | fake | 0 / 0 / 0 ✗ |
| *Diese Region sollte abgeschnitten und sich selbst überlassen werden.* | hostile, not fake | 1 / 0 / 0 ✗ |
| *Die Anhänger dieser Partei sind alle Idioten und gehören eingesperrt.* | toxic, not fake | 1 / 0 / 1 ✗ |
| *Diese Minderheit zerstört unser Land mit ihrer Kultur.* | hate | 1 / 1 / 0 (partial) |
| *Diese Sprache sollte in unserem Land verboten werden.* | hate + toxic | 0 / 1 / 0 (partial) |

*Table 5: Selected adversarial errors of the unified model (F = fake news, H = hate speech, T = toxicity).*

The model misses a calm, declarative conspiracy claim, yet flags aggressive political statements as "fake". It has learned that **polarised, us-versus-them language** signals disinformation. Its hate and toxicity heads are also confused. A group-targeted demand to ban a language is caught as hate but not as toxic. A dehumanising statement about a minority is caught as hate but also mislabelled as fake. Across all three tasks, then, the models detect *registers* (polarised, hostile, topical) more reliably than the harm each label is meant to capture.

---

## 6. Discussion

### 6.1 Why specialists win on fake news and sharing wins on toxicity

The results fit the transfer-versus-interference view of MTL. Toxicity and hate speech share surface features (insults, slurs, aggressive syntax), so a shared encoder trained on both gets more evidence for "hostile language" and toxicity improves. Fake news in DeFaktS depends on different cues: topic, framing, hedging, missing sources, links. When one encoder is optimised for all three objectives, the hostility signal dominates its representation, and the fake-news head inherits a feature space tuned for abuse. The adversarial errors show this directly: hostile posts leak into the fake-news head.

Two further factors probably widen the fake-news gap. First, the single-task model used a full DeFaktS Twitter split, while the unified model saw a 5,000-tweet subset. Second, there may be train/test leakage in the single-task result (Section 6.4). Both inflate the specialist's advantage. The true gap is likely smaller than 6 points.

**For practitioners (RQ1):** a unified model is a strong default when cost or latency matters. It loses little on average and gains on tasks with shared signals. A dedicated model is worth keeping for the task that is least like the others, which here is disinformation. Methods that soften interference, such as task-specific adapters, loss re-weighting, or gradient surgery, are natural next steps.

### 6.2 Hate speech and toxicity: hard boundaries, different shortcuts

Hate speech and toxicity look alike but behave differently. On GAHD, every fine-tuned model stays below 79%, and the unified model barely changes the score (−2 points). Adversarial data removes the lexical shortcuts that make other hate-speech benchmarks look easy (Röttger et al., 2021), so shared toxicity features cannot help much: much of GAHD's hate contains no insult at all. Telling *„Ich liebe Inder“* from *„Jeder weiß, dass Chinesen …“* requires knowing which statements about a group are demeaning, which is world knowledge more than surface form.

Toxicity is the opposite case. Its labels correlate with surface features (vulgarity, exclamation, insults) *and* with topics, so a shared encoder trained alongside hate speech sharpens a useful "hostility" representation. This explains the +2.3-point multi-task gain. The topic shortcut found by SHAP (*„Querdenker“*) is the risk on this side: toxicity models trained on debate comments may mark entire political discussions as toxic. Both tasks therefore call for evaluation beyond aggregate accuracy, with balanced counter-examples that mention identity groups or contested topics in a neutral tone.

### 6.3 Why LLMs struggle with DeFaktS but not with hate speech

Hate speech has a widely shared definition, so an instruction-tuned LLM can apply it from a prompt. "Disinformation" in DeFaktS is a *dataset-specific construct*: a pattern of framing identified by annotators under a particular codebook. Without labelled examples, the LLM falls back on its own idea of fake news, namely *factual falsehood*. Short tweets offer it little to check, and many framed but factually unremarkable posts carry the "fake" label. Fine-tuned encoders learn the annotators' construct directly from thousands of examples. This also explains why few-shot examples help so much: they partly convey the construct.

The broader lesson for RQ3 is that **prompting suits tasks with a stable, shared definition; fine-tuning suits tasks defined by an annotation scheme.** A promising middle ground, left for future work, is to use the LLM to *produce* fine-grained framing indicators (polarisation, vagueness, missing sources, as in the original project proposal) as features for a lightweight classifier.

### 6.4 Methodological pitfalls we found

Re-implementing the pipeline (code released with this paper) turned up issues that affect the reported numbers. Each is common enough in applied NLP to be worth naming.

1. **Oversampling before splitting.** The fake-news and hate-speech notebooks oversampled the minority class *and then* split into train and test. Duplicated tweets therefore appear on both sides, and the model is partly tested on examples it has memorised. This is a textbook case of *data leakage* (Kaufman et al., 2012), which Kapoor and Narayanan (2023) identify as a widespread cause of irreproducible machine-learning results. The fix is to split first and rebalance only the training data; our implementation also removes exact duplicates and tweets with conflicting labels.
2. **Model selection on the test set.** In all three single-task models, early stopping and "best checkpoint" selection used the evaluation split that was later reported as the test result. This biases the score upwards. The corrected pipeline uses separate train, validation and test splits and evaluates on the test split once.
3. **Destructive cleaning.** Lowercasing GAHD and deleting all digits, punctuation and emojis removed real signal (sarcasm emojis, German noun capitalisation, quotation marks that signal irony). It also made hate-speech preprocessing inconsistent with the other two tasks. Cased transformer tokenizers need no such normalisation.
4. **Missing labels in text-to-text training.** For T5, the −1 placeholder was written into the target string ("fake_news: -1 …"). The model was literally trained to *generate* "-1" for two of the three tasks in every example, and evaluation did not exclude those rows. In addition, padding tokens in the targets were not masked from the loss, and the learning-rate schedule was configured for 7 epochs but training stopped after 4. The English-only `t5-base` vocabulary is also a poor fit for German. Together, these issues (not the encoder-decoder architecture itself) best explain T5's weak results. The released code therefore builds one task-prefixed example per *labelled* task ("hate_speech: … → 1"), masks target padding, and uses multilingual mT5.
5. **Silent defaults in LLM parsing.** Unparseable LLM answers were mapped to class 0. This inflates accuracy on balanced data whenever a model fails to follow the format, and makes models less comparable. Invalid answers should be reported separately.
6. **Metric reporting.** One table cell reported an F1 above both precision and recall, which is mathematically impossible. Automatically generated result tables avoid such transcription errors.

None of these issues changes the qualitative conclusions. They do mean the exact numbers in Tables 2 to 4 should be read as optimistic for the single-task fake-news and hate-speech models and pessimistic for T5.

### 6.5 Combined detection and healthier discourse (RQ2)

Combined detection gives a richer signal than three separate binary flags. A post that is both *framed as disinformation* and *hateful* is a different moderation case from one that is merely rude. Joint outputs support graduated responses: adding context labels or fact-check links for disinformation, de-ranking toxic replies, and escalating group-targeted hate for human review. A single shared model also lowers the cost of screening at scale.

Our findings also argue for caution. A disinformation detector that keys on polarised framing will disproportionately flag *legitimate* political criticism and activist speech, the very speech democratic discourse needs. Combined with known biases in hate-speech classifiers (Sap et al., 2019), this means such systems should *assist* human moderators, offering explanations (for example, SHAP highlights), rather than remove content automatically. Engagement improves only if users trust the moderation, and explainability and appeal mechanisms are part of that trust.

---

## 7. Ethical Considerations

The data contains real social media posts, some of them hateful. Sharing it must follow the dataset licences and the platforms' terms of service, so our repository does not redistribute it. Labels such as "fake" carry a risk of censorship when applied to contested political speech, and errors are not neutral: false positives silence users, false negatives expose them to harm. Hand-written adversarial examples, as in Table 5, are a cheap way to probe these failure modes before deployment. They should become part of standard evaluation, in the spirit of HateCheck (Röttger et al., 2021).

---

## 8. Limitations and Future Work

- **Single runs and small test sets.** No confidence intervals or significance tests are reported. Several differences (for example, hate speech 78% vs. 76%) are within plausible run-to-run variance.
- **Unequal data.** The single-task and unified fake-news models saw different amounts of DeFaktS data.
- **Leakage.** The single-task fake-news and hate-speech figures need re-running with the leakage-safe pipeline. The released code does this by default for all three tasks.
- **Binary labels only.** DeFaktS's fine-grained framing categories were not used. Predicting them, or using LLM-extracted indicators, could make the fake-news task both more accurate and more interpretable.
- **Architectures.** Stronger German encoders (GBERT-large, GELECTRA), adapter-based MTL, and instruction-tuned open LLMs fine-tuned with LoRA would give a fairer comparison of paradigms.
- **Deployment.** Latency optimisation, multilingual extension and evaluation with real moderators remain future work.

---

## 9. Conclusion

We reviewed four paradigms for detecting fake news, hate speech and toxicity, and compared them on German social media data. There is no single winner. Specialised encoders are most accurate on the task least like the others (disinformation, 91%). A unified multi-task BERT is nearly as good on average and best on toxicity (91%), with a third of the serving cost. Prompted LLMs are competitive only where the task definition is universal (hate speech, 79%). The explainability analysis shows that every model learns dataset-specific registers. "Disinformation detection" on framing-annotated data is really *framing detection*, toxicity models pick up contested *topics*, and adversarial hate speech resists lexical shortcuts altogether. This shapes both what models can achieve and how they should be used. Finally, the methodological lessons (leakage-safe splitting, a clean validation protocol, careful handling of missing labels, and honest reporting of invalid outputs) matter as much as the architecture choice. The corrected implementation is available at <https://github.com/farhanaliarshad1/german-harmful-content-detection>.

**Recommendation:** use single-task models where accuracy on one harm is critical, multi-task models where efficiency across harms matters, and LLM prompting for rapid prototyping and for tasks with clear, shared definitions.

---

## Acknowledgements

This work was carried out as a team project in the Research Case Studies course at Universität Trier, within the TWON research context. I thank my fellow students Aastha Shreya, Shivang Sinha, Syed Huzaifa Ali, Donya Sheikhi and Muhammad Ahmed for the collaboration on the experiments reviewed here. The original team repository is available at <https://github.com/Aastha031295/TWON_Metrics>.

---

## References

Ashraf, S., Bezzaoui, I., Andone, I., Markowetz, A., Fegert, J., & Flek, L. (2024). DeFaktS: A German dataset for fine-grained disinformation detection through social media framing. In *Proceedings of LREC-COLING 2024* (pp. 4580–4591).

Brown, T., et al. (2020). Language models are few-shot learners. In *Advances in Neural Information Processing Systems 33*.

Caruana, R. (1997). Multitask learning. *Machine Learning, 28*(1), 41–75.

Chan, B., Schweter, S., & Möller, T. (2020). German's next language model. In *Proceedings of COLING 2020* (pp. 6788–6796).

Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of ACL 2020*.

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of NAACL-HLT 2019*.

Entman, R. M. (1993). Framing: Toward clarification of a fractured paradigm. *Journal of Communication, 43*(4), 51–58.

Fortuna, P., & Nunes, S. (2018). A survey on automatic detection of hate speech in text. *ACM Computing Surveys, 51*(4), 1–30.

Goldzycher, J., Röttger, P., & Schneider, G. (2024). Improving adversarial data collection by supporting annotators: Lessons from GAHD, a German hate speech dataset. In *Proceedings of NAACL 2024* (pp. 4405–4424).

Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns, 4*(9), 100804.

Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data, 6*(4), 1–21.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30*.

Pires, T., Schlinger, E., & Garrette, D. (2019). How multilingual is multilingual BERT? In *Proceedings of ACL 2019*.

Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research, 21*(140), 1–67.

Risch, J., Stoll, A., Wilms, L., & Wiegand, M. (2021). Overview of the GermEval 2021 shared task on the identification of toxic, engaging, and fact-claiming comments. In *Proceedings of the GermEval 2021 Workshop* (KONVENS 2021).

Röttger, P., Vidgen, B., Nguyen, D., Waseem, Z., Margetts, H., & Pierrehumbert, J. (2021). HateCheck: Functional tests for hate speech detection models. In *Proceedings of ACL-IJCNLP 2021*.

Ruder, S. (2017). An overview of multi-task learning in deep neural networks. *arXiv:1706.05098*.

Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv:1910.01108*.

Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The risk of racial bias in hate speech detection. In *Proceedings of ACL 2019*.

Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). Fake news detection on social media: A data mining perspective. *ACM SIGKDD Explorations Newsletter, 19*(1), 22–36.

Standley, T., Zamir, A., Chen, D., Guibas, L., Malik, J., & Savarese, S. (2020). Which tasks should be learned together in multi-task learning? In *Proceedings of ICML 2020*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30*.

Vosoughi, S., Roy, D., & Aral, S. (2018). The spread of true and false news online. *Science, 359*(6380), 1146–1151.

Wiegand, M., Siegel, M., & Ruppenhofer, J. (2018). Overview of the GermEval 2018 shared task on the identification of offensive language. In *Proceedings of GermEval 2018* (KONVENS 2018).

Zhou, X., & Zafarani, R. (2020). A survey of fake news: Fundamental theories, detection methods, and opportunities. *ACM Computing Surveys, 53*(5), 1–40.
