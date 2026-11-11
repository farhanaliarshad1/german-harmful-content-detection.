# Data

Raw data is **not** committed. Place the CSV files in this folder. All texts are **German**.

| File | Task | Source | Columns | Size |
|---|---|---|---|---|
| `fake_news.csv` | Fake news / disinformation | [DeFaktS](https://aclanthology.org/2024.lrec-main.409/) (Ashraf et al., LREC-COLING 2024): German tweets annotated for disinformation framing | `text`, `binary_label` (1 = fake) | 5,000 (balanced) |
| `hateSpeech.csv` | Hate speech | [GAHD](https://aclanthology.org/2024.naacl-long.248) (Goldzycher et al., NAACL 2024): adversarially collected German hate speech ([data](https://github.com/jagol/gahd), CC BY 4.0) | `text`, `label` (1 = hate) | 10,996 (58 / 42) |
| `toxicity.csv` | Toxicity | DeTox and GermEval 2018 / 2021 German offensive and toxic comments | `text`, `toxic` (1 = toxic) | 5,000 (balanced) |
| `merged_dataset.csv` | Unified (all three) | Concatenation of the three files above | `text`, `isFake`, `isHateSpeech`, `isToxic` (`-1` = not annotated for that task) | 15,000 |

Respect each dataset's licence and the platforms' terms of service. DeFaktS is distributed by its authors for research use.
