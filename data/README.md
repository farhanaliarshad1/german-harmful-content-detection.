# Data

Raw data is **not** committed. Place the CSV files here:

| File | Source | Expected columns |
|---|---|---|
| `fake_news.csv` | [DeFaktS](https://aclanthology.org/2024.lrec-main.409/) (Ashraf et al., LREC-COLING 2024) (German tweets, disinformation framing) | `text`, `binary_label` (1 = fake / disinformation, 0 = neutral) |
| `merged_dataset.csv` *(optional, multi-task)* | DeFaktS + [GAHD](https://aclanthology.org/2024.naacl-long.248) ([data](https://github.com/jagol/gahd), CC BY 4.0) + DeTox (GermEval 2018/2021) | `text`, `isFake`, `isHateSpeech`, `isToxic` (`-1` = not annotated for that task) |

DeFaktS is distributed by its authors for research use. Request access from them and respect their licence and the platform's terms of service.
