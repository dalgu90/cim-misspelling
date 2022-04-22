# cim-misspelling

Pytorch implementation of Context-Sensitive Spelling Correction of Clinical Text via Conditional Independence, CHIL 2022.

<img width="860" alt="image" src="https://user-images.githubusercontent.com/13655756/158745297-e899feb8-e023-4070-b1c1-eda3779aa4c4.png">

This model (CIM) corrects misspellings with a char-based language model and a corruption model (edit distance).
The model is being pre-trained and evaluated on clinical corpus and datasets.
Please see the [paper](https://proceedings.mlr.press/v174/kim22b/kim22b.pdf) for more detailed explanation.


## Requirements

- Python 3.8 and packages in `requirements.txt`
- The MIMIC-III dataset (v1.4): [PhysioNet link](https://physionet.org/)
- BlueBERT: [GitHub link](https://github.com/ncbi-nlp/bluebert)
- The SPECIALIST Lexicon of UMLS: [LSG website](https://lhncbc.nlm.nih.gov/LSG/Projects/lexicon/current/web/release/)
- English dictionary (DWYL): [GitHub link](https://github.com/dwyl/english-words)


## How to Run

### Clone the repo

```
$ git clone --recursive https://github.com/dalgu90/cim-misspelling.git
```

### Data preparing

1. Download the MIMIC-III dataset from [PhysioNet](https://physionet.org/), especially `NOTEEVENTS.csv` and put under `data/mimic3`

2. Download `LRWD` and `prevariants` of the SPECIALIST Lexicon from the [LSG website](https://lhncbc.nlm.nih.gov/LSG/Projects/lexicon/current/web/release/) (2018AB version) and put under `data/umls`.

3. Download the English dictionary `english.txt` from [here](https://github.com/dwyl/english-words/tree/7cb484da5de560c11109c8f3925565966015e5a9) (commit 7cb484d) and put under `data/english_words`.

4. Run `scripts/build_vocab_corpus.ipynb` to build the dictionary and split the MIMIC-III notes into files.

5. Run the Jupyter notebook for the dataset that you want to download/pre-process:
    - MIMIC-III misspelling dataset, or [ClinSpell](https://github.com/clips/clinspell) (Fivez et al., 2017): `scripts/preprocess_clinspell.ipynb`
    - [CSpell](https://lsg3.nlm.nih.gov/LexSysGroup/Projects/cSpell/current/web/index.html) dataset (Lu et al., 2019): `scripts/preprocess_cspell.ipynb`
    - Synthetic misspelling dataset from the MIMIC-III: `scripts/synthetic_dataset.ipynb`

6. Download the BlueBERT model from [here](https://github.com/ncbi-nlp/bluebert) under `bert/ncbi_bert_{base|large}`.
    - For CIM-Base, please download "BlueBERT-Base, Uncased, PubMed+MIMIC-III"
    - For CIM-Large, please download "BlueBERT-Large, Uncased, PubMed+MIMIC-III"

### Pre-training the char-based LM on MIMIC-III

Please run `pretrain_cim_base.sh` (CIM-Base) or `pretrain_cim_large.sh`(CIM-Large) and to pretrain the character langauge model of CIM.
The pre-training will evaluate the LM periodically by correcting synthetic misspells generated from the MIMIC-III data.
You may need 2~4 GPUs (XXGB+ GPU memory for CIM-Base and YYGB+ for CIM-Large) to pre-train with the batch size 256.
There are several options you may want to configure:
- `num_gpus`: number of GPUs
- `batch_size`: batch size
- `training_step`: total number of steps to train
- `init_ckpt`/`init_step`: the checkpoint file/steps to resume pretraining
- `num_beams`: beam search width for evaluation
- `mimic_csv_dir`: directory of the MIMIC-III csv splits
- `bert_dir`: directory of the BlueBERT files

You can also download the pre-trained LMs and put under `model/`:
- [CIM-Base (12-layer)](#)
- [CIM-Large (24-layer)](#)

### Misspelling Correction with CIM

Please specify the dataset dir and the file to evaluate in the evaluation script (`eval_cim_base.sh` or `eval_cim_large.sh`), and run the script.  
You may want to set `init_step` to specify the checkpoint you want to load


## Cite this work

```
@InProceedings{juyong2022context,
  title = {Context-Sensitive Spelling Correction of Clinical Text via Conditional Independence},
  author = {Kim, Juyong and Weiss, Jeremy C and Ravikumar, Pradeep},
  booktitle = {Proceedings of the Conference on Health, Inference, and Learning},
  pages = {234--247},
  year = {2022},
  volume = {174},
  series = {Proceedings of Machine Learning Research},
  month = {07--08 Apr},
  publisher = {PMLR}
}
```
