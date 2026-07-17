# MagarKura
This repository contains a neural machine translation (NMT) model for bidirectional translation between Nepali (`np`) and Magar (`mgr`) languages. The model is built using the Fairseq toolkit and utilizes a Transformer architecture.

The project includes the complete training pipeline, from data preprocessing to model evaluation, as well as a FastAPI application to serve the trained model for single-sentence and bulk translations.

## Features

- **Bidirectional Translation**: Translate from Nepali to Magar and Magar to Nepali using the same model, controlled by special tokens (`<2mgr>` and `<2np>`).
- **REST API**: A simple FastAPI server (`app.py`) to expose translation capabilities.
- **Single Sentence Translation**: Endpoint for translating individual sentences.
- **Bulk Translation**: Endpoint to translate a `.txt` file (one sentence per line) and receive a downloadable `.csv` with the results.

## API

The included FastAPI application allows you to easily interact with the trained model.

>[!Note]
> checkpoint.pt file should be available for inference api, follow the steps from [Replicating Training](#Replicating-Training) or take a look at [magarkura_v3.ipynb](magarkura_v3.ipynb) script.

### 1. Setup

First, clone the repository and set up a Python virtual environment.

```bash
git clone https://github.com/raytsm/MagarKura.git
cd MagarKura
python -m venv .venv
source .venv/bin/activate
```

Next, install the required dependencies. Note that this requires `torch` and a specific fork of `fairseq`.

```bash
pip install -r requirements.txt
```

### 2. Run the Server

Launch the web server using Uvicorn. The `--reload` flag will automatically restart the server when you make changes to the code.

```bash
uvicorn app:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### 3. API Endpoints

#### Single Sentence Translation

**Endpoint**: `/translate` (POST)

- **`text`**: The sentence to translate.
- **`direction`**: `np2mgr` for Nepali to Magar, `mgr2np` for Magar to Nepali.

**Example (Nepali to Magar):**

```bash
curl -X POST "http://127.0.0.1:8000/translate" \
-H "Content-Type: application/json" \
-d '{
    "text": "मेरो नाम रोशन हो।",
    "direction": "np2mgr"
}'
```
Response:
```json
{
  "translation": "ङौ म्यार्मिन रोशन आले।"
}
```

**Example (Magar to Nepali):**

```bash
curl -X POST "http://127.0.0.1:8000/translate" \
-H "Content-Type: application/json" \
-d '{
    "text": "ङौ म्यार्मिन रोशन आले।",
    "direction": "mgr2np"
}'
```
Response:
```json
{
  "translation": "मेरो नाम रोशन हो।"
}
```

#### Bulk File Translation

**Endpoint**: `/bulk_translate` (POST)

Accepts a `.txt` file with one sentence per line and returns a `.csv` file containing the original and translated sentences.

- **`file`**: The `.txt` file to upload.
- **`direction`**: `np2mgr` or `mgr2np`.

**Example:**

Create a file named `nepali.txt` with the following content:
```
मेरो नाम रोशन हो।
म काठमाडौंमा बस्छु।
आज मौसम धेरै राम्रो छ।
```

Then run the following command:

```bash
curl -X POST "http://127.0.0.1:8000/bulk_translate" \
-F "file=@nepali.txt" \
-F "direction=np2mgr" \
-o translations.csv
```

This will save a `translations.csv` file with the input and translated sentences.

## Replicating Training

The entire training process is documented in the `magarkura_v3.ipynb` notebook. The key steps to replicate the model are outlined below.

### 1. Prerequisites

Install the necessary libraries for data processing and model training.

```bash
pip install -q fairseq sacrebleu sentencepiece
pip install indic-nlp-library
pip install torch==2.4.0 --quiet
pip install git+https://github.com/One-sixth/fairseq.git
```

### 2. Data Preparation

- **Normalization**: The `magar-nepali.csv` dataset is normalized using `indic-nlp-library`.
- **Splitting**: The normalized data is shuffled and split into training (80%), validation (10%), and test (10%) sets.

### 3. Tokenization

A SentencePiece (BPE) model with a vocabulary size of 1200 is trained on the combined Nepali and Magar text. Special tokens `<2np>` and `<2mgr>` are included to handle bidirectional translation.

### 4. Fairseq Preprocessing

The tokenized text data is binarized for efficient training with Fairseq using the following command:

```bash
fairseq-preprocess \
--source-lang np \
--target-lang mgr \
--trainpref data/train --validpref data/valid --testpref data/test \
--destdir data-bin \
--joined-dictionary \
--srcdict fairseq_dict.txt \
--workers 8
```

### 5. Model Training

The Transformer model is trained for 100 epochs using the `fairseq-train` command:

```bash
fairseq-train data-bin \
--source-lang np --target-lang mgr \
--arch transformer_iwslt_de_en \
--share-decoder-input-output-embed \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 1.0 \
--lr 5e-4 --lr-scheduler inverse_sqrt \
--warmup-updates 2000 \
--dropout 0.3 --weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--max-tokens 4000 \
--max-epoch 100 \
--patience 10 \
--save-dir checkpoints \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok space \
--eval-bleu-remove-bpe sentencepiece \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--fp16
```

### 6. Evaluation

The best-performing model is used to generate translations for the test set, and the BLEU score is calculated.

```bash
fairseq-generate data-bin --path checkpoints/checkpoint_best.pt \
--batch-size 64 --beam 5 --remove-bpe=sentencepiece \
--source-lang np --target-lang mgr --gen-subset test > test_output.txt
```

## Repository Structure

- `app.py`: The FastAPI application for serving the translation model.
- `magarkura_v3.ipynb`: Jupyter notebook containing the entire workflow from data preparation to training and evaluation.
- `magar-nepali.csv`: The original parallel corpus (not included, referenced in notebook).
- `output-normalized.csv`: The normalized dataset used for training.
- `requirements.txt`: Python dependencies for the FastAPI application.
- `/data-bin`: Binarized Fairseq data files.
- `spm-twelve-hundred.model` / `.vocab`: The trained SentencePiece tokenizer model and its vocabulary.
- `dict.np.txt` / `dict.mgr.txt`: Vocabulary dictionaries for Fairseq.
- `checkpoints/checkpoint_best.pt`: The best trained model checkpoint (not included, generated during training).
- `/figures`: Training performance plots.
- `test_output.txt`, `results.csv`, `translations.csv`: Example output files.
