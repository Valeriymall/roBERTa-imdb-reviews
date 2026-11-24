# IMDB sentiment classification with RoBERTa

Fine-tunes `roberta-base` to classify movie reviews as positive or negative.

**Test accuracy: 95.26%** on the full 25k IMDB test set.

![Training curves](BERT/assets/roberta_accuracy.png)

## Setup

- Model: `roberta-base`, max sequence length 512, 2 epochs, learning rate 2e-5
- Data: [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb), with 10% of the training set held out for validation
- Best checkpoint chosen on validation accuracy only, so the test set was not used for selection
- Trained on an AMD RX 9070 XT (ROCm)

## Files

- `BERT/train_imdb.py`: training, evaluation and plot
- `BERT/predict.py`: classify a single review

## Usage

Install PyTorch for your hardware first (see pytorch.org), then:

```bash
pip install transformers datasets evaluate accelerate matplotlib
python BERT/train_imdb.py
python BERT/predict.py "This movie is atrocious."
```

Training writes the model to `BERT/results/`, which is not included in this repo.
