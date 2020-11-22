# HeBERT: Pre-training BERT for modern Hebrew
HeBERT is an Hebrew pretrained lanaguage model based on Google's BERT architechture and it's BERT-Base config. 

BERT is a bidirectional model that is based on the transformer architecture, it replaces the sequential nature of RNN (LSTM & GRU) with a much faster Attention-based approach \cite{devlin2018bert}. ROBERTA builds on BERT and modifies key hyperparameters, and training with much larger mini-batches and learning rates \cite{liu2019roberta}. 

We train a BERT model and tokenizer on OSCAR corpus, which is a huge multilingual corpus obtained by language classification and filtering of the Common Crawl corpus using the goclassy architecture. For Hebrew language it has 9.8 GB of corpus, including 1 billion words and over 20.8 millions sentences (after deduplicate the original data) (\cite{ortiz-suarez-etal-2020-monolingual}. We also train it on all user generated content we have and Hebrew Wikipedia's dump (630 MB, 3833141 sentences).

We evalaute the model on emotion recognition and sentiment analysis (based on Amram, A., Ben-David, A., and Tsarfaty, R. (2018) dataset), for a downstream tasks and masked fill-in-the-blank task (the main task of our model).

# Results
## emotion recognition 
|              |           |      |      |             |      |      |       |      |      |       |      |      |      |      |      |          |      |      |         |      |      |         |      |      |       |      |      |
|:-------------|----------:|-----:|-----:|------------:|-----:|-----:|------:|-----:|-----:|------:|-----:|-----:|-----:|-----:|-----:|---------:|-----:|-----:|--------:|-----:|-----:|--------:|-----:|-----:|------:|-----:|-----:|
| emotion\_en  | sentiment |      |      | expectation |      |      | happy |      |      | trust |      |      | fear |      |      | surprise |      |      | sadness |      |      | disgust |      |      | anger |      |      |
| index        |         p |    r |   f1 |           p |    r |   f1 |     p |    r |   f1 |     p |    r |   f1 |    p |    r |   f1 |        p |    r |   f1 |       p |    r |   f1 |       p |    r |   f1 |     p |    r |   f1 |
| 0            |      0.96 | 0.93 | 0.94 |        0.85 | 0.81 | 0.83 |  0.98 | 0.98 | 0.98 |  0.96 | 0.99 | 0.97 | 0.77 | 0.84 | 0.81 |     0.84 | 0.89 | 0.86 |    0.71 | 0.70 | 0.70 |    0.73 | 0.79 | 0.76 |  0.88 | 0.88 | 0.88 |
| 1            |      0.83 | 0.89 | 0.86 |        0.83 | 0.87 | 0.85 |  0.89 | 0.87 | 0.88 |  0.88 | 0.70 | 0.78 | 0.84 | 0.77 | 0.80 |     0.47 | 0.37 | 0.41 |    0.83 | 0.84 | 0.84 |    0.97 | 0.95 | 0.96 |  0.97 | 0.97 | 0.97 |
| weighted avg |      0.92 | 0.92 | 0.92 |        0.84 | 0.84 | 0.84 |  0.97 | 0.97 | 0.97 |  0.95 | 0.95 | 0.95 | 0.81 | 0.80 | 0.80 |     0.76 | 0.78 | 0.77 |    0.79 | 0.79 | 0.79 |    0.93 | 0.93 | 0.93 |  0.95 | 0.95 | 0.95 |
| accuracy     |      0.92 | 0.92 | 0.92 |        0.84 | 0.84 | 0.84 |  0.97 | 0.97 | 0.97 |  0.95 | 0.95 | 0.95 | 0.80 | 0.80 | 0.80 |     0.78 | 0.78 | 0.78 |    0.79 | 0.79 | 0.79 |    0.93 | 0.93 | 0.93 |  0.95 | 0.95 | 0.95 |


based on comments scarped from 3 big Israeli news-papaers sites we have annotated

## sentiment analysis


# How to use

# If you used this model please cite us as :
