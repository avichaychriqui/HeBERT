# HeBERT: Pre-trained BERT for Polarity Analysis and Emotion Recognition
<img align="right" src="https://github.com/avichaychriqui/HeBERT/blob/main/data/heBERT_logo.png?raw=true" width="250">

HeBERT is a Hebrew pretrained language model. It is based on [Google's BERT](https://arxiv.org/abs/1810.04805) architecture and it is BERT-Base config. <br>

HeBert was trained on three dataset: 
1. A Hebrew version of [OSCAR](https://oscar-corpus.com/): ~9.8 GB of data, including 1 billion words and over 20.8 millions sentences. 
2. A Hebrew dump of [Wikipedia](https://dumps.wikimedia.org/): ~650 MB of data, including over 63 millions words and 3.8 millions sentences
3. Emotion User Generated Content (UGC) data that was collected for the purpose of this study (described below).
<br>
<div>
We evaluated the model on downstream tasks: emotions recognition and sentiment analysis. 

## Emotion UGC Data Description
Our UGC data include comments posted on news articles collected from 3 major Israeli news sites, between January 2020 to August 2020. The total size of the data is ~150 MB, including over 7 millions words and 350K sentences.
~2000 sentences were annotated by crowd members (3-10 annotators per sentence) for overall sentiment (polarity) and [eight emotions](https://en.wikipedia.org/wiki/Robert_Plutchik#Plutchik's_wheel_of_emotions): anger, disgust, expectation , fear, happy, sadness, surprise and trust. 
The percentage of sentences in which each emotion appeared is found in the table below.

|       | anger | disgust | expectation | fear | happy | sadness | surprise | trust | sentiment |
|------:|------:|--------:|------------:|-----:|------:|--------:|---------:|------:|-----------|
| **ratio** |  0.78 |    0.83 |        0.58 | 0.45 |  0.12 |    0.59 |     0.17 |  0.11 | 0.25      |

## Performance
### Emotion Recognition
| emotion     | f1-score | precision | recall   |
|-------------|----------|-----------|----------|
|       anger | 0.96 |  0.99 | 0.93 |
|     disgust | 0.97 |  0.98 | 0.96 |
| expectation | 0.82 |  0.80 | 0.87 |
|        fear | 0.79 |  0.88 | 0.72 |
|       happy | 0.90 |  0.97 | 0.84 |
|     sadness | 0.90 |  0.86 | 0.94 |
|   sentiment | 0.88 |  0.90 | 0.87 |
|    surprise | 0.40 |  0.44 | 0.37 |
|       trust | 0.83 |  0.86 | 0.80 |

*The above metrics for positive class (meaning, the emotion is reflected in text).*

### Sentiment (Polarity) Analysis
|              | precision | recall | f1-score |
|--------------|-----------|--------|----------|
| natural      | 0.83      | 0.56   | 0.67     |
| positive     | 0.96      | 0.92   | 0.94     |
| negative     | 0.97      | 0.99   | 0.98     |
| accuracy     |           |        | 0.97     |
| macro avg    | 0.92      | 0.82   | 0.86     |
| weighted avg | 0.96      | 0.97   | 0.96     |

## How to use
### For Emotion Recognition Model
An online model can be found at [huggingface spaces](https://huggingface.co/spaces/avichr/HebEMO_demo) or as [colab notebook](https://colab.research.google.com/drive/1Jw3gOWjwVMcZslu-ttXoNeD17lms1-ff?usp=sharing)
```
# !pip install pyplutchik==0.0.7
# !pip install transformers==4.14.1

!git clone https://github.com/avichaychriqui/HeBERT.git
from HeBERT.src.HebEMO import *
HebEMO_model = HebEMO()

HebEMO_model.hebemo(input_path = 'data/text_example.txt')
# return analyzed pandas.DataFrame  

hebEMO_df = HebEMO_model.hebemo(text='החיים יפים ומאושרים', plot=True)
```
<img src="https://github.com/avichaychriqui/HeBERT/blob/main/data/hebEMO1.png?raw=true" width="300" height="300" />


	
### For masked-LM model (can be fine-tunned to any down-stream task)
	from transformers import AutoTokenizer, AutoModel
	tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
	model = AutoModel.from_pretrained("avichr/heBERT")
	
	from transformers import pipeline
	fill_mask = pipeline(
	    "fill-mask",
	    model="avichr/heBERT",
	    tokenizer="avichr/heBERT"
	)
	fill_mask("הקורונה לקחה את [MASK] ולנו לא נשאר דבר.")

### For sentiment classification model (polarity ONLY):
	from transformers import AutoTokenizer, AutoModel, pipeline
	tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_sentiment_analysis") #same as 'avichr/heBERT' tokenizer
	model = AutoModel.from_pretrained("avichr/heBERT_sentiment_analysis")
	
	# how to use?
	sentiment_analysis = pipeline(
	    "sentiment-analysis",
	    model="avichr/heBERT_sentiment_analysis",
	    tokenizer="avichr/heBERT_sentiment_analysis",
	    return_all_scores = True
	)
	
	sentiment_analysis('אני מתלבט מה לאכול לארוחת צהריים')	
	>>>  [[{'label': 'natural', 'score': 0.9978172183036804},
	>>>  {'label': 'positive', 'score': 0.0014792329166084528},
	>>>  {'label': 'negative', 'score': 0.0007035882445052266}]]

	sentiment_analysis('קפה זה טעים')
	>>>  [[{'label': 'natural', 'score': 0.00047328314394690096},
	>>>  {'label': 'possitive', 'score': 0.9994067549705505},
	>>>  {'label': 'negetive', 'score': 0.00011996887042187154}]]

	sentiment_analysis('אני לא אוהב את העולם')
	>>>  [[{'label': 'natural', 'score': 9.214012970915064e-05}, 
	>>>  {'label': 'possitive', 'score': 8.876807987689972e-05}, 
	>>>  {'label': 'negetive', 'score': 0.9998190999031067}]]

	
Our model is also available on AWS! for more information visit [AWS' git](https://github.com/aws-samples/aws-lambda-docker-serverless-inference/tree/main/hebert-sentiment-analysis-inference-docker-lambda)

## Stay tuned!
We are still working on our model and will edit this page as we progress. <br>
Note that we have released only sentiment analysis (polarity) at this point, emotion detection will be released later on.

## Contact us
[Avichay Chriqui](mailto:avichayc@mail.tau.ac.il) <br>
[Inbal yahav](mailto:inbalyahav@tauex.tau.ac.il) <br>
The Coller Semitic Languages AI Lab <br>
Thank you, תודה, شكرا <br>

## If you used this model please cite us as :
Chriqui, A., & Yahav, I. (2021). HeBERT & HebEMO: a Hebrew BERT Model and a Tool for Polarity Analysis and Emotion Recognition. arXiv preprint arXiv:2102.01909.
```
@article{chriqui2021hebert,
  title={HeBERT \& HebEMO: a Hebrew BERT Model and a Tool for Polarity Analysis and Emotion Recognition},
  author={Chriqui, Avihay and Yahav, Inbal},
  journal={arXiv preprint arXiv:2102.01909},
  year={2021}
}
```
