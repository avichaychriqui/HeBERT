# HeBERT: Pre-trained BERT for Polarity Analysis and Emotion Recognition
HeBERT is a Hebrew pretrained language model. It is based on [Google's BERT](https://arxiv.org/abs/1810.04805) architecture and it is BERT-Base config. <br>

HeBert was trained on three dataset: 
1. A Hebrew version of [OSCAR](https://oscar-corpus.com/): ~9.8 GB of data, including 1 billion words and over 20.8 millions sentences. 
2. A Hebrew dump of [Wikipedia](https://dumps.wikimedia.org/): ~650 MB of data, including over 63 millions words and 3.8 millions sentences
3. Emotion User Generated Content (UGC) data that was collected for the purpose of this study (described below).<br>
We evaluated the model on downstream tasks: emotions recognition and sentiment analysis. 

## Emotion UGC Data Description
Our UGC data include comments posted on news articles collected from 3 major Israeli news sites, between January 2020 to August 2020. The total size of the data is ~150 MB, including over 7 millions words and 350K sentences.
~2000 sentences were annotated by crowd members (3-10 annotators per sentence) for overall sentiment (polarity) and [eight emotions](https://en.wikipedia.org/wiki/Robert_Plutchik#Plutchik's_wheel_of_emotions): anger, disgust, expectation , fear, happy, sadness, surprise and trust. 
The percentage of sentences in which each emotion appeared is found in the table below.

|       | anger | disgust | expectation | fear | happy | sadness | surprise | trust | sentiment |
|------:|------:|--------:|------------:|-----:|------:|--------:|---------:|------:|-----------|
| **ratio** |  0.78 |    0.83 |        0.58 | 0.45 |  0.12 |    0.59 |     0.17 |  0.11 | 0.25      |

## Performance
|             | f1   | precision | recall | accuracy |
|-------------|------|-----------|--------|----------|
| **polarity**   | 0.86 | 0.83      | 0.89   | 0.92     |
| **anger**       | 0.97 | 0.97      | 0.97   | 0.95     |
| **disgust**     | 0.96 | 0.97      | 0.95   | 0.93     |
| **expectation** | 0.85 | 0.83      | 0.87   | 0.84     |
| **fear**        | 0.8  | 0.84      | 0.77   | 0.8      |
| **happy**       | 0.88 | 0.89      | 0.87   | 0.97     |
| **sadness**     | 0.84 | 0.83      | 0.84   | 0.79     |
| **surprise**    | 0.41 | 0.47      | 0.37   | 0.78     |
| **trust**       | 0.78 | 0.88      | 0.7    | 0.95     |

*The above metrics for possitive class (==emotion shown in text).*

## How to use
	# For masked-LM model (can be fine-tunned to any down-stream task)
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
	
	# For sentiment classification model (polarity ONLY):
	from transformers import AutoTokenizer, AutoModel
	tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_UGC_sentiment_analysis") #same as 'avichr/heBERT' tokenizer
	model = AutoModel.from_pretrained("avichr/heBERT_UGC_sentiment_analysis")
	
	# how to use?
	sentiment_analysis = pipeline(
	    "sentiment-analysis",
	    model="avichr/heBERT_sentiment_analysis",
	    tokenizer="avichr/heBERT_sentiment_analysis"
	)
	sentiment_analysis('אוהב את העולם')
	# [{'label': 'LABEL_1', 'score': 0.999920666217804}]
	sentiment_analysis('שונא את העולם')
	# [{'label': 'LABEL_0', 'score': 0.9997172951698303}]

	


## Stay tuned!
We are still working on our model and will edit this page as we progress. <br>
Note that we have released only sentiment analysis (polarity) at this point, emotion detection will be released later on.

## Contact us
[Avichay Chriqui](mailto:avichayc@mail.tau.ac.il) <br>
[Inbal yahav](mailto:inbalyahav@tauex.tau.ac.il) <br>
The Coller Semitic Languages AI Lab <br>
Thank you, תודה, شكرا <br>

## If you used this model please cite us as :
TBD in a couple weeks :)

