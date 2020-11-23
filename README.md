# HeBERT: Pre-trained BERT for Polarity Analysis and Emotion Recognition
HeBERT is a Hebrew pretrained language model. It is based on Google's BERT architecture and it is BERT-Base config [(Devlin et al. 2018)](https://arxiv.org/abs/1810.04805). <br>

HeBert was trained on three dataset: 
1. A Hebrew version of OSCAR [(Ortiz, 2019)](https://oscar-corpus.com/): ~9.8 GB of data, including 1 billion words and over 20.8 millions sentences. 
2. A Hebrew dump of Wikipedia: ~650 MB of data, including over 63 millions words and 3.8 millions sentences
3. Emotion UGC data that was collected for the purpose of this study. (described below)
We evaluated the model on emotion recognition and sentiment analysis, for a downstream tasks. 

## Emotion UGC Data Description
Our User Genrated Content (UGC) is comments written on articles collected from 3 major news sites, between January 2020 to August 2020,. Total data size ~150 MB of data, including over 7 millions words and 350K sentences.
4000 sentences annotated by crowd members (3-10 annotators per sentence) for 8 emotions (anger, disgust, expectation , fear, happy, sadness, surprise and trust) and overall sentiment / polarity<br>
In order to valid the annotation, we search an agreement between raters to emotion in each sentence using krippendorff's alpha [(krippendorff, 1970)](https://journals.sagepub.com/doi/pdf/10.1177/001316447003000105). We left sentences that got alpha > 0.7. Note that while we found a general agreement between raters about emotion like happy, trust and disgust, there are few emotion with general disagreement about them, apparently given the complexity of finding them in the text (e.g. expectation and surprise).


|       | anger | disgust | expectation | fear | happy | sadness | surprise | trust | sentiment |
|------:|------:|--------:|------------:|-----:|------:|--------:|---------:|------:|-----------|
| **count** |  1979 |    2115 |         681 | 1041 |  2342 |     998 |      698 |  1956 | 2049      |
| **ratio** |  0.78 |    0.83 |        0.58 | 0.45 |  0.12 |    0.59 |     0.17 |  0.11 | 0.25      |

## Performance
### Emotion recognition
|             | f1   | precision | recall | accuracy |
|-------------|------|-----------|--------|----------|
| **anger**       | 0.97 | 0.97      | 0.97   | 0.95     |
| **disgust**     | 0.96 | 0.97      | 0.95   | 0.93     |
| **expectation** | 0.85 | 0.83      | 0.87   | 0.84     |
| **fear**        | 0.8  | 0.84      | 0.77   | 0.8      |
| **happy**       | 0.88 | 0.89      | 0.87   | 0.97     |
| **sadness**     | 0.84 | 0.83      | 0.84   | 0.79     |
| **sentiment**   | 0.86 | 0.83      | 0.89   | 0.92     |
| **surprise**    | 0.41 | 0.47      | 0.37   | 0.78     |
| **trust**       | 0.78 | 0.88      | 0.7    | 0.95     |

Metrics for possitive class (==emotion shown in text)
				
				

### sentiment analysis  				
|              | precision | recall | f1-score |
|--------------|-----------|--------|----------|
| 0            | 0.94      | 0.95   | 0.95     |
| 1            | 0.89      | 0.88   | 0.89     |
| 2            | 0.73      | 0.56   | 0.63     |
| accuracy     |           |        | 0.92     |
| macro avg    | 0.85      | 0.8    | 0.82     |
| weighted avg | 0.92      | 0.92   | 0.92     |

based on [Amram, A., Ben-David, A., and Tsarfaty, R. (2018) dataset](https://github.com/omilab/Neural-Sentiment-Analyzer-for-Modern-Hebrew)

## How to use
	# For masked-LM model (can be fine-tunned to any down-stream task)
	from transformers import AutoTokenizer, AutoModel
	tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
	model = AutoModel.from_pretrained("avichr/heBERT")
	
	# For sentiment classification model:
	from transformers import AutoTokenizer, AutoModel
	tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
	model = AutoModel.from_pretrained("avichr/heBERT_UGC_sentiment_analysis")


## Stay tuned!
Our model is still building. We will edit this page as we progress. 

## Contact us
[Avichay Chriqui](mailto:avichayc@mail.tau.ac.il) <br>
[Inbal yahav](mailto:inbalyahav@tauex.tau.ac.il) <br>
The Coller AI Lab <br>
Thank you, תודה, شكرا <br>

## If you used this model please cite us as :
TBD in a couple week :)

