# HeBERT: Pre-trained BERT for Polarity Analysis and Emotion Recognition
HeBERT is a Hebrew pretrained language model. It is based on Google's BERT architecture and it is BERT-Base config. <br>

HeBert was trained on three dataset: 
1. A Hebrew version of OSCAR: ~9.8 GB of data, including 1 billion words and over 20.8 millions sentences. 
2. A Hebrew dump of Wikipedia: ~650 MB of data, including over 63 millions words and 3.8 millions sentences
3. Emotion UGC data that was collected for the purpose of this study. (described below)
We evaluated the model on emotion recognition and sentiment analysis, for a downstream tasks and masked fill-in-the-blank task (the main task of BERT model). 

## Emotion UGC Data Description
Our User Genrated Content (UGC) is comments written on articles collected from 3 major news sites, between January 2020 to August 2020,. Total data size ~150 MB of data, including over 7 millions words and 350K sentences.
XX sentences annotation by crowd members (3-10 annotators per sentence) for 8 emotions (list here) and overall sentiment/ polarity
Percent of sentiment + each emotion in the labelled data


## Performance
### Emotion recognition
Emotion	Precision	Recall 	F1 Score	Overall Accuracy
				
				
				

### sentiment analysis  
HeBERT's				
	precision	recall	f1-score	
0	0.94	0.95	0.95	
1	0.89	0.88	0.89	
2	0.73	0.56	0.63	
accuracy			0.92	
macro avg	0.85	0.80	0.82	
weighted avg	0.92	0.92	0.92	
based on Amram, A., Ben-David, A., and Tsarfaty, R. (2018) dataset

## How to use

## Stay tuned!
1.	Our model is still building. We will edit this page as we progress. 

## Contact us
Avichay Chriqui <br>
Inbal yahav <br>
The Coller AI Lab <br>
Thank you, תודה, شكرا <br>

## If you used this model please cite us as :
TBD in a couple week :)

