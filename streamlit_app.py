from src.HebEMO import HebEMO
from transformers import pipeline
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from spider_plot import spider_plot


# @st.cache
HebEMO_model = HebEMO()


st.title("Emotion Recognition in Hebrew Texts")
st.write("HebEMO is a tool to detect polarity and extract emotions from Hebrew user-generated content (UGC), which was trained on a unique Covid-19 related dataset that we collected and annotated. HebEMO yielded a high performance of weighted average F1-score = 0.96 for polarity classification. Emotion detection reached an F1-score of 0.78-0.97, with the exception of *surprise*, which the model failed to capture (F1 = 0.41). More information can be found in our git: https://github.com/avichaychriqui/HeBERT")
st.write("Write Hebrew sentences in the text box below to analyze (each sentence in a different rew). It takes a while, be patient :). An additional demo can be found in the Colab notebook: https://colab.research.google.com/drive/1Jw3gOWjwVMcZslu-ttXoNeD17lms1-ff ")

sent = st.text_area("Text", "החיים יפים ומאושרים", height = 20)
# interact(HebEMO_model.hebemo, text='החיים יפים ומאושרי', plot=fixed(True), input_path=fixed(False), save_results=fixed(False),)

hebEMO_df = HebEMO_model.hebemo(sent, read_lines=True, plot=False)
hebEMO = pd.DataFrame()
for emo in hebEMO_df.columns[1::2]:
    hebEMO[emo] = abs(hebEMO_df[emo]-(1-hebEMO_df['confidence_'+emo]))

st.write (hebEMO)

