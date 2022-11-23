from HebEMO import HebEMO
from transformers import pipeline
import streamlit as st
from io import StringIO 
import pandas as pd

st.title("Emotion Recognition in Hebrew Texts")
st.write("HebEMO is a tool to detect polarity and extract emotions from Hebrew user-generated content (UGC), which was trained on a unique Covid-19 related dataset that we collected and annotated. HebEMO yielded a high performance of weighted average F1-score = 0.96 for polarity classification. Emotion detection reached an F1-score of 0.78-0.97, with the exception of *surprise*, which the model failed to capture (F1 = 0.41). More information can be found in our git: https://github.com/avichaychriqui/HeBERT")
st.write("Write Hebrew sentences in the text box below to analyze (each sentence in a different rew). An additional demo can be found in the Colab notebook: https://colab.research.google.com/drive/1Jw3gOWjwVMcZslu-ttXoNeD17lms1-ff ")
max_length = st.slider('What is the maximum length of the text to analyze? \n (The longer the text, the longer the calculation time will be)',
  5, 512, 20)


def st_function_HebEMO_model():  
  return HebEMO()


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    string_data = stringio.read()
    st.write('It takes a while, be patient :)')
    HebEMO_model = HebEMO()

    hebEMO_df = HebEMO_model.hebemo(string_data, read_lines=True, plot=False, batch_size=32, max_length = max_length)
    hebEMO = pd.DataFrame(hebEMO_df[0])
    for emo in hebEMO_df.columns[1::2]:
        hebEMO[emo] = abs(hebEMO_df[emo]-(1-hebEMO_df['confidence_'+emo]))

    @st.experimental_memo
    def convert_df(df):
      return df.to_csv(index=False).encode('utf-8')


    csv = convert_df(hebEMO)

    st.download_button(
      "Press to Download",
      csv,
      "hebEMO-ed.csv",
      "text/csv",
      key='download-csv'
    )

    st.write (hebEMO)

