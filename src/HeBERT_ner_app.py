from transformers import pipeline
import streamlit as st
from io import StringIO 
import pandas as pd

st.title("NER in Hebrew Texts")
st.write("Named-entity recognition (NER) (also known as (named) entity identification, entity chunking, and entity extraction) is a subtask of information extraction that seeks to locate and classify named entities mentioned in unstructured text into pre-defined categories such as person names, organizations, locations, medical codes, time expressions, quantities, monetary values, percentages, etc. [wikipedia]")
st.write("Write Hebrew sentences in the text box below to analyze (each sentence in a different rew). ")
max_length = st.slider('What is the maximum length of the text to analyze? \n (The longer the text, the longer the calculation time will be)',
  5, 512, 20)

NER = pipeline(
    "token-classification",
    model="avichr/heBERT_NER",
    tokenizer="avichr/heBERT_NER",
    ignore_labels = [],
    aggregation_strategy = 'simple'
)

method = st.selectbox(
    'What is your input?',
    ('File', 'Free Text'))

if method == 'File':
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        # To read file as bytes:
        bytes_data = uploaded_file.getvalue()

        # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

        # To read file as string:
        string_data = stringio.read()

else: 
    string_data = st.text_area(label = 'Text to analyze', value="",
    placeholder = '''
        It was the best of times, it was the worst of times, it was
        the age of wisdom, it was the age of foolishness, it was
        the epoch of belief, it was the epoch of incredulity, it
        was the season of Light, it was the season of Darkness, it
        was the spring of hope, it was the winter of despair, (...)
        ''')
if 'string_data' in locals():
    if (string_data != '' and string_data is not None):
        st.write('It takes a while, be patient :)')
        ner_df = pd.DataFrame(NER(string_data))
        @st.cache_data
        def convert_df(df):
          return df.to_csv(index=False).encode('utf-8')


        csv = convert_df(ner_df)

        st.download_button(
          "Press to Download",
          csv,
          "ner_df.csv",
          "text/csv",
          key='download-csv'
        )

        st.write (ner_df)
