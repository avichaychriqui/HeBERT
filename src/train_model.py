import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer, BertModel, BertConfig
import random


class BertTrainer:
    def __init__(self, num_classes, max_length=15, model_name = 'bert-base-uncased', device="cpu", seed_num=42):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.model.to(device)
        self.max_length = max_length
        self.device = device

        random.seed(seed_num)
        torch.manual_seed(seed_num)
        np.random.seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def tokenize(self, texts):
        encoded_dict = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded_dict['input_ids'].to(self.device)
        attention_masks = encoded_dict['attention_mask'].to(self.device)
        return input_ids, attention_masks

    def train(self, texts, labels, batch_size=32, epochs=5, learning_rate=2e-5, return_loss = False):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        Epochs_Losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_steps = 0

            for i in range(0, len(texts), batch_size):
                optimizer.zero_grad()

                input_ids, attention_masks = self.tokenize(texts[i:i+batch_size])
                batch_labels = torch.tensor(labels[i:i+batch_size]).to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_masks, labels=batch_labels)
                loss = outputs[0]


                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_steps += 1

            print(f"Epoch {epoch+1} Loss: {epoch_loss / epoch_steps:.4f}")
            Epochs_Losses.append(epoch_loss / epoch_steps)

        if return_loss:
            return(Epochs_Losses)

    def predict(self, texts):
        self.model.eval()
        input_ids, attention_masks = self.tokenize(texts)
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_masks)
            logits = outputs[0]
            predictions = torch.argmax(logits, dim=1)

        return_dic = {}
        return_dic['predictions'] = predictions.cpu().numpy()
        return_dic['logits'] = logits.cpu().numpy()

        return return_dic

    def eval (self, texts, y_true):
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

        predictions_outputs = self.predict(texts)
        st.write(confusion_matrix(y_true, predictions_outputs['predictions']),
        classification_report(y_true, predictions_outputs['predictions'], output_dict=True),
        'auc : ', roc_auc_score(y_true, predictions_outputs['logits'][:, 1]), sep = '\n')



from transformers import pipeline
import streamlit as st
from io import StringIO
import pandas as pd
from sklearn.model_selection import train_test_split


st.title("Train your own model for Hebrew texts classification")

model_name = st.selectbox('What is the base language model to train a model?', ('avichr/heBERT', 'avichr/Legal-heBERT_ft', 'avichr/Legal-heBERT',
                                                                                  'bert-base-uncased'))

max_length = st.slider('What is the maximum length of the text to analyze? \n (The longer the text, the longer the calculation time will be)',
  5, 512, 20)

num_classes = st.number_input('How many classes you have in the data?', 2,10,2)
learning_rate = st.number_input('What is your learning rate?', min_value = 1e-10, max_value= 1e-1, value = 5e-5, step = 1e-10)
batch_size = st.slider('What is your batch size?', 2,512, 32)
epochs = st.slider('How many epochs to learn?', 1, 50, 5)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
trainer = BertTrainer(num_classes=num_classes, max_length=max_length, device=device, model_name= model_name)

uploaded_file = st.file_uploader("Choose a file to train the model")
to_predict_file = st.file_uploader("Choose a file to predict", key = 'to_predict_file')

if uploaded_file is not None and to_predict_file is not None:
    df = pd.read_csv(uploaded_file)
    to_predict = pd.read_csv(to_predict_file)

    text_column = st.selectbox('What is the text column in your data?', (df.columns))
    label_column = st.selectbox('What is the label column in your data? \n labels should be integers only', (df.columns), key = '_')
    text_column_to_predict = st.selectbox('What is the text column in your data to predict?', (to_predict.columns))

    df = df[[text_column, label_column]].dropna()


    test_size = st.slider('% to split test from data? \n The selected file will be automatically divided into training and validation splits (recommended).', 0.0, 1.0, .33)

    if 'text_column' in locals() and 'label_column' in locals():
        X_train, X_test, y_train, y_test = train_test_split(df[text_column].to_list(), df[label_column].to_list(), test_size=test_size, stratify= df[label_column], random_state=42)

    if 'X_train' in locals() and 'trainer' in locals() and st.button('Train', key='train'):
        trainer.train(X_train, y_train, batch_size=batch_size, epochs=epochs, learning_rate=5e-5)
        trained = True

        st.write('model perfomances:')
        st.write(trainer.eval(X_test, y_test))


        st.write('Predictions:')
        preds = pd.DataFrame({'text': to_predict[text_column_to_predict],
                                'predictions': trainer.predict(to_predict[text_column_to_predict])['predictions']
                                })

        @st.cache_data
        def convert_df(df):
          return df.to_csv(index=False).encode('utf-8')


        preds_df = convert_df(preds)

        st.download_button(
          "Press to Download",
          preds_df,
          "preds.csv",
          "text/csv",
          key='download-csv'
        )

        st.write (preds)
