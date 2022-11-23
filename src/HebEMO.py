class HebEMO:
    def __init__(self, device=-1, emotions = ['anticipation', 'joy', 'trust', 'fear', 'surprise', 'anger',
      'sadness', 'disgust']):
        from transformers import pipeline
        from tqdm import tqdm
        self.device = device
        if type(emotions) == str:
            self.emotions = [emotions]
        elif type(emotions) != list:
          raise ValueError('emotions should be emotion as a text or list of emotions.')
        else:
            self.emotions = emotions
        self.hebemo_models = {}
        for emo in tqdm(self.emotions): 
            self.hebemo_models[emo] = pipeline(
                "sentiment-analysis",
                model="avichr/hebEMO_"+emo,
                tokenizer="avichr/heBERT",
                device = self.device #-1 run on CPU, else - device ID
            )
    
    def hebemo(self, text = None, input_path=False, save_results=False, read_lines=False, plot=False, batch_size=32, max_length = 512, truncation=True):
        '''
        text (str): a text or list of text to analyze
        input_path(str): the path to the text file (txt file, each row for different instance)
        '''
        from pyplutchik import plutchik
        import matplotlib.pyplot as plt
        import pandas as pd
        import time
        import torch
        from tqdm import tqdm
        if text is None and type(input_path) is str:
            # read the file
            with open(input_path, encoding='utf8') as p:
                txt = p.readlines()
        elif text is not None and (input_path is None or input_path is False):
            if type(text) is str:
                if read_lines:
                    txt = text.split('\n')
                else:
                    txt = [text]
            elif type(text) is list:
                txt = text
            else: 
                raise ValueError('text should be text or list of text.')
        else:
            raise ValueError('you should provide a text string, list of strings or text path.')
        # run hebEMO
        hebEMO_df = pd.DataFrame(txt) 
        for emo in tqdm(self.emotions): 
            x = self.hebemo_models[emo](txt, truncation=truncation, max_length=max_length, batch_size=batch_size)
            hebEMO_df = hebEMO_df.join(pd.DataFrame(x).rename(columns = {'label': emo, 'score':'confidence_'+emo}))
            del x
            torch.cuda.empty_cache()
        hebEMO_df = hebEMO_df.applymap(lambda x: 0 if x=='LABEL_0' else 1 if x=='LABEL_1' else x)
        if save_results is not False:
            gen_name = str(int(time.time()*1e7))
            if type(save_results) is str:      
                hebEMO_df.to_csv(save_results+'/'+gen_name+'_heEMOed.csv', encoding='utf8')
            else: 
                hebEMO_df.to_csv(gen_name+'_heEMOed.csv', encoding='utf8')
        if plot:
            hebEMO = pd.DataFrame()
            for emo in hebEMO_df.columns[1::2]:
                hebEMO[emo] = abs(hebEMO_df[emo]-(1-hebEMO_df['confidence_'+emo]))
            for i in range(0,1):    
                ax = plutchik(hebEMO.to_dict(orient='records')[i])
                print(hebEMO_df[0][i])
                plt.show()
            return (hebEMO_df[0][i], ax)
        else:
            return (hebEMO_df)

        
def run_me_for_interactive_usage():
  # for using streamlit in colab
  # check https://colab.research.google.com/drive/1Jw3gOWjwVMcZslu-ttXoNeD17lms1-ff?usp=sharing#scrollTo=EeEPkdDo3AQf for demo
  !pip install -r HeBERT/requirements.txt &> /dev/nullc
  from HeBERT.src.HebEMO import HebEMO
  HebEMO_model = HebEMO()
  !streamlit run HebEMO_demo/colab_app.py &>/content/logs.txt &
  !npx localtunnel --port 8501
