class HebEMO:
    def __init__(self, device=-1, emotions = ['anticipation', 'joy', 'trust', 'fear', 'surprise', 'anger',
      'sadness', 'disgust']):
        from transformers import pipeline
        from tqdm import tqdm
        
        self.device = device
        self.emotions = emotions
        self.hebemo_models = {}
        for emo in tqdm(emotions): 
            self.hebemo_models[emo] = pipeline(
                "sentiment-analysis",
                model="avichr/hebEMO_"+emo,
                tokenizer="avichr/heBERT",
                device = self.device #-1 run on CPU, else - device ID
            )
    
    def hebemo(self, text = None, input_path=False, save_results=False, read_lines=False, plot=False):
        '''
        text (str): a text or list of text to analyze
        input_path(str): the path to the text file (txt file, each row for different instance)
        '''
        try: from pyplutchik import plutchik
        except: from spider_plot import spider_plot
        
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
            x = self.hebemo_models[emo](txt)
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
                try: ax = plutchik(hebEMO.to_dict(orient='records')[i])
                except: ax = spider_plot(hebEMO); print('we recommend installing pyplutchik library for better visualization')
                print(hebEMO_df[0][i])
                plt.show()
            return (hebEMO_df[0][i], ax)
        else:
            return (hebEMO_df)
