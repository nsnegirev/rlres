import torch
import pandas as pd
import numpy as np

from fastapi import FastAPI

from ml.model import load_model
from data.data_preprocessing import preprocessing

model = None

def eval(text_dataloader):
    #model, tokenizer = load_model()
    forecast_df = pd.DataFrame()
    with torch.no_grad():   
        for i, batch in enumerate(text_dataloader):
            X_temp_batch = []
            batch = {key: value.to('cpu') for key, value in batch.items()}
            segment_info = torch.LongTensor([[1]*len(batch['attention_mask'][0]) for _ in range(len(batch['attention_mask']))]).to('cpu')
            output = model(batch['input_ids'], segment_info, batch['attention_mask'])
            forecast_df = pd.concat([forecast_df, pd.DataFrame(data=output.detach().cpu().numpy(), columns=range(2893))], ignore_index=True)
        return forecast_df
    
def result_conc(forecast, i, KSR, n):
    forecast_sorted = pd.DataFrame(forecast.iloc[i,:]).sort_values(by=[i], key=lambda item: -item)[:n]
    forecast_sorted = forecast_sorted.rename(columns={i: "probs"})
    forecast_scaled = forecast_sorted.map(lambda x: 1.1**x)#math.exp(x/8))
    forecast_sorted = forecast_scaled.apply(lambda x: x / x.sum(), axis=0)
    forecast_sorted = forecast_sorted.reset_index(names='code1')
    result = pd.merge(forecast_sorted, KSR, how = 'left', left_on='code1', right_on='code')
    result = result.reindex(columns=['code_KSR', 'name_KSR', 'probs'])
    return result

def type_cor(text):
    text_new = dict()
    if isinstance(text, dict) == False:
        text_new['ID'] = '0'
        text_new['ResourceName'] = text
        return text_new
    return text

def startup_event():
    global model, tokenizer, KSR
    model, tokenizer = load_model()
    model.to('cpu')
    KSR = pd.read_excel('../data/KSR.xlsx', sheet_name = 'KSR', usecols=['code', 'code_KSR', 'name_KSR'])
    
#{'new':[{'ID':'ddd', 'ResourceName': 'лист хризотилцементный'}, 'ID':'aaa', 'ResourceName': 'бетон'}]})	
def predictdictn(text: dict):
    text_corr = text['new']
    text_pd = pd.DataFrame(text_corr)
    forecast = eval(preprocessing(data=list(text_pd['ResourceName']), tokenizer=tokenizer))
    for i in range(len(text_corr)):
        text_corr[i]['eval'] = result_conc(forecast, i, KSR=KSR, n=10)
    return text_corr

#text
def predicttext(text):
    text_corr = type_cor(text)
    forecast = eval(preprocessing(data=[text_corr['ResourceName']], tokenizer=tokenizer))
    text_corr['eval'] = result_conc(forecast=forecast, i=0, KSR=KSR, n=10)
    return text_corr

#{'ID':'aaa', 'ResourceName': 'бетон'}]})
def predictdict(text: dict):
    text_corr = type_cor(text)
    forecast = eval(preprocessing(data=[text_corr['ResourceName']], tokenizer=tokenizer))
    text_corr['eval'] = result_conc(forecast=forecast, i=0, KSR=KSR, n=10)
    return text_corr
