from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import pandas as pd
import pickle
from statsmodels.stats import weightstats as stests
from collections import Counter, defaultdict
import networkx as nx 
from sklearn import metrics
import statsmodels.api as sm


def main():
    for FIELD in ['abstract_Marketing', 'abstract', 'Title']:
        main_worker(FIELD)
    return


def main_worker(FIELD):
    import spacy
    spacy_nlp = spacy.load("en_core_web_sm")
    pos_tags_vocab = [
        'ADJ', 'ADV', 'VERB', 'NOUN', 'PROPN', 'PUNCT', 
    ]
    from main_analyze import get_main_data
    df_all = get_main_data(assign_net_income_rank_flag=True)
    df_all = df_all.dropna(subset=[FIELD]) 

    punc_counter = Counter()
    adj_counter = Counter()
    pos_percentage_list = defaultdict(list)
    for idx, row in tqdm(df_all.iterrows(), total=df_all.shape[0]):
        fileNumber = row['DocketNumber'] 
        Title = row[FIELD] 
        assert isinstance(Title, str) and len(Title) > 0
        text = Title
        text = text.replace('-', ' ') 
        text = text.replace('_x000d_', ' ') 
        doc = spacy_nlp(text,) 
        pos_counter = Counter()

        for token in doc:
            pos_counter[token.pos_] += 1
            if token.pos_ == 'PUNCT':
                punc_counter[token.text] += 1
            if token.pos_ == 'ADJ':
                adj_counter[token.text.lower() ] += 1
        for pos_ in pos_tags_vocab:
            pos_percentage_list[pos_ + '_percentage'].append( pos_counter[pos_] / len(doc) ) # a percentage 
            pos_percentage_list[pos_ + '_count'].append( pos_counter[pos_]  ) # occurence 
        pos_percentage_list['Title_len'].append(  len(doc) )
    for dict_key in pos_percentage_list.keys():
        df_all[dict_key] = pos_percentage_list[dict_key]
    cache_df_path = "notebook_plots/linguistic_spacy_{}.pkl".format(FIELD)
    df_all.to_pickle(cache_df_path)
    print('File saved to ' + cache_df_path)
    print(FIELD, 'adj_counter', adj_counter)
    return 

if __name__ == '__main__':
    main()
