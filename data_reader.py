from collections import Counter
import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from table_bert import VerticalAttentionTableBert,Table, Column
from tqdm import tqdm
from random import randint
import json
import pandas as pd
from pytorch_pretrained_bert.tokenization import BertTokenizer
# from transformers import BertTokenizer

from preprocess import data_utils

import argparse
import random
import re
from collections import OrderedDict
import spacy
import pickle

from utils import process_table

# class DataAndQueryReader(Dataset):
#     def __init__(self,data,data_folder):

#         model=BertTokenizer.from_pretrained('bert-base-uncased')
#         nlp_model=spacy.load('en_core_web_sm')

#         max_tokens=50
#         labels = []

#         tables1 = []
#         tables2 = []

#         metas1 = []
#         metas2 = []


#         with tqdm(total=len(data)) as pbar:
#             for index,row in enumerate(data):

#                 tab1=row[0]
#                 tab2=row[1]
#                 label=int(row[2])
#                 table1,meta1=process_table(tab1, data_folder, model, max_tokens, nlp_model)
#                 table2, meta2 = process_table(tab2, data_folder, model, max_tokens, nlp_model)

#                 tables1.append(table1)
#                 tables2.append(table2)
#                 metas1.append(meta1)
#                 metas2.append(meta2)
#                 labels.append(label)
                
#                 pbar.update(1)
                    
#         self.tables1=tables1
#         self.tables2=tables2
#         self.metas1 = metas1
#         self.metas2 = metas2
#         self.labels = labels


#     def __getitem__(self, t):

#         return self.tables1[t], self.tables2[t], self.metas1[t], self.metas2[t],  self.labels[t]

#     def __len__(self):

#         return len(self.tables1)

# ===================== GLOBAL CACHE (per worker) =====================
# Each DataLoader worker will have its own copy
TABLE_CACHE = {}

def load_spacy():
    # Load once per worker
    if not hasattr(load_spacy, "nlp"):
        load_spacy.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return load_spacy.nlp


class DataAndQueryReader(Dataset):
    def __init__(self, data, data_folder, cache_file=None):
        """
        data: list of (tab1, tab2, label)
        data_folder: path to tables
        cache_file: optional path to save/load preprocessed dataset
        """

        self.data = data
        self.data_folder = data_folder
        self.max_tokens = 50

        # tokenizer is lightweight enough to keep here
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Optional full dataset cache
        self.cache_file = cache_file
        self.full_cache = None

        if cache_file and os.path.exists(cache_file):
            print(f"[INFO] Loading cached dataset from {cache_file}")
            with open(cache_file, "rb") as f:
                self.full_cache = pickle.load(f)

    def process_with_cache(self, tab):
        """
        Cache per-table processing (BIG speedup)
        """
        global TABLE_CACHE

        if tab in TABLE_CACHE:
            return TABLE_CACHE[tab]

        nlp_model = load_spacy()

        table, meta = process_table(
            tab,
            self.data_folder,
            self.tokenizer,
            self.max_tokens,
            nlp_model
        )

        TABLE_CACHE[tab] = (table, meta)
        return table, meta

    def __getitem__(self, idx):
        # If full dataset cached → fastest path
        if self.full_cache is not None:
            return self.full_cache[idx]

        tab1, tab2, label = self.data[idx]

        table1, meta1 = self.process_with_cache(tab1)
        table2, meta2 = self.process_with_cache(tab2)

        return table1, table2, meta1, meta2, int(label)

    def __len__(self):
        return len(self.data)

    def build_full_cache(self, save_path):
        """
        Run once → saves fully processed dataset
        """
        print("[INFO] Building full dataset cache...")

        all_data = []
        for i in range(len(self.data)):
            all_data.append(self[i])

        with open(save_path, "wb") as f:
            pickle.dump(all_data, f)

        print(f"[INFO] Saved cache to {save_path}")