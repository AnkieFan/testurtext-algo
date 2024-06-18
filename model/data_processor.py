# -*- coding: utf-8 -*-

import pandas as pd
import jieba
import re
import os
from tqdm import tqdm
import json


label2ind = {}

def get_paragraphs(text:str, min_length=200):
    text = re.sub(r'第.+章', '', text)
    text = re.sub(r'(^\[\d+\].*\n?|^[①-⑩].*\n?)', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[\d+\]|[\u2460-\u24FF]', '', text)

    text_no_n = text.rstrip()
    paragraphs = [t.strip() for t in text_no_n.split("\n") if len(t.strip()) > 0]
    filtered_paragraphs = [p for p in paragraphs if not (p.startswith('www') or re.match(r'\[.*?\]', p)) and contains_chinese(p) and len(p)>3]

    final_paragraphs = []
    current_paragraph = ""
    
    for paragraph in filtered_paragraphs:
        if len(current_paragraph) > 0:
            current_paragraph += '\n'
        current_paragraph += paragraph
        
        if len(current_paragraph) >= min_length:
            final_paragraphs.append(current_paragraph.strip())
            current_paragraph = ""
    
    if len(current_paragraph) > 0:
        final_paragraphs.append(current_paragraph.strip())
    
    return final_paragraphs

def contains_chinese(text:str):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def preprocess(folders_path, out_path = 'data/files/train.txt'): # For FastText only
    output = open(out_path, "w", encoding='utf-8') 
    
    for folder_path in folders_path:
        files = os.listdir(folder_path)
        for file_name in tqdm(files):
            file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(file_path):
                try:
                    f = open(file_path,"r",encoding='utf-8')
                    author = file_name.split('_')[0]
                    if author not in label2ind.keys():
                        label2ind[author] = "__label__" + str(len(label2ind))
                    paras = get_paragraphs(f.read())
                    paras = clean_paragraph(paras, label2ind[author])

                    output.writelines(paras)
                except UnicodeDecodeError:
                    print(f"UnicodeDecodeError: Failed to decode file '{file_name}'. Skipping...")
                    continue

def clean_paragraph(paras, label):
    new_paras = []
    for each in paras:
        each = each.rstrip()
        each = each.replace('\n', '')
        each = each.replace('·', ' ')
        each = re.sub(r'\s+', ' ', each)
        each = re.sub(r'\b○\b', '零', each)
        each = each.replace('...', '…')
        each = each.replace('「', '“')
        each = each.replace('」', '”')
        punctuation_map = {',': '，','!': '！','?': '？','(': '（',')': '）', ':':'：',';':'；'}
        for eng_punc, chi_punc in punctuation_map.items():
            each = each.replace(eng_punc, chi_punc)
        each = re.sub(r'[#$%&\'*/<=>@[\\]^_`{|}~]', '', each)
        words = jieba.cut(each)
        stopwords = list(open(r'data/files/stopwords.txt','r', encoding='utf-8'))   
        clean_words = [word for word in words if word not in stopwords and word.strip() != '']
        if(len(clean_words) > 1):
            new_paras.append(label + " "+ " ".join(clean_words) + "\n")
    return new_paras
        

def statistics(folder_path:str):
    sta = {'Author':[], 'Works Contained':[], 'Total Para Count':[]}
    files = os.listdir(folder_path)
    
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        
        if os.path.isfile(file_path):
            try:
                f = open(file_path,"r",encoding='utf-8')
                author = file_name.split('_')[0]
                work = file_name.split('_')[1][:-4] # [:-4]:remove .txt
                paras = get_paragraphs(f.read())
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError: Failed to decode file '{file_name}'. Skipping...")
                continue

            if author in sta['Author']:
                index = sta['Author'].index(author)
                sta['Works Contained'][index]+= ("," + work)
                sta['Total Para Count'][index] += len(paras)
            else:
                sta['Author'].append(author)
                sta['Works Contained'].append(work)
                sta['Total Para Count'].append(len(paras))
        
    return pd.DataFrame(sta)

def text_predict(text):
    text = text.rstrip()
    text = text.replace('\n', '')
    text = text.replace('·', ' ')
    text = re.sub(r'\b○\b', '零', text)
    text = text.replace('...', '…')
    punctuation_map = {',': '，','!': '！','?': '？','(': '（',')': '）', ':':'：',';':'；'}
    for eng_punc, chi_punc in punctuation_map.items():
        text = text.replace(eng_punc, chi_punc)
    text = re.sub(r'[#$%&\'*/<=>@[\\]^_`{|}~]', '', text)
    text = re.sub('[a-zA-Z]',' ', text) # Remove all the English characters
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    words = jieba.lcut(text)
    return " ".join(words)

def get_works_contained(author_names):
    author_names = []
    df = pd.read_excel('data/files/info.xlsx')
    works_contained = {}
    for author in author_names:
        author_row = df.loc[df['Author'] == author]
        if author_row.empty:
            works_contained[author] = None
        else:
            works_contained[author] = author_row['Works Contained'].values[0]
    return works_contained

if __name__ == '__main__':
    # Foreign Authors:
    df = statistics('data/Foreign')
    df.to_excel('data/files/info.xlsx')
    preprocess(['data/Foreign'], out_path='data/files/train_F.txt')
    
    # Chinese Authors:
    df = statistics('data/Chinese')
    df.to_excel('data/files/info_CN.xlsx')
    preprocess(['data/燃冬有好兆头', 'data/Chinese'], out_path='data/files/train_C.txt')
    
    label2ind_json = open('data/files/label2ind.json', "w", encoding='utf-8') 
    label2ind_json.write(json.dumps(label2ind, ensure_ascii = False))
    

