# encoding=utf-8
# !/usr/bin/env python

import pickle
import re

term_pattern = re.compile('[A-Za-z]+')
vocab_size = 40000
##############################################

stopwords = '../DATA/stopwords.txt'
# Define stopwords
with open(stopwords) as f:
    stopwords = []
    for line in f:
        stopwords.append(line.strip())
stopwords = set(stopwords)

# Write vocab to file
vocab = set()
with open('../DATA/MIMIC_vocabulary2', 'r') as f:
    for key in f.readlines():
        vocab.add(key.strip())
assert len(vocab) == vocab_size
# Tokenize and write document counts to file
all_text = []
all_code = []
with open('../DATA/MIMIC_FILTERED_DSUMS') as f:
    for i, line in enumerate(f):
        rows = line.split('|')
        raw_dsum = rows[6]
        codes = line.strip().split('|')[5].strip('"').split(',')
        raw_dsum = re.sub(r'\[[^\]]+\]', ' ', raw_dsum)
        raw_dsum = re.sub(r'admission date:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'discharge date:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'date of birth:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'sex:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'service:', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'dictated by:.*$', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'completed by:.*$', ' ', raw_dsum, flags=re.I)
        raw_dsum = re.sub(r'signed electronically by:.*$', ' ', raw_dsum, flags=re.I)

        tokens = [token.lower() for token in re.findall(term_pattern, raw_dsum)]
        tokens = [token for token in tokens if token in vocab]
        all_text.append(tokens)
        all_code.append(codes)

with open('../PKL/all_text.pkl', 'wb') as file:
    pickle.dump(all_text, file)

with open('../PKL/all_code.pkl', 'wb') as file:
    pickle.dump(all_code, file)

print(max([len(text) for text in all_text]))
print(min([len(text) for text in all_text]))
print(sum([len(text) for text in all_text])/len(all_text))