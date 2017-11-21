# encoding=utf-8
import pickle
import pandas as pd
import time
import re

code_set = set()
with open('../DATA/ICD9_descriptions', 'r') as file:
    file.readline()
    for line in file.readlines():
        code_set.add(line.split()[0])
print(len(code_set))


def code_format(code):
    if code.startswith('E'):
        if len(code) >= 5:
            code = code[:4] + '.' + code[4:]
            if code not in code_set:  # 保留小数点一位
                code = code[:6]
    else:
        if len(code) >= 4:
            code = code[:3] + '.' + code[3:]
            if code not in code_set:  # 保留小数点一位
                code = code[:5]
    return code


with open('../PKL/hadm2codes.pkl', 'rb') as file:
    hadm2codes = pickle.load(file)

print('loading reorganize raw_data module')
start_time = time.time()

noteeventFile = '../DATA/NOTEEVENTS.csv'

table = pd.read_csv(noteeventFile, usecols=['SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CATEGORY', 'DESCRIPTION', 'TEXT'],
                    na_filter=False,
                    dtype={'SUBJECT_ID': str, 'HADM_ID': str, 'CHARTDATE': str, 'CATEGORY': str, 'DESCRIPTION': str,
                           'TEXT': str})

print('raw table lens:', len(table))
# drop the note without HADM_ID info
table = table[table['HADM_ID'] != '']
table = table[(table['CATEGORY'] == 'Discharge summary') & (table['DESCRIPTION'] == 'Report')]

print('generated discharge summary:', len(table))

print(time.time() - start_time)

with open('../DATA/MIMIC3_RAW_DSUMS', 'w') as file:
    file.write('"subject_id"|"hadm_id"|"charttime"|"category"|"title"|"icd9_codes"|"text"\n')
    for line in table.values:
        # line is a ndarray
        subject_id = line[0]
        hadm_id = line[1]
        chartdate = line[2]
        categoty = line[3]
        note_text = str.lower(line[5]).strip()
        if hadm_id not in hadm2codes:
            print(hadm_id)
            continue
        flist = []
        break_loop = False
        for code in hadm2codes[hadm_id]:
            fcode = code_format(code)
            if fcode not in code_set:
                break_loop = True
                break
            flist.append(fcode)
        if break_loop:
            continue

        file.write(subject_id + '|')
        file.write(hadm_id + '|"')
        file.write(chartdate + '"|"')
        file.write(categoty + '"|"')
        file.write('"|"')
        # write codes
        file.write(flist[0])
        for fcode in flist[1:]:
            file.write(',' + fcode)
        file.write('"|"')
        note_text = re.sub(r'\n', '[newline]', note_text)
        note_text = re.sub(r'\s+', ' ', note_text)
        file.write(note_text + '\n')

print(time.time() - start_time)
