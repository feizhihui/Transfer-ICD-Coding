# encoding=utf-8
import pandas as pd
import pickle
import time

print('loading process_note module')
start_time = time.time()

noteeventFile = '../DATA/NOTEEVENTS.csv'

table = pd.read_csv(noteeventFile, usecols=['SUBJECT_ID', 'HADM_ID', 'CATEGORY', 'DESCRIPTION', 'TEXT'],
                    na_filter=False,
                    dtype={'SUBJECT_ID': str, 'HADM_ID': str, 'CATEGORY': str, 'DESCRIPTION': str, 'TEXT': str})

print('raw table lens:', len(table))
# drop the note without HADM_ID info
table = table[table['HADM_ID'] != '']
table = table[(table['CATEGORY'] == 'Discharge summary') & (table['DESCRIPTION'] == 'Report')]

print('generated discharge summary:', len(table))

print(time.time() - start_time)

count = 0
hadm2note = {}
for line in table.values:
    # line is a ndarray
    subject_id = int(line[0])
    hadm_id = int(line[1])
    categoty = line[2]
    assert categoty == 'Discharge summary'
    note_text = str.lower(line[3])
    if hadm_id in hadm2note:
        print(hadm_id)
        count += 1
        hadm2note[hadm_id] = note_text
    else:
        hadm2note[hadm_id] = note_text

print(count)
print(time.time() - start_time)
print('begin to dump pickle!')

with open('../PKL/hadm2note.pkl', 'wb') as file:
    pickle.dump(hadm2note, file)
print(time.time() - start_time)
