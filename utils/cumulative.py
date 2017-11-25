# encoding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import collections

# ====================
top_num = 6984

diagnosisFile = '../data/raw_data/DIAGNOSES_ICD.csv'

icd9list = []
admset = set()
adm2icd = {}
infd = open(diagnosisFile, 'r')
infd.readline()
for line in infd:
    tokens = line.strip().split(',')
    pid = int(tokens[1])
    admId = int(tokens[2])
    icd9code = tokens[4][1:-1]
    if icd9code == '':
        continue
    if admId not in adm2icd:
        adm2icd[admId] = [icd9code]
    else:
        adm2icd[admId].append(icd9code)

    icd9list.append(icd9code)
    admset.add(admId)
infd.close()

assert len(set(icd9list)) == top_num
total_freq = len(icd9list)
print(total_freq)
c = collections.Counter(icd9list)
print('total code number:', len(c))
top_list = c.most_common(top_num)
# psum = len(pset)
admnum = len(admset)
print('admnum', admnum)
top_list = [(code, num / total_freq) for code, num in top_list]
# top_list = [(code, num / admnum) for code, num in top_list]
print('ICD Distribution:')
print(top_list)
print('number of medical record:', admnum)

# print('Avg. # of codes per patient', total_freq / admnum)
# print('Max # of codes per patient', max([len(codes) for adm, codes in adm2icd.items()]))
# print('Min # of codes per patient', min([len(codes) for adm, codes in adm2icd.items()]))
print("============================================")
data = list(range(top_num + 1))
values = np.cumsum(list(map(lambda x: x[1], top_list)))
values = np.concatenate([[0], values])
print(data)
print(values)

plt.plot(data, values, c='green', linewidth=4)  # , c='blue'
plt.axis([-100, 7000, 0, 1.05])
plt.title('ICD-9 Code Distribution')  # give plot a title
plt.xlabel('Number of Codes')  # make axis labels
plt.ylabel('Cumulative Frequency of Codes Covered')
plt.savefig("examples.tif")
plt.show()
