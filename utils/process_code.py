# encoding=utf-8
import pickle

with open('../PKL/all_code.pkl', 'rb') as file:
    all_code = pickle.load(file)

code_dict = {}
for line in all_code:
    for code in line:
        code_dict[code] = code_dict.get(code, 0) + 1

code_set = sorted(code_dict.items(), key=lambda code: (-code[1], code[0]))

print(len(code_set))

code_dict = {e[0]: i for i, e in enumerate(code_set)}
with open('../PKL/code_dict.pkl', 'wb') as file:
    pickle.dump(code_dict, file)
