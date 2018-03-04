# encoding=utf-8
import json
import pickle

# total 12834584
# front_select = 6834585
front_select = 128345

big_path = '/media/cb201/8A5EE1D45EE1B959/fzh/allMeSH_2017.json'

with open(big_path, encoding='utf-8', errors='ignore') as file:
    doc = json.load(file)

articles = doc['articles'][:front_select]
print('total articles:', len(doc['articles']), 'and selection:', len(articles))

meshDict = dict()
all_mesh = []
for cita in articles:
    mesh_list = []
    for mesh in cita['meshMajor']:
        mesh_list.append(mesh)
        if mesh not in meshDict:
            meshDict[mesh] = 1
        else:
            meshDict[mesh] += 1
    all_mesh.append(mesh_list)

meshItems = sorted(meshDict.items(), key=lambda item: (-item[1], item[0]))

print('meshMajor numbers:', len(meshItems))
# print(meshItems)

with open('../PKL/all_mesh.pkl', 'wb') as file:
    pickle.dump(all_mesh, file)

# note the meshList as a dictionary
with open('../DATA/mesh_frequency.txt', 'w') as file:
    for mesh, num in meshItems:
        file.write(mesh + '\t' + str(num) + '\n')

meshDict = {mesh[0]: id for id, mesh in enumerate(meshItems)}
with open('../PKL/meshDict.pkl', 'wb') as file:
    pickle.dump(meshDict, file)
print(meshDict)
