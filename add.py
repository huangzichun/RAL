import json

with open("embedding.json","r") as f1:
    word_list = json.load(f1)

with open('embedding2.json','r') as f2:
    word_list2 = json.load(f2)

word_list.update(word_list2)

with open('embedding.json','w') as f3:
    json.dump(word_list, f3)