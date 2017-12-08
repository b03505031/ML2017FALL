import gensim
import math
trainFile='./training_label.txt'
unlabelFile='./training_nolabel.txt'
gensimFile='./vectors_gensim.txt'
tokens=[]
with open(trainFile,'r',encoding='utf-8') as file:
    lines=file.readlines()
for i in range(len(lines)):
    tmpline=lines[i].split(' +++$+++ ')[1]
    tmpline=tmpline.replace(" ' ", "")
    tmpline=tmpline.replace("\n", "")
    tmp=tmpline.split(' ')
    tokens.append(tmp)

with open(unlabelFile,'r',encoding='utf-8') as file:
    lines=file.readlines()
for i in range(len(lines)):
    lines[i]=lines[i].replace(" ' ", "")
    lines[i]=lines[i].replace("\n", "")
    tmp=lines[i].split(' ')
    tokens.append(tmp)
    
model=gensim.models.Word2Vec(size=300,min_count=13,alpha=0.005)
model.build_vocab(tokens,keep_raw_vocab=False,trim_rule=None,progress_per=10000,update=False)
model.train(tokens,total_examples=model.corpus_count,epochs=model.iter)
model.wv.save_word2vec_format(gensimFile)