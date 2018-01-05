import jieba as jb
import sys
import pickle
import gensim
import os
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text
from gensim.models.word2vec import Word2Vec
import numpy as np

dictF='./dict.txt.big'
trainDataF='./all_sents.txt'
segF='./jieba.pickle'
gensimF='./vectors_gensim.txt'
def readRaw():
    print("\n===Reading RAW===")
    with open(trainDataF,'r',encoding='utf-8') as file:
        content= file.readlines()
        print(str(len(content))+" lines read")
        return content
def jieba(content):
    print("\n===Jieba stage===")
    jb.set_dictionary(dictF)
    jb_res=[]
    i=1
    
    for line in content:
        line=line.strip("\n")
        words = jb.cut(line,cut_all=False)
        tmp=[]
        for word in words:
            tmp.append(word)
        #print(tmp)
        jb_res.append(tmp)
        print(str(i)+ " lines segment done",end="\r")
        i+=1
    with open(segF, 'wb') as handle:
        pickle.dump(jb_res, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return jb_res



if '-j' in sys.argv:
    jieba(readRaw())

if '-w2v' in sys.argv:
    if os.path.exists(segF):
        with open(segF, 'rb') as handle:
            segment = pickle.load(handle)
    else:
        segment = jieba(readRaw())
    model=gensim.models.Word2Vec(size=100,min_count=4000,alpha=0.005)
    model.build_vocab(segment,keep_raw_vocab=False,trim_rule=None,progress_per=10000,update=False)
    model.train(segment,total_examples=model.corpus_count,epochs=model.iter)
    model.wv.save_word2vec_format(gensimF)

if '-vis' in sys.argv:
    


    plt.rcParams['font.sans-serif'] = ['simhei']
    np.random.seed(13)  # For Reproducibility
    model = gensim.models.KeyedVectors.load_word2vec_format(gensimF, unicode_errors='ignore')
    #model = word2vec.Word2Vec.load(modelName)
    vocabs = []
    vecs = []
    for vocab in model.vocab:
        vocabs.append(vocab)
        vecs.append(model[vocab])
    vecs = np.array(vecs)
    vocabs = vocabs
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform(vecs)
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    plt.figure()
    texts = []
    for i, label in enumerate(vocabs):
        x, y = reduced[i,:]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
    # plt.savefig('hp.png', dpi=600)
    plt.show()