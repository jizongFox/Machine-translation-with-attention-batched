#coding:utf8
import torch as t
import numpy as np
import json
import jieba
import tqdm
import operator
import pandas as pd
# from main2 import  opt
# START='</START>'
# MAX_LENS = 25,
class language_config (object):
    def __init__(self,name,opt) -> None:
        super().__init__()
        self.opt = opt
        self.language_name = name

        self.word2ix={}
        self.ix2word={}
        self.wordcount={}
        self.word_index_begin = 1

    def cut_sentence(self,sentence):
        return sentence.split(' ')

    def update_dicts(self,list_of_word):
        for word in list_of_word:
            self.wordcount[word]= self.wordcount.get(word,0)+1

    def delete_low_frequency_words(self):
        sorted_word_list = sorted(self.wordcount.items(),key = operator.itemgetter(1),reverse=True)
        sorted_word_list = sorted_word_list[:self.opt.max_words]
        sorted_word_list = [w[0] for w in sorted_word_list]

        list_of_frequecy_word = [k for k,v in self.wordcount.items() if v >self.opt.min_appear]
        merged_list = [x for x in sorted_word_list if x in list_of_frequecy_word]
            # list(set(sorted_word_list)  & set(list_of_frequecy_word))
        merged_list= [self.opt.start, self.opt.unknown, self.opt.padding, self.opt.end] + merged_list
        return merged_list

    def generatefulldicts(self,dataset):
        _=[self.update_dicts(self.cut_sentence(sentence)) for sentence in dataset]
        merged_list = self.delete_low_frequency_words()
        self.word2ix = {word: ix for ix, word in enumerate(merged_list)}
        self.ix2word = {ix: word for word, ix in self.word2ix.items()}
        self.dataset =dataset

    def __str__(self) -> str:
        return 'length of word2ix:%d'%len(self.word2ix)

    def ix_data(self):
        ix_captions = [[self.word2ix.get(word, self.word2ix.get(self.opt.unknown)) for word in self.cut_sentence(sentence)] for sentence in self.dataset]
        return np.array(ix_captions)


if __name__=='__main__':
    en_dataset = pd.read_csv('data.csv')['en'].values
    fr_dataset = pd.read_csv('data.csv')['fr'].values
    en_config = language_config('en')
    en_config.generatefulldicts(en_dataset)
    en_source = en_config.ix_data()
    print(en_source.shape)

    fr_config = language_config('fr')
    fr_config.generatefulldicts(fr_dataset)
    fr_source = fr_config.ix_data()
    print(fr_source.shape)
    print('data cleaning is done')






'''


def process(**kwargs):
    opt = Config()
    for k,v in kwargs.items():
        setattr(opt,k,v)
    
    with open(opt.annotation_file) as f:
        data = pd.read_csv(f)
    
    # 8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg -> 0
    id2ix = {item['image_id']:ix for ix,item in enumerate(data) }
    # 0-> 8f00f3d0f1008e085ab660e70dffced16a8259f6.jpg
    ix2id = {ix:id for id,ix in (id2ix.items())}
    assert id2ix[ix2id[10]] == 10

    captions = [item['caption'] for item in data]
    # 分词结果

    cut_captions = [ [list(jieba.cut(ii,cut_all=False)) for ii in item ] for item in tqdm.tqdm(captions) ]

   
    word_nums = {}  # '快乐'-> 10000 (次)
    def update(word_nums):
        def fun(word):
            word_nums[word] = word_nums.get(word,0)+1
            return None
        return fun
    lambda_ = update(word_nums)
    _ =  {lambda_(word) for sentences in cut_captions for sentence in sentences for  word  in sentence}

    vocabs = list(word_nums.keys())
    # [ (10000,u'快乐')，(9999,u'开心') ...]
    word_nums_list = sorted([(num,word) for word,num in word_nums.items() ],reverse=True)

    #### 以上的操作是无损，可逆的操作###############################
    #**********以下会删除一些信息******************

    #1. 丢弃词频不够的词
    #2. ~~丢弃长度过长的词~~ 

    words = [word[1] for word in  word_nums_list[:opt.max_words] if word[0]>=opt.min_appear]
    words =  [opt.unknown,opt.padding,opt.end] + words
    word2ix = {word:ix for ix,word in enumerate(words)}
    ix2word = {ix:word for word,ix in word2ix.items()}
    assert word2ix[ix2word[123]] == 123

    ix_captions =[ [  [ word2ix.get(word,word2ix.get(opt.unknown)) for word in sentence] 
                        for sentence in item]
                        for item in cut_captions]
    # readme=u
    # word：词
    # ix:index
    # id:图片名
    # caption: 分词之后的描述，通过ix2word可以获得原始中文词
    
    results = {
        'caption':ix_captions,
        'word2ix':word2ix,
        'ix2word':ix2word,
        'ix2id':ix2id,
        'id2ix':id2ix,
        'padding':'</PAD>',
        'end':'</EOS>',
        'readme':readme
    }
    t.save(results,opt.save_path)
    print('save file in %s' %opt.save_path)

    def test(ix,ix2=4):
        results =  t.load(opt.save_path)
        ix2word = results['ix2word']
        examples = results['caption'][ix][4]
        sentences_p = (''.join([ix2word[ii] for ii in examples]) )
        sentences_r = data[ix]['caption'][ix2]
        assert sentences_p == sentences_r,'test failed'
        

    test(1000)
    print('test success')

if __name__ == '__main__':
    import fire
    fire.Fire()
    # python data_preprocess.py process --annotation-file=/data/annotation.json --max-words=5000
    
'''