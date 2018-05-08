#coding:utf8
import torch as t,functools
from torch.utils import data
import os
from PIL import Image
import torchvision as tv
import numpy as np

IMAGENET_MEAN =  [0.485, 0.456, 0.406]
IMAGENET_STD =  [0.229, 0.224, 0.225]

# - 区分训练集和验证集
# - 不是随机返回每句话，而是根据index%5
# - 

# def create_collate_fn():
#     def collate_fn():
#         pass
#     return collate_fn

def collate_fn(input_output_pairs, padding, eos, max_length):
    '''
    将多个样本拼接在一起成一个batch
    输入： list of data，形如
    [(img1, cap1, index1), (img2, cap2, index2) ....]

    拼接策略如下：
    - batch中每个样本的描述长度都是在变化的，不丢弃任何一个词\
      选取长度最长的句子，将所有句子pad成一样长
    - 长度不够的用</PAD>在结尾PAD
    - 没有START标识符
    - 如果长度刚好和词一样，那么就没有</EOS>

    返回：
    - imgs(Tensor): batch_sie*2048
    - cap_tensor(Tensor): batch_size*max_length
    - lengths(list of int): 长度为batch_size
    - index(list of int): 长度为batch_size
    '''
    pairs=np.array(input_output_pairs)
    seq_pairs = sorted(zip(pairs[:,0], pairs[:,1]), key=lambda p: len(p[0]), reverse=True)
    batch_size = len(seq_pairs)
    input_batch,output_batch = zip(*seq_pairs)
    input_lengths = [len(x) for x in input_batch]
    output_lengths = [len(x) for x in output_batch]
    input_max_length,output_max_length = max([len(x) for x in input_batch])+1,max([len(x) for x in output_batch])+1
    input_tensor, output_tensor = t.LongTensor(input_max_length,batch_size).fill_(padding),t.LongTensor(output_max_length,batch_size).fill_(padding)

    # input_put eos
    for i, c in enumerate(input_batch):
        end_cap = len(c)
        input_tensor[end_cap, i] = eos
        input_tensor[:end_cap, i].copy_(t.LongTensor(c[:end_cap]))
    # output eos
    for i, c in enumerate(output_batch):
        end_cap = len(c)
        output_tensor[end_cap, i] = eos
        output_tensor[:end_cap, i].copy_(t.LongTensor(c[:end_cap]))

    return (input_tensor,input_lengths),(output_tensor,output_lengths)

def create_collate_fn(padding,eos,max_length=50):

    # make it adapted to windows multiprocessing scenario. Instead, in linux you can use:
    # return lambda imgs: collate_fn(img_cap=imgs,padding=padding,eos=eos,max_length=max_length)

    partial_parameters={"padding":padding, "eos":eos, "max_length":max_length}
    return functools.partial(collate_fn,**partial_parameters)


class CaptionDataset(data.Dataset):
    def __init__(self,opt,train=True):
        '''
        Attributes:
            _data (dict): 预处理之后的数据，包括所有图片的文件名，以及处理过后的描述
            all_imgs (tensor): 利用resnet50提取的图片特征，形状（200000，2048）
            caption(list): 长度为20万的list，包括每张图片的文字描述
            ix2id(dict): 指定序号的图片对应的文件名
            start_(int): 起始序号，训练集的起始序号是0，验证集的起始序号是190000，即
                前190000张图片是训练集，剩下的10000张图片是验证集
            len_(init): 数据集大小，如果是训练集，长度就是190000，验证集长度为10000
            traininig(bool): 是训练集(True),还是验证集(False)
        '''
        self.train= True
        self.opt = opt
        data = t.load(opt.caption_data_path)
        word2ix = data['word2ix']
        self.captions = data['caption']
        self.padding = word2ix.get(data.get('padding'))
        self.end = word2ix.get(data.get('end'))
        self._data = data
        self.ix2id = data['ix2id']
        self.all_imgs = t.load(opt.img_feature_path)

    def __getitem__(self,index):
        '''
        返回：
        - img: 图像features 2048的向量
        - caption: 描述，形如LongTensor([1,3,5,2]),长度取决于描述长度
        - index: 下标，图像的序号，可以通过ix2id[index]获取对应图片文件名
        '''
        img = self.all_imgs[index]
        caption = self.captions[index]
        # 5句描述随机选一句
        rdn_index = np.random.choice(len(caption),1)[0]
        caption = caption[rdn_index]
        return img,t.LongTensor(caption),index

    def __len__(self):
        return len(self.ix2id)
        
def get_dataloader(opt,training=True):
    dataset = CaptionDataset(opt)
    dataloader = data.DataLoader(dataset,
                    batch_size=opt.batch_size,
                    shuffle=opt.shuffle,
                    num_workers=opt.num_workers,
                   collate_fn=create_collate_fn(dataset.padding,dataset.end)
    )
    return dataloader


class langDataset(data.Dataset):
    def __init__(self,input_lang,output_lang,input_conf,output_config,full_config,train =True):
        '''
        Attributes:
            _data (dict): 预处理之后的数据，包括所有图片的文件名，以及处理过后的描述
            all_imgs (tensor): 利用resnet50提取的图片特征，形状（200000，2048）
            caption(list): 长度为20万的list，包括每张图片的文字描述
            ix2id(dict): 指定序号的图片对应的文件名
            start_(int): 起始序号，训练集的起始序号是0，验证集的起始序号是190000，即
                前190000张图片是训练集，剩下的10000张图片是验证集
            len_(init): 数据集大小，如果是训练集，长度就是190000，验证集长度为10000
            traininig(bool): 是训练集(True),还是验证集(False)
        '''
        self.pairs = np.stack([input_lang,output_lang])
        self.input_config = input_conf
        self.output_config = output_config
        self.padding = input_conf.word2ix.get(full_config.padding)
        self.end = input_conf.word2ix.get(full_config.end)
        if train ==True:

            self.input_lang = input_lang[:int(len(input_lang)*0.9)]
            self.output_lang = output_lang[:int(len(input_lang)*0.9)]

        else:

            self.input_lang = input_lang[int(len(input_lang)*0.9):]
            self.output_lang = output_lang[int(len(input_lang)*0.9):]

    def __getitem__(self, index):

        input_sentence = self.input_lang[index]
        output_sentence = self.output_lang[index]

        return input_sentence,output_sentence

    def __len__(self):
        return int(len(self.input_lang)/10)

def language_DataLoader(datas,configs,opt, train=True):
    input_,output_ = datas
    input_c,output_c = configs
    dataset = langDataset(input_,output_,input_c,output_c,opt,train=train)
    dataloader = data.DataLoader(dataset,
                    batch_size=opt.batch_size,
                    shuffle=opt.shuffle,
                    num_workers=opt.num_workers,
                    collate_fn=create_collate_fn(dataset.padding,dataset.end),
    )
    return dataloader



if __name__=='__main__':
    from config import Config
    opt = Config()
    dataloader = get_dataloader(opt) 
    for ii,data in enumerate(dataloader):
        print(ii,data)
        break
