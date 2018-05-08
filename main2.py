# coding:utf8
import pandas as pd, matplotlib.pyplot as plt
import torch
from data_preprocess import language_config
from model import Encoder,BahdanauAttnDecoderRNN,evaluate,evaluate_for_one_sentence,showAttention
from masked_cross_entropy import masked_cross_entropy
from tqdm import  tqdm
import numpy as np
from torchnet.meter import AverageValueMeter

class Config:
    caption_data_path='caption.pth'# 经过预处理后的人工描述信息
    # img_path='/home/cy/caption_data/'
    img_path='./ai_challenger_caption_train_20170902/caption_train_images_20170902/'
    img_feature_path = 'results.pth' # 所有图片的features,20w*2048的向量
    scale_size = 300
    img_size = 224
    batch_size=16
    shuffle = True
    num_workers = 4
    rnn_hidden = 512
    embedding_dim = 128
    num_layers = 4
    share_embedding_weights=False
    prefix='checkpoints/caption'#模型保存前缀
    env = 'caption'
    plot_every = 10
    debug_file = '/tmp/debugc'

    annotation_file   = 'data.csv'
    start = '</SOS>'
    unknown = '</UNKNOWN>'
    end='</EOS>'
    padding='</PAD>'
    max_words=4000
    min_appear=2
    save_path='checkpoints/'
    # save_path=None
    lr= 1e-6


if __name__=='__main__':
    np.random.seed(1)
    opt = Config()
    en_dataset = pd.read_csv('data.csv')['en'].values
    fr_dataset = pd.read_csv('data.csv')['fr'].values
    # permute the order
    shuffle_order = np.random.permutation(len(en_dataset))
    en_dataset = en_dataset[shuffle_order]
    fr_dataset = fr_dataset[shuffle_order]

    en_config = language_config('en',opt)

    en_config.generatefulldicts(en_dataset)
    en_source = en_config.ix_data()
    # print(en_source.shape)

    fr_config = language_config('fr',opt)
    fr_config.generatefulldicts(fr_dataset)
    fr_source = fr_config.ix_data()
    # print(fr_source.shape)
    #去掉unknown的项目
    pairs= [[fr, en] for en,fr in zip(en_source,fr_source) if ((en_config.word2ix[opt.unknown] not in en) and fr_config.word2ix[opt.unknown] not in fr)]
    pairs=np.array(pairs)
    fr_source,en_source = pairs[:,0],pairs[:,1]

    print('data cleaning is done')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from data import language_DataLoader
    train_dataloader = language_DataLoader((fr_source,en_source),(fr_config,en_config),opt,train=True)
    test_dataloader = language_DataLoader((fr_source,en_source),(fr_config,en_config),opt,train=False)

    encoder = Encoder(fr_config,embedding_dimension=opt.embedding_dim,hidden_size=opt.rnn_hidden,num_layer=opt.num_layers)
    optimizer1 = torch.optim.Adam(encoder.parameters(),lr=opt.lr)
    decoder = BahdanauAttnDecoderRNN(opt.rnn_hidden, opt.embedding_dim, len(en_config.word2ix), n_layers=2, dropout_p=0.1)
    # decoder =
    optimizer2 = torch.optim.Adam(decoder.parameters(),lr=opt.lr)
    if opt.save_path:
        encoder.load_state_dict(torch.load(opt.save_path+'encoder.pth'))
        decoder.load_state_dict(torch.load(opt.save_path+'decoder.pth'))
        print('load update model')
    encoder.to(device)
    decoder.to(device)
    loss_meter = AverageValueMeter()
    '''
    for epoch in range(200):
        loss_meter.reset()

        for ii, ((in_lang,in_lengths),(out_lang,out_lengths)) in tqdm(enumerate(train_dataloader)):
            in_lang = in_lang.to(device)
            out_lang = out_lang.to(device)
            optimizer1.zero_grad()
            optimizer2.zero_grad()

            encoder_outputs, encoder_hidden = encoder(in_lang,in_lengths) # MAX_LENGTH, BATCH_SIZE, EMBEDDING DIMENSION // n_layer, BATCH_SIZE, EMBEDDING DIMENSION
            # Prepare input and output variables
            decoder_input = torch.LongTensor([fr_config.word2ix[opt.start]] * in_lang.shape[1]).to(device)
            decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

            # output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs = torch.zeros(max(out_lengths)+1, out_lang.shape[1], decoder.output_size).to(device)

            decoder_output, decoder_hidden, attention_matrix = decoder(decoder_input, decoder_hidden, encoder_outputs)
            all_decoder_outputs[0] = decoder_output
            for ii in range(0,out_lang.shape[0]-2):
                word_batch = out_lang[ii,:]
                decoder_output, decoder_hidden,_= decoder(word_batch, decoder_hidden, encoder_outputs)
                all_decoder_outputs[ii+1] = decoder_output
                # decoder_input = target_batches[t]  # Next input is current target

            # Loss calculation and backpropagation
            loss = masked_cross_entropy(
                all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
                out_lang.transpose(0, 1).contiguous(),  # -> batch x seq
                out_lengths)
            loss.backward()
            loss_meter.add(loss.item())

            # Clip gradient norms
            clip = 50
            ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
            dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

            # Update parameters with optimizers
            optimizer1.step()
            optimizer2.step()

        # if epoch % 5 == 0:
        # idx=np.random.randint(0,in_lang.shape[1])
        # print(' '.join([fr_config.ix2word[int(x)] for x in in_lang[:, idx]]))
        # print(' '.join([en_config.ix2word[int(x)] for x in out_lang[:, idx]]))
        # print(' '.join([en_config.ix2word[int(x)] for x in torch.max(all_decoder_outputs[:, idx, :], 1)[1]]))
        encoder.save()
        decoder.save()
        print(loss_meter.value()[0])

        if epoch%20==0:
            val_input, val_out, val_translation =  evaluate(encoder,decoder,test_dataloader,opt)
            val_input.to_csv('input.csv',encoding='utf8')
            val_out.to_csv('output.csv',encoding='utf8')
            val_translation.to_csv('translation.csv',encoding='utf8')
            
            
            
    '''
    sentenceto_test='elle n a peur de rien .'
    output,attention = evaluate_for_one_sentence(sentenceto_test, input_config=fr_config, output_config= en_config, encoder=encoder,decoder=decoder,opt=opt)
    showAttention(sentenceto_test, output, attention)
    1



