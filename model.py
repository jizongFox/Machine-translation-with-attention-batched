# coding:utf8
import torch as t
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils.beam_search import CaptionGenerator
import time
import torch.nn.functional as F
import math,numpy as np,pandas as pd
import matplotlib.pyplot as plt

max_length =50

class CaptionModel(nn.Module):
    def __init__(self, opt, word2ix, ix2word):

        super(CaptionModel, self).__init__()
        self.ix2word = ix2word
        self.word2ix = word2ix
        self.opt = opt
        self.fc = nn.Linear(2048, opt.rnn_hidden)

        self.rnn = nn.LSTM(opt.embedding_dim, opt.rnn_hidden, num_layers=opt.num_layers)
        self.classifier = nn.Linear(opt.rnn_hidden, len(word2ix))
        self.embedding = nn.Embedding(len(word2ix), opt.embedding_dim)
        # if opt.share_embedding_weights:
        #     # rnn_hidden=embedding_dim的时候才可以
        #     self.embedding.weight

    def forward(self, img_feats, captions, lengths):
        embeddings = self.embedding(captions)
        # img_feats是2048维的向量,通过全连接层转为256维的向量,和词向量一样
        img_feats = self.fc(img_feats).unsqueeze(0)
        # 将img_feats看成第一个词的词向量 
        embeddings = t.cat([img_feats, embeddings], 0)
        # PackedSequence
        packed_embeddings = pack_padded_sequence(embeddings, lengths)
        outputs, state = self.rnn(packed_embeddings)
        pred = self.classifier(outputs[0])
        return pred, state

    def generate(self, img, eos_token='</EOS>', beam_size=3, max_caption_length=30, length_normalization_factor=0.0):
        """
        根据图片生成描述,主要是使用beam search算法以得到更好的描述
        """
        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.word2ix[eos_token],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        if next(self.parameters()).is_cuda:
            img = img.cuda()
        with t.no_grad():
            img = t.autograd.Variable(img.unsqueeze(0))
            img = self.fc(img).unsqueeze(0)
        sentences, score = cap_gen.beam_search(img)
        sentences = [' '.join([self.ix2word[int(idx)] for idx in sent])
                     for sent in sentences]
        return sentences

    def states(self):
        opt_state_dict = {attr: getattr(self.opt, attr)
                          for attr in dir(self.opt)
                          if not attr.startswith('__')}
        return {
            'state_dict': self.state_dict(),
            'opt': opt_state_dict
        }

    def save(self, path=None, **kwargs):
        if path is None:
            path = '{prefix}_{time}'.format(prefix=self.opt.prefix,
                                            time=time.strftime('%m%d_%H%M'))
        states = self.states()
        states.update(kwargs)
        t.save(states, path)
        return path

    def load(self, path, load_opt=False):
        data = t.load(path, map_location=lambda s, l: s)
        state_dict = data['state_dict']
        self.load_state_dict(state_dict)

        if load_opt:
            for k, v in data['opt'].items():
                setattr(self.opt, k, v)
        return self

    def get_optimizer(self, lr):
        return t.optim.Adam(self.parameters(), lr=lr)

class Encoder(nn.Module):
    def __init__(self, language_conf,embedding_dimension,hidden_size,num_layer):
        super(Encoder, self).__init__()
        self.model_name = 'encoder'
        self.embedding_dimension =embedding_dimension
        self.hidden_size = hidden_size
        self.word2ix = language_conf.word2ix
        self.ix2word = language_conf.ix2word
        self.embedding = nn.Embedding(len(self.word2ix),embedding_dimension)
        self.rnn = nn.GRU(embedding_dimension,hidden_size,num_layers=num_layer,bidirectional=True)

    def forward(self, input_batch_sequences, input_lengths,hidden = None):
        embeddings = self.embedding(input_batch_sequences)
        packed_embeddings = pack_padded_sequence(embeddings,lengths=input_lengths)
        outputs,state = self.rnn(packed_embeddings,hidden)
        outputs, _ = t.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, state
    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name
            name = time.strftime(prefix + '.pth')
        t.save(self.state_dict(), name)
        return name

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(t.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden:
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1) # B T H
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        return F.softmax(attn_energies,dim=1).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(t.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = t.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class BahdanauAttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0.1):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.model_name = 'decoder'
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout_p)
        #self.attn_combine = nn.Linear(hidden_size + embed_size, hidden_size)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''
        dimension check:
        word_input:[B]
        last_hidden:[2,B,H]
        encoder_output:[T*B*H]
        :param word_input:
            word input for current time step, in shape (B) right
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H) T=8.B=19, H=128
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''

        # Get the embedding of the current input word (last output word)

        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)

        # Calculate attention weights and apply to encoder outputs

        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V)
        context = context.transpose(0, 1)  # (1,B,V)

        # Combine embedded input word and attended context, run through RNN

        rnn_input = t.cat((word_embedded, context), 2) #[1,B,H+V]

        #rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different

        output, hidden = self.gru(rnn_input, last_hidden) #seq_len, batch, hidden_size * num_directio
        output = output.squeeze(0)  # (1,B,V)->(B,V) V here should be H

        context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        output = F.log_softmax(self.out(t.cat((output, context), 1)),dim=1)

        # output = F.log_softmax(self.out(output),dim=1)

        # Return final output, hidden state
        return output, hidden , attn_weights

    def save(self, name=None):
        '''
        保存模型，默认使用“模型名字+时间”作为文件名
        '''
        if name is None:
            prefix = 'checkpoints/' + self.model_name
            name = time.strftime(prefix + '.pth')
        t.save(self.state_dict(), name)
        return name

def unpack_batch(word_table):
    largest_length = max([len(x) for x in word_table])

    for i, batch in enumerate(word_table):
        if batch.shape[0]<largest_length:
            new_batch = np.ones((largest_length,batch.shape[1]))*3
            new_batch[:batch.shape[0]]=batch
        else:
            new_batch = batch

        if i == 0:

            machine_translation = new_batch
        else:
            machine_translation = np.concatenate([machine_translation, new_batch], axis=1)
    machine_translation = machine_translation.T
    return machine_translation

def evaluate(encoder,decoder, dataloader,opt, max_length=10):
    encoder.train(False)
    decoder.train(False)
    input_config = dataloader.dataset.input_config
    output_config = dataloader.dataset.output_config
    traslation_results={}
    traslation_results['fr']=np.array([])
    with t.no_grad():
        all_decoded_words = []
        all_input_words=[]
        all_output_words=[]
        attention_matrix=[]
        for ii, ((in_lang,in_lengths),(out_lang,out_lengths)) in enumerate(dataloader):
            all_input_words.append(in_lang.numpy())
            all_output_words.append(out_lang.numpy())
            in_lang=in_lang.to(t.device('cuda:0'))
            out_lang=out_lang.to(t.device('cuda:0'))


        # Run through encoder
            encoder_outputs, encoder_hidden = encoder(in_lang, in_lengths)
        # Create starting vectors for decoder
            decoder_input = t.LongTensor([output_config.word2ix[opt.start]] * in_lang.shape[1]).to(t.device(in_lang.device))
            decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

        # Store output words and attention states
            batch_decoded_words = []
            batch_attention =[]
            # decoder_attentions = t.zeros(max_length + 1, max_length + 1)

        # Run through decoder
            for di in range(max_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                # decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data
                # Choose top word from output
                topv, topi = decoder_output.data.topk(k=1,dim=1)
                ni = topi
                # batch_decoded_words.append([output_config.ix2word[int(x)] for x in ni])
                batch_decoded_words.append(ni.cpu().numpy().squeeze())
                batch_attention.append(decoder_attention.cpu().numpy().squeeze())
                # Next input is chosen word
                decoder_input = t.LongTensor(t.LongTensor(ni.to(t.device("cpu")))).squeeze(1).to(t.device(in_lang.device))
                # if torch.cuda.is_available(): decoder_input = decoder_input.cuda()
            all_decoded_words.append(np.array(batch_decoded_words))
            attention_matrix.append(np.array(batch_attention))

        # 字表后处理
        machine_translation=unpack_batch(all_decoded_words)
        all_inputs = unpack_batch(all_input_words)
        all_outputs = unpack_batch(all_output_words)
        # all_attention = unpack_batch(attention_matrix)

        for ix in range(machine_translation.shape[0]):
            eos_indicator = max_length-1
            try:
                eos_indicator = machine_translation[0].tolist().index(3)
                machine_translation[ix][eos_indicator:]=3
            except:
                1

        machine_translation = pd.DataFrame(machine_translation)
        all_inputs = pd.DataFrame(all_inputs)
        all_outputs = pd.DataFrame(all_outputs)

        machine_translation=machine_translation.applymap(lambda x:output_config.ix2word[int(x)])
        all_inputs = all_inputs.applymap(lambda x: input_config.ix2word[int(x)])
        all_outputs = all_outputs.applymap(lambda x: output_config.ix2word[int(x)])

        idx = np.random.randint(0, machine_translation.shape[0],7)
        for k in range(7):
            print(' '.join(all_inputs.ix[idx[k]].tolist()))
            print(' '.join(all_outputs.ix[idx[k]].tolist()))
            print(' '.join(machine_translation.ix[idx[k]].tolist()))
            print('==' * 10)

        # print(all_inputs.head())
        # print(all_outputs.head())
        # print(machine_translation.head())
        encoder.train(True)
        decoder.train(True)

    return all_inputs,all_outputs,machine_translation

def evaluate_for_one_sentence(sentence,input_config, output_config,encoder,decoder, opt,max_length=10):
    input_language = sentence.split(' ')
    input_language.append('</EOS>')
    input_language = [input_config.word2ix[x] for x in input_language]
    input_ = t.LongTensor(input_language).unsqueeze(1).cuda()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_, [len(input_)])
    # Create starting vectors for decoder
    decoder_input = t.LongTensor([output_config.word2ix[opt.start]] * input_.shape[1]).to(t.device(input_.device))
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    decoded_words=[]
    attentions= []
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data
        # Choose top word from output
        topv, topi = decoder_output.data.topk(k=1, dim=1)
        ni = topi
        attentions.append(decoder_attention.data.cpu().numpy())
        if ni==output_config.word2ix['</EOS>']:
            decoded_words.append(3)
            break
        else:
            decoded_words.append(ni)
        # batch_decoded_words.append([output_config.ix2word[int(x)] for x in ni])
        # Next input is chosen word
        decoder_input = t.LongTensor(t.LongTensor(ni.to(t.device("cpu")))).squeeze(1).to(t.device(input_.device))
        # if torch.cuda.is_available(): decoder_input = decoder_input.cuda()
    return [output_config.ix2word[int(x)] for x  in decoded_words],np.array(attentions).squeeze()

def showAttention(input_sentence, output_words, attentions):
    import matplotlib.ticker as ticker
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['</EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show(block =False)


