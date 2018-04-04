# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils
import numpy as np
from .CaptionModel import CaptionModel

class OldModel(CaptionModel):
    def __init__(self, opt):
        super(OldModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.encoder_rnn_size = opt.encoder_rnn_size
        self.phrase_rnn_size = opt.phrase_rnn_size
        self.decoder_rnn_size = opt.decoder_rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.max_seg_len = opt.max_seg_len
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.ss_prob = 0.0 # Schedule sampling probability
        self.init_encoder = opt.init_encoder
        self.init_phrase = opt.init_phrase
        self.init_decoder = opt.init_decoder
        if self.init_encoder: 
            self.fc_encoder_sta = nn.Linear(self.fc_feat_size, self.encoder_rnn_size) # feature to rnn_size
            if self.rnn_type == 'lstm':
                self.fc_encoder_mem = nn.Linear(self.fc_feat_size, self.encoder_rnn_size)
        else :
            self.fc_encoder_sta = None
            self.fc_encoder_mem = None
        if self.init_phrase: 
            self.fc_phrase_sta = nn.Linear(self.fc_feat_size, self.encoder_rnn_size) # feature to rnn_size
            if self.rnn_type == 'lstm':
                self.fc_phrase_mem = nn.Linear(self.fc_feat_size, self.encoder_rnn_size)
        else :
            self.fc_phrase_sta = None
            self.fc_phrase_mem = None
            
        if self.init_decoder: 
            self.fc_decoder_sta = nn.Linear(self.fc_feat_size, self.encoder_rnn_size) # feature to rnn_size
            if self.rnn_type == 'lstm':
                self.fc_decoder_mem = nn.Linear(self.fc_feat_size, self.encoder_rnn_size)
        else :
            self.fc_decoder_sta = None
            self.fc_decoder_mem = None
            
        self.embed = nn.Embedding(self.vocab_size + 1, self.encoder_rnn_size)
        self.logit = nn.Linear(self.decoder_rnn_size, self.vocab_size + 1)
        self.feat_output_layer = nn.Linear(self.fc_feat_size, self.decoder_rnn_size)
        self.dropout = nn.Dropout(self.drop_prob_lm)
        self.encoder_rnn = getattr(nn, self.rnn_type.upper())(self.encoder_rnn_size, 
                self.encoder_rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        self.decoder_rnn = getattr(nn, self.rnn_type.upper())(self.phrase_rnn_size+self.encoder_rnn_size, 
                self.decoder_rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, init_layer, init_size, fc_feats,batch_size):
        if getattr(self,init_layer+'_sta') != None:
            init_state = getattr(self,init_layer+'_sta')(fc_feats).view(-1, init_size).transpose(0, 1)
            if self.rnn_type == 'lstm':
                init_memory = getattr(self,init_layer+'_mem')(fc_feats).view(-1, init_size).transpose(0, 1)
                return (init_state, init_memory)
            else:
                return init_state
        else :
            init_state = Variable(torch.FloatTensor(batch_size, init_size).zero_()).cuda().view(-1,batch_size, init_size)
            if self.rnn_type == 'lstm':
                init_memory = Variable(torch.FloatTensor(batch_size, init_size).zero_()).cuda().view(-1,batch_size, init_size) 
                return (init_state, init_memory)
            else :
                return init_state

    def forward(self, fc_feats, seq, phrase_mask_0, phrase_mask_1, seq_mask, seg_mask, remove_len):
        batch_size = fc_feats.size(0)
        remove_len = remove_len
        mean_fc_feats = torch.mean(fc_feats, 1)
        encoder_state = self.init_hidden('fc_encoder',self.encoder_rnn_size, mean_fc_feats, batch_size)
        phrase_state = self.init_hidden('fc_phrase',self.phrase_rnn_size, mean_fc_feats, batch_size)
        decoder_state = self.init_hidden('fc_decoder',self.decoder_rnn_size, mean_fc_feats, batch_size)
        outputs = []
        encoder_outputs = []
        xts = []
        count = 0
        encoder_output_ = None
        num_seg, len_seq, _ = phrase_mask_0.size()
        phrase_mask_0 = phrase_mask_0[:,:-remove_len+1,:]
        phrase_mask_1 = phrase_mask_1[:,:-remove_len,:]
        
        def get_pre_now(output, output_, mask):
            return ((mask.unsqueeze(0)).unsqueeze(-1) * output ) + ((1-(mask.unsqueeze(0)).unsqueeze(-1)) * output_ )
        
        self.decoder_rnn.flatten_parameters()
        self.encoder_rnn.flatten_parameters()
        self.core.phrase_rnn.flatten_parameters()
        for i in range(len_seq-remove_len+1):
            count = count + 1
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = mean_fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[i,:].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[i,:].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[i,:].clone()          
            # break if all the sequences end
            xt = self.embed(it)
            
            encoder_output, encoder_state = self.encoder_rnn(xt.unsqueeze(0), encoder_state)
            if i >= 1:
                encoder_output = get_pre_now(encoder_output, encoder_output_, seq_mask[i])
                encoder_state_0 = get_pre_now(encoder_state[0], encoder_state_[0], seq_mask[i])
                encoder_state_1 = get_pre_now(encoder_state[1], encoder_state_[1], seq_mask[i])
                encoder_state = (encoder_state_0, encoder_state_1)
            encoder_output_ = encoder_output
            encoder_state_ = encoder_state
            encoder_outputs.append(encoder_output.squeeze(0))
            xts.append(xt)
            
        xts = torch.cat([_.unsqueeze(1) for _ in xts], 1) # len_sen, batch_size, embedding_dim  --> batch_size, len_sen, embedding_dim
        encoder_outputs = torch.cat([_.unsqueeze(1) for _ in encoder_outputs], 1) # batch_size, len_seq, encoder_size
        phrase_mask_0_t = phrase_mask_0.permute(2,0,1)
        tu_h_dot_pmask0 = torch.bmm(phrase_mask_0_t, encoder_outputs)/torch.sum(phrase_mask_0_t + 1e-9,-1,True)# batch_size, num_seg, tu_dim, 
        phrase_mask_1_t = phrase_mask_1.permute(2,1,0)
        phrase_outputs = []
        att_feats = []
        alphas = []
        
        for i in range(num_seg):
            phrase_output, phrase_state, att_feat, alpha = self.core(tu_h_dot_pmask0[:,i,:], mean_fc_feats, fc_feats, phrase_state)
            if i >= 1:
                phrase_output = get_pre_now(phrase_output, phrase_output_, seg_mask[i])
                phrase_state_0 = get_pre_now(phrase_state[0], phrase_state_[0], seg_mask[i])
                phrase_state_1 = get_pre_now(phrase_state[1], phrase_state_[1], seg_mask[i])
                phrase_state = (phrase_state_0, phrase_state_1)
                att_feat = get_pre_now(att_feat.unsqueeze(0), att_feat_.unsqueeze(0), seg_mask[i])
                att_feat = att_feat.squeeze(0)
                alpha = get_pre_now(alpha.unsqueeze(0), alpha_.unsqueeze(0), seg_mask[i])
                alpha = alpha.squeeze(0)
            phrase_output_ = phrase_output
            phrase_state_ = phrase_state
            att_feat_ = att_feat
            alpha_ = alpha
            phrase_outputs.append(phrase_output.squeeze(0))
            att_feats.append(att_feat)
            alphas.append(alpha)
        alphas = torch.cat([_.unsqueeze(1) for _ in alphas], 1)
        phrase_outputs = torch.cat([_.unsqueeze(1) for _ in phrase_outputs], 1)
        att_feats = torch.cat([_.unsqueeze(1) for _ in att_feats], 1)
        decoder_inputs = torch.bmm(phrase_mask_1_t, phrase_outputs) # batch_size, len_seq, encoder_size
        att_feats = torch.bmm(phrase_mask_1_t, att_feats)
        decoder_outputs = []
        for i in range(len_seq-remove_len):
            decoder_input = torch.cat([decoder_inputs[:,i,:],xts[:,i+remove_len-1,:]],-1)
            decoder_output, decoder_state = self.decoder_rnn(decoder_input.unsqueeze(0), decoder_state)
            if i >= 1:
                decoder_output = get_pre_now(decoder_output, decoder_output_, seq_mask[i])
                decoder_state_0 = get_pre_now(decoder_state[0], decoder_state_[0], seq_mask[i])
                decoder_state_1 = get_pre_now(decoder_state[1], decoder_state_[1], seq_mask[i])
                decoder_state = (decoder_state_0,decoder_state_1)
            decoder_state_ = decoder_state
            decoder_output_ = decoder_output
            feat_output = self.feat_output_layer(att_feats[:,i,:])
            output = F.log_softmax(self.logit(self.dropout(torch.add(decoder_output.squeeze(0), feat_output))))
            outputs.append(output)
        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), alphas

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)

        output, state, _ = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)))

        return logprobs, state
        
    def transpose_state(self, state):
        # transpose batch_size, len_state, 1, hid_dim --> len_state, 1, batch_size, state_dim
        new_state = []
        for i in range(len(state[0])):
            new_state.append(Variable(torch.zeros(state[0][0].size(-2), len(state), state[0][0].size(-1)), requires_grad=False).cuda())
        for beam_idx in range(len(state)):
            for state_idx in range(len(state[beam_idx])):
                #  copy over state in previous beam q to new beam at vix
                '''
                if hasattr(state[beam_idx][state_idx],'data'):
                    new_state[state_idx][:,beam_idx,:] = state[beam_idx][state_idx].data # dimension one is time step
                else :
                '''
                new_state[state_idx][:,beam_idx,:] = state[beam_idx][state_idx] # dimension one is time step
        return new_state
        
    def get_state(self, state, beam_idx):
        # state shape: 2, 1, batch_size, hid_dim
        # out shape: 2, 1, hid_dim
        new_state = [[]] * len(state)
        for state_ix in range(len(state)):
        #  copy over state in previous beam q to new beam at vix
                new_state[state_ix] = state[state_ix][:,beam_idx] # dimension one is time step
        return new_state
        
    def beam_search(self, encoder_state, phrase_state, decoder_state, tmp_mean_fc_feats, tmp_fc_feats, opt):
        pre_phrase_np = np.array([[1,self.seg_index[0]]]).astype('int64') # <bos>,<SEG>
        pre_phrase = torch.from_numpy(pre_phrase_np).long().cuda()
        live_k = 1
        dead_k = 0
        beam_size = opt.get('beam_size', 5)
        k = beam_size
        beam_logprobs_sum = torch.zeros(live_k).cuda()
        hyp_sample =  [[]] * live_k
        dead_phrase = 0
        beam_sample = []
        sample = []
        sample_logpro = []
        seg_len = [2]
        for i in range(self.seq_length):
            encoder_state_l = []
            encoder_output_l = []
            for j in range(max(seg_len)):
                pre_phrase_embed = self.embed(Variable(pre_phrase[:,j], requires_grad=False))
                encoder_output, encoder_state = self.encoder_rnn(pre_phrase_embed.unsqueeze(0), encoder_state)
                encoder_output_l.append(encoder_output.squeeze(0))
                encoder_state_l.append(encoder_state)
            #encoder_state_l shape: seg_len, 2 , 1, batch_size, hid_dim
            encoder_output = [] # len_seg, batch_size, hid_dim
            encoder_state = []
            phrase_input = []
            encoder_output_l = torch.cat([_.unsqueeze(0) for _ in encoder_output_l], 0)
            #print('encoder_output_l',encoder_output_l.size())
            for j in range(len(seg_len)):
                encoder_output.append(encoder_output_l[seg_len[j]-1, j,:])
                encoder_state.append(self.get_state(encoder_state_l[seg_len[j]-1],j))           # batch_size, encode_dim
                phrase_input.append(torch.mean(encoder_output_l[:seg_len[j], j,:],0))
            #encoder_state = torch.cat([_.unsqueeze(1) for _ in encoder_state], 1)
            encoder_state = self.transpose_state(encoder_state)
            # len_state, 1, batch_size, hid_dim
            phrase_input = torch.cat([_.unsqueeze(0) for _ in phrase_input], 0)
            batch_size = phrase_input.size(0)
            #print('tmp_fc_feats_t',tmp_fc_feats)
            tmp_mean_fc_feats_t = tmp_mean_fc_feats.expand(batch_size,-1).contiguous()
            tmp_fc_feats_t = tmp_fc_feats.expand(batch_size,-1,-1).contiguous()

            phrase_output, phrase_state, att_feat, alpha = self.core(phrase_input, tmp_mean_fc_feats_t, tmp_fc_feats_t, phrase_state)
            phrase_state = phrase_state # 1, batch_size, hid_dim
            decoder_inputs = phrase_output
            live_phrase = live_k
            dead_phrase = dead_k
            next_phrase = []
            hyp_phrase = [[]] * (k-dead_k)
            finish_phrase = []
            finish_phrase_en_state = []
            finish_phrase_ph_state = []
            finish_phrase_att_feats = []
            finish_phrase_de_state = []
            finish_pros_sum = []
            finial_word = []
            finish_sample = []
            for i in range(len(seg_len)):
                finial_word.append(pre_phrase[i, seg_len[i]-1])
            seg_len = []
            finial_word_tensor = torch.LongTensor(finial_word).cuda()
            for j in range(self.max_seg_len):
                pre_word_embed = self.embed(Variable(finial_word_tensor, requires_grad=False))
                decoder_input = torch.cat([decoder_inputs, pre_word_embed],-1)
                decoder_output, decoder_state = self.decoder_rnn(decoder_input.unsqueeze(0), decoder_state)  
                #decoder_output shape: 1, batch_size, hid_dim
                
                decoder_output = decoder_output[-1]
                feat_output = self.feat_output_layer(att_feat)
                logprobs = F.log_softmax(self.logit(self.dropout(torch.add(decoder_output, feat_output))))
                logprobsf = logprobs.data
                beam_logprobs_sum_t = beam_logprobs_sum.unsqueeze(1) - logprobsf
                #beam_logprobs_sum = beam_logprobs_sum_t
                num_samples, voc_size = logprobsf.size()
                logprobsf_r = beam_logprobs_sum_t.view(-1)
                sort_logprobsf, sort_index = torch.sort(logprobsf_r)
                sort_index = sort_index[:(beam_size-dead_phrase)]
                beam_index = sort_index / voc_size
                word_index = sort_index % voc_size
                new_decoder_state = []
                new_decoder_input = []
                new_phrase_state = []
                new_encoder_state = []
                new_logpro_sum = []
                new_att_feat = []
                new_phrase = []
                new_word = []
                new_sample = []
                new_live_phrase = 0
                new_live_k = 0
                for i in range(len(word_index)):
                    if word_index[i] == 0:
                        sample.append(hyp_sample[beam_index[i]] + [word_index[i]])
                        sample_logpro.append(beam_logprobs_sum_t[beam_index[i]][word_index[i]])
                        dead_k += 1
                        dead_phrase += 1
                    else :
                        new_live_k = new_live_k + 1
                        if word_index[i] in self.seg_index :
                            dead_phrase = dead_phrase + 1
                            seg_len.append(j+1)
                            finish_phrase.append(hyp_phrase[beam_index[i]] + [word_index[i]])
                            finish_phrase_en_state.append(self.get_state(encoder_state,beam_index[i]))
                            finish_phrase_ph_state.append(self.get_state(phrase_state,beam_index[i]))
                            finish_phrase_de_state.append(self.get_state(decoder_state,beam_index[i]))
                            finish_pros_sum.append(beam_logprobs_sum_t[beam_index[i]][word_index[i]])
                            finish_sample.append(hyp_sample[beam_index[i]] + [word_index[i]])
                        else :
                            new_live_phrase = new_live_phrase + 1
                            new_encoder_state.append(self.get_state(encoder_state,beam_index[i]))
                            new_phrase_state.append(self.get_state(phrase_state,beam_index[i]))
                            new_decoder_input.append(decoder_inputs[beam_index[i]])
                            new_att_feat.append(att_feat[beam_index[i]])
                            new_decoder_state.append(self.get_state(decoder_state,beam_index[i]))
                            new_sample.append(hyp_sample[beam_index[i]] + [word_index[i]])
                            new_phrase.append(hyp_phrase[beam_index[i]] + [word_index[i]])
                            new_word.append(word_index[i])
                            new_logpro_sum.append(beam_logprobs_sum_t[beam_index[i]][word_index[i]])
                hyp_sample = new_sample
                hyp_phrase = new_phrase
                live_k = new_live_k
                live_phrase = new_live_phrase
                if dead_k >= beam_size or dead_phrase >= beam_size:
                    break
                if live_phrase == 0 or new_live_k == 0:
                    break                            
                decoder_inputs = torch.cat([_.unsqueeze(0) for _ in new_decoder_input],0)
                phrase_state = self.transpose_state(new_phrase_state)
                att_feat =  torch.cat([_.unsqueeze(0) for _ in new_att_feat],0)
                finial_word_tensor =  torch.LongTensor(new_word).cuda()
                encoder_state = self.transpose_state(new_encoder_state)
                decoder_state = self.transpose_state(new_decoder_state)
                beam_logprobs_sum = torch.FloatTensor(new_logpro_sum).cuda()


            if live_phrase > 0 :
                for i in range(live_phrase):
                    seg_len.append(self.max_seg_len)
                    finish_pros_sum.append(beam_logprobs_sum[i])
                    finish_phrase.append(hyp_phrase[i])
                    finish_phrase_en_state.append(self.get_state(encoder_state,i))
                    finish_phrase_ph_state.append(self.get_state(phrase_state,i))
                    finish_phrase_de_state.append(self.get_state(decoder_state,i))
                    finish_sample.append(hyp_sample[i])
            if dead_k >= beam_size:
                break
            live_k = len(finish_phrase)
            hyp_sample = finish_sample
            beam_logprobs_sum = torch.FloatTensor(finish_pros_sum).cuda()
            if len(finish_phrase) == 0:
                break

            pre_phrase = np.zeros([k-dead_k, max(seg_len)]).astype('int64')
            for i in range(len(finish_phrase)):
                pre_phrase[i,:seg_len[i]] = finish_phrase[i]
            pre_phrase = torch.LongTensor(pre_phrase).cuda()
            encoder_state = self.transpose_state(finish_phrase_en_state)
            phrase_state = self.transpose_state(finish_phrase_ph_state)
            decoder_state = self.transpose_state(finish_phrase_de_state)
        if live_k > 0:
            for i in range(live_k):
                sample.append(hyp_sample[i])
                sample_logpro.append(beam_logprobs_sum[i])
        sample_logpro = torch.FloatTensor(sample_logpro).cuda() 
        min_sample_logpro, min_idx = torch.min(sample_logpro,0)
        min_idx = min_idx.cpu()
        min_idx = min_idx.numpy()[0]
        return sample[min_idx], min_sample_logpro
        
        
    def sample_beam(self, fc_feats, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = []
        seqLogprobs = []
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            tmp_fc_feats = fc_feats[k:k+1]
            mean_fc_feats = torch.mean(tmp_fc_feats, -2)
            
            encoder_state = self.init_hidden('fc_encoder',self.encoder_rnn_size, mean_fc_feats,1)
            phrase_state = self.init_hidden('fc_phrase',self.phrase_rnn_size, mean_fc_feats,1)
            decoder_state = self.init_hidden('fc_decoder',self.decoder_rnn_size, mean_fc_feats,1)

            done_beams = []
            tmp_att_feats = None
            seq_i, prob_i = self.beam_search(encoder_state, phrase_state, decoder_state, mean_fc_feats, tmp_fc_feats, opt=opt)
            seq.append(seq_i) # the first beam has highest cumulative score
            seqLogprobs.append(prob_i)
        # return the samples and their log likelihoods
        return seq, seqLogprobs

    def sample(self, fc_feats, att_feats, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            output, state = self.core(xt, fc_feats, att_feats, state)
            logprobs = F.log_softmax(self.logit(self.dropout(output)))

        return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


class PhraseAttendTellCore(nn.Module):
    def __init__(self, opt):
        super(PhraseAttendTellCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.encoder_rnn_size = opt.encoder_rnn_size
        self.phrase_rnn_size = opt.phrase_rnn_size
        self.phrase_rnn = getattr(nn, self.rnn_type.upper())(self.encoder_rnn_size + self.att_feat_size, 
                self.phrase_rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        if self.att_hid_size > 0:
            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.h2att = nn.Linear(self.phrase_rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1)
        else:
            self.ctx2att = nn.Linear(self.att_feat_size, 1)
            self.h2att = nn.Linear(self.phrase_rnn_size, 1)

    def forward(self, xt, fc_feats, att_feats, state):
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = att_feats.view(-1, self.att_feat_size)
        if self.att_hid_size > 0:
            att = self.ctx2att(att)                             # (batch * att_size) * att_hid_size
            att = att.view(-1, att_size, self.att_hid_size)     # batch * att_size * att_hid_size
            att_h = self.h2att(state[0][-1])                    # batch * att_hid_size
            att_h = att_h.unsqueeze(1).expand_as(att)           # batch * att_size * att_hid_size
            dot = att + att_h                                   # batch * att_size * att_hid_size
            dot = F.tanh(dot)                                   # batch * att_size * att_hid_size
            dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
            dot = self.alpha_net(dot)                           # (batch * att_size) * 1
            dot = dot.view(-1, att_size)                        # batch * att_size
        else:
            att = self.ctx2att(att)(att)                        # (batch * att_size) * 1
            att = att.view(-1, att_size)                        # batch * att_size
            att_h = self.h2att(state[0][-1])                    # batch * 1
            att_h = att_h.expand_as(att)                        # batch * att_size
            dot = att_h + att                                   # batch * att_size
        
        weight = F.softmax(dot)
        att_feats_ = att_feats.view(-1, att_size, self.att_feat_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        output, state = self.phrase_rnn(torch.cat([xt, att_res], 1).unsqueeze(0), state)
        return output.squeeze(0), state, att_res, weight


class PhraseAttendTellModel(OldModel):
    def __init__(self, opt):
        super(PhraseAttendTellModel, self).__init__(opt)
        self.core = PhraseAttendTellCore(opt)
        self.seg_index = opt.seg_ix
