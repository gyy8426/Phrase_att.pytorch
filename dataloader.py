from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
import torch.utils.data as data
import misc.utils as utils
import multiprocessing
import math

class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split,
                                                    self, split == 'train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.maxlen_sen
        
    def _filter_googlenet(self, vidID):
        feat_file = os.path.join(self.feat_path, vidID + '.npy')
        feat = np.load(feat_file)
        feat = self.get_sub_frames(feat)
        return feat
    
    def get_video_features(self, vidID):
        if self.video_feature == 'googlenet' or self.video_feature == 'resnet':
            y = self._filter_googlenet(vidID)
        else:
            raise NotImplementedError()
        return y

    def pad_frames(self, frames, limit, jpegs):
        # pad frames with 0, compatible with both conv and fully connected layers
        last_frame = frames[-1]
        if jpegs:
            frames_padded = frames + [last_frame]*(limit-len(frames))
        else:
            padding = np.asarray([last_frame * 0.]*(limit-len(frames)))
            frames_padded = np.concatenate([frames, padding], axis=0)
        return frames_padded
    
    def extract_frames_equally_spaced(self, frames, how_many):
        # chunk frames into 'how_many' segments and use the first frame
        # from each segment
        n_frames = len(frames)
        splits = np.array_split(range(n_frames), self.num_frames)
        idx_taken = [s[0] for s in splits]
        sub_frames = frames[idx_taken]
        return sub_frames
    
    def add_end_of_video_frame(self, frames):
        if len(frames.shape) == 4:
            # feat from conv layer
            _,a,b,c = frames.shape
            eos = np.zeros((1,a,b,c),dtype='float32') - 1.
        elif len(frames.shape) == 2:
            # feat from full connected layer
            _,b = frames.shape
            eos = np.zeros((1,b),dtype='float32') - 1.
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        frames = np.concatenate([frames, eos], axis=0)
        return frames
    
    def get_sub_frames(self, frames, jpegs=False):
        # from all frames, take num_frames of them, then add end of video frame
        # jpegs: to be compatible with visualizations
        if len(frames) < self.num_frames:
            #frames_ = self.add_end_of_video_frame(frames)
            frames_ = self.pad_frames(frames, self.num_frames, jpegs)
        else:

            frames_ = self.extract_frames_equally_spaced(frames, self.num_frames)
            #frames_ = self.add_end_of_video_frame(frames_)
        if jpegs:
            frames_ = numpy.asarray(frames_)
        return frames_
        
    def get_data(self, ix):
        if '_' in ix: 
            vidid, capid = ix.split('_')
        else :
            vidid = ix
        feat = self.get_video_features(vidid)
        if self.use_att:
            return (np.array(feat).astype('float32'), ix)
        else:
            return (np.array(feat).astype('float32'), ix)

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.use_att = getattr(opt, 'use_att', True)
        self.dataset = opt.dataset
        self.pre_dataset_path = opt.dataset_path + str.upper(opt.dataset) + opt.pre_dataset_path
        self.feat_path = opt.dataset_path + str.upper(opt.dataset) + opt.feat_path
        # separate out indexes for each of the provided splits
        self.ctx_dim = opt.fc_feat_size
        self.video_feature = opt.feat_type
        self.num_frames = opt.num_frames
        self.maxlen_sen = opt.maxlen_sen
        self.salicy_hard = opt.salicy_hard
        self.max_seg_len = opt.max_seg_len
        self.inc_seg = opt.inc_seg
        self.rng = np.random.RandomState(1234)
        self.split_ix = {'train': [], 'val': [], 'test': []}
        self.train = utils.load_pkl(self.pre_dataset_path + 'train.pkl')
        self.valid = utils.load_pkl(self.pre_dataset_path + 'valid.pkl')
        self.test = utils.load_pkl(self.pre_dataset_path + 'test.pkl')
        self.CAP = utils.load_pkl(self.pre_dataset_path + 'CAP.pkl')
        self.CAP_seg = utils.load_pkl(self.pre_dataset_path + 'CAP_seg.pkl')
        self.split_ix['train'] = self.train
        self.split_ix['val'] = self.valid
        self.split_ix['test'] = self.test
        if opt.dataset == 'msvd':
            self.train_ids = ['vid%s'%i for i in range(1,1201)]
            self.valid_ids = ['vid%s'%i for i in range(1201,1301)]
            self.test_ids = ['vid%s'%i for i in range(1301,1971)]
            self.word_ix = utils.load_pkl(self.pre_dataset_path + 'worddict.pkl')
            self.salicy =  utils.load_pkl(self.pre_dataset_path + 'salicy_map_1.pkl')
        elif opt.dataset == 'msr-vtt':
            self.train_ids = ['video%s'%i for i in range(0,6513)]
            self.valid_ids = ['video%s'%i for i in range(6513,7910)]
            self.test_ids = ['video%s'%i for i in range(7910,10000)]
            self.word_ix = utils.load_pkl(self.pre_dataset_path + 'worddict_small.pkl') 
            self.salicy =  utils.load_pkl(self.pre_dataset_path + 'salicy_map_100.pkl')
            
        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))
        
        # load json file which contains additional information about dataset
        self.ix_to_word = dict()
        # word_ix start with index 2
        for kk, vv in self.word_ix.iteritems():
            self.ix_to_word[vv] = kk
        self.ix_to_word[1] = '<bos>'
        self.ix_to_word[0] = '<eos>'
        self.word_ix['<bos>'] = 1
        self.word_ix['<eos>'] = 0
        len_dict = len(self.ix_to_word)
        self.word_ix['<UNK>'] = len_dict
        self.ix_to_word[len_dict] = '<UNK>'
        self.seg_ix = []
        len_dict = len(self.ix_to_word)
        self.word_ix['<SEG>'] = len_dict
        self.ix_to_word[len_dict] = '<SEG>'
        self.seg_ix.append(len_dict)
        len_dict = len(self.ix_to_word)
        if self.opt.inc_seg == True:
            for i in range(30):
                self.word_ix['<SEG>'+str(i)] = len_dict+i
                self.ix_to_word[len_dict+i] = '<SEG>'+str(i)
                self.seg_ix.append(len_dict+i)
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split,
                                                        self,
                                                        split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)
        
    def get_words(self, vidID, capID):
        caps = self.CAP_seg[vidID]
        rval = None
        for cap in caps:
            if cap['cap_id'] == capID:
                rval = cap['seg_phrase']
                if rval != None:
                    rval = rval.split(' ')
                    rval.append('<SEG>')
                    if self.inc_seg == True:
                        count = 0
                        for i in range(len(rval)):
                            if rval[i] == '<SEG>':
                                rval[i] = rval[i] + str(count)
                                if count < 30:
                                    count = count + 1
                break
        if rval != None:
            rval.insert(0,'<SEG>')
            rval.insert(0,'<bos>')
            rval.append('<eos>')
        return rval
        
    def get_ctx_mask(self, ctx):
        if ctx.ndim == 3:
            rval = (ctx[:,:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 2:
            rval = (ctx[:,:self.ctx_dim].sum(axis=-1) != 0).astype('int32').astype('float32')
        elif ctx.ndim == 5 or ctx.ndim == 4:
            assert self.video_feature == 'oxfordnet_conv3_512'
            # in case of oxfordnet features
            # (m, 26, 512, 14, 14)
            rval = (ctx.sum(-1).sum(-1).sum(-1) != 0).astype('int32').astype('float32')
        else:
            import pdb; pdb.set_trace()
            raise NotImplementedError()
        
        return rval
        
    def get_val_batch(self, split):
        if split == 'val':
            ids = self.valid_ids
        elif split == 'test':
            ids = self.test_ids
        feat_list = []
        IDs = []
        for i in range(len(ids)):
            feat = self.get_video_features(ids[i])
            IDs.append(ids[i])
            feat_list.append(feat)
        return np.array(feat_list).astype('float32'), ids
        
    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img
        feat_batch = []
        wrapped = False
        infos = []
        gts = []
        IDs = []
            
        for i in range(batch_size):
            # fetch image
            tmp_feat,\
                ix, tmp_wrapped = self._prefetch_process[split].get()
            feat_batch.append(tmp_feat)
            vidID, capID = ix.split('_')
            # fetch the sequence labels
            none_id = []
            IDs.append(ix)
            words = self.get_words(vidID, capID)
            if words == None:
                none_id.append(ix)
                continue
            gts.append([self.word_ix[w]
                if w in self.word_ix else self.word_ix['<UNK>'] for w in words])
            if tmp_wrapped:
                wrapped = True
        lengths = [len(s) for s in gts]
        if self.maxlen_sen != None:
            new_seqs = []
            new_feat_list = []
            new_lengths = []
            new_caps = []
            for l, s, y, c in zip(lengths, gts, feat_batch, IDs):
                # sequences that have length >= maxlen will be thrown away 
                if split == 'train':
                    if c in none_id or c not in self.salicy.keys():
                        continue
                else :
                    if c in none_id:
                        continue
                if l < (self.maxlen_sen):
                    new_seqs.append(s)
                    new_feat_list.append(y)
                    new_lengths.append(l)
                    new_caps.append(c)
            lengths = new_lengths
            feat_list = new_feat_list
            seqs = new_seqs

        salicy_list = [] 
        if split == 'train':
            for ids in new_caps:
                salicy_i = self.salicy[ids][:,:,0]  #num_frames, len_sen
                salicy_list.append(salicy_i/(salicy_i.sum(0)+1e-9))
        seqs_seg = []
        len_seg = []
        ind_seg = []
        num_seg = []
        max_len_seg = 0
        max_num_seg = 0
        for i in seqs:
            seq_seg_i = [[]]
            count_num_seg = 0
            count_len_seg = 0
            count_ = 0
            len_seg.append([])
            ind_seg.append([])
            for j in i :
                if j not in self.seg_ix and j != self.word_ix['<eos>']:
                    seq_seg_i[-1].append(j)
                    count_len_seg = count_len_seg + 1
                    count_ = count_ + 1
                elif j in self.seg_ix:
                    len_seg[-1].append(count_len_seg+1)
                    count_ = count_ + 1
                    ind_seg[-1].append(count_)
                    if count_len_seg+1 > max_len_seg:
                        max_len_seg = count_len_seg+1
                    count_len_seg = 0
                    count_num_seg = count_num_seg + 1
                    seq_seg_i[-1].append(j)
                    seq_seg_i.append([])
            len_seg[-1].append(count_len_seg)
            #ind_seg[-1].append(count_)
            if count_len_seg+1 > max_len_seg:
                max_len_seg = count_len_seg+1
            if count_num_seg+1 > max_num_seg:
                max_num_seg = count_num_seg+1
            num_seg.append(count_num_seg)
            seqs_seg.append(seq_seg_i)
        # seqs_seg shape: batch_size, num_seg, len_seg
        #
        y = np.asarray(feat_list)
        y_mask = self.get_ctx_mask(y)
        n_samples = len(seqs)
        maxlen = np.max(lengths)+self.max_seg_len+1
        #maxlen = int(math.ceil(maxlen/(self.max_seg_len*1.0))*self.max_seg_len)+self.max_seg_len
        x = np.zeros((maxlen, n_samples)).astype('int64')
        x_mask = np.zeros((maxlen, n_samples)).astype('float32')
        #print("max_len_seg: ",max_len_seg)
        #print("max_num_seg: ",max_num_seg)
        x_seg = np.zeros((max_len_seg, max_num_seg, n_samples)).astype('int64')
        x_seg_mask = np.zeros((max_len_seg, max_num_seg, n_samples)).astype('float32')
        x_phrase_mask_0 = np.zeros((max_num_seg+1, maxlen, n_samples)).astype('float32')
        x_phrase_mask_1 = np.zeros((max_num_seg+1, maxlen, n_samples)).astype('float32')
        salicy_arr = np.zeros((maxlen, n_samples, y.shape[-2])).astype('float32')
        salicy_seg = np.zeros((max_num_seg+1, n_samples, y.shape[-2])).astype('float32')
        seg_mask = np.zeros((max_num_seg+1, n_samples)).astype('float32')
        for idx, s in enumerate(seqs):
            x[:lengths[idx],idx] = s
            x_mask[:lengths[idx]+self.max_seg_len,idx] = 1.0
            if split == 'train':
                if self.salicy_hard == False:
                    if lengths[idx]+1 < salicy_list[idx].shape[-1]:
                        salicy_arr[:lengths[idx]+1,idx,:] = (salicy_list[idx][:,:lengths[idx]+1]).transpose(1,0)
                    else : 
                        salicy_arr[:salicy_list[idx].shape[-1],idx,:] = (salicy_list[idx]).transpose(1,0)
                elif self.salicy_hard == True:
                    for word_idx in range(salicy_list[idx].shape[-1]):
                        #print(len(salicy_list[idx][:,word_idx]))
                        arg_max_rn = self.rng.choice(len(salicy_list[idx][:,word_idx]),1,p=(salicy_list[idx][:,word_idx]/(salicy_list[idx][:,word_idx].sum())))
                        salicy_arr[word_idx,idx,arg_max_rn[0]] = 1.0
            finial_end = 0 
            start_sa = 0 
            end_sa = 1
            for idx_seg, s_seg in enumerate(seqs_seg[idx]):
                if s_seg == []:
                    continue
                x_seg[:len_seg[idx][idx_seg],idx_seg,idx] = s_seg
                x_seg_mask[:len_seg[idx][idx_seg],idx_seg,idx] = 1.0
                if idx_seg+1 < len(ind_seg[idx]):
                    end_sa = start_sa + ind_seg[idx][idx_seg+1] - ind_seg[idx][idx_seg] - 1  #   map to word index to salicy index  
                    finial_end = end_sa
                    salicy_seg[idx_seg,idx,:] = (salicy_arr[start_sa:end_sa,idx,:].sum(0)) / ((salicy_arr[start_sa:end_sa,idx,:]).sum()+1e-9)
                    start_sa = end_sa
                if idx_seg == 0:
                    x_phrase_mask_0[idx_seg,:ind_seg[idx][idx_seg],idx] = 1.0
                else: 
                    x_phrase_mask_0[idx_seg,ind_seg[idx][idx_seg-1]:ind_seg[idx][idx_seg],idx] = 1.0
                if idx_seg == len(seqs_seg[idx])-1:
                    x_seg_mask[:len_seg[idx][idx_seg],idx_seg,idx] = 1.0
                else :
                    x_seg_mask[:len_seg[idx][idx_seg],idx_seg,idx] = 1.0
            x_phrase_mask_1[:-1,:-2,idx] = x_phrase_mask_0[1:,2:,idx]
            x_phrase_mask_1[len(seqs_seg[idx])-2, ind_seg[idx][-1]-2:ind_seg[idx][-1]-2+self.max_seg_len, idx] = 1.0
            salicy_seg[len(seqs_seg[idx])-2,idx,:] = salicy_arr[finial_end,idx,:]
            seg_mask[:num_seg[idx]+1,idx] =1.0
        data = {}
        #print('ind_seg: ',ind_seg[0])
        data['fc_feats'] = y
        data['feat_masks'] = y_mask
        data['labels'] = x
        #print('labels: ',x[:,0])
        data['label_masks'] = x_mask
        #print('label_masks: ',x_mask[:,0])
        data['x_phrase_mask_0'] = x_phrase_mask_0
        #print('x_phrase_mask_0: \n',x_phrase_mask_0[:,:,0])
        data['x_phrase_mask_1'] = x_phrase_mask_1
        #print('x_phrase_mask_1: \n',x_phrase_mask_1[:,:,0])
        data['salicy_arr'] = salicy_arr
        #print('salicy_arr: \n',salicy_arr[:,0,:])
        data['salicy_seg'] = salicy_seg
        #print('salicy_seg: \n',salicy_seg[:,0,:])
        data['seg_mask'] = seg_mask
        #print('seg_mask: ',seg_mask[:,0])
        data['gts'] = gts
        #print('gts: ',gts[0])
        data['bounds'] = {'it_pos_now': self.iterators[split],
                          'it_max': len(self.split_ix[split]),
                          'wrapped': wrapped}
        return data

    # It's not coherent to make DataLoader a subclass of Dataset,
    # but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according
    # the index. However, it's minimum change to switch to pytorch data loading
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        return self.get_data(ix)

    def __len__(self):
        return len(self.info['images'])


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name,
        caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases:
        1. not hasattr(self, 'split_loader'): Resume from previous training.
        Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in
         the get_minibatch_inds already.
        """
        # batch_size is 0, the merge is done in DataLoader class
        sampler = self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]
        self.split_loader = iter(
            data.DataLoader(dataset=self.dataloader,
                            batch_size=1,
                            sampler=sampler,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=multiprocessing.cpu_count(),
                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()
        assert tmp[1] == ix, "ix not equal"

        return tmp + [wrapped]
