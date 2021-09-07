import torch.utils.data
import torch
import os
import numpy as np
import torch.nn as nn
from model.Constant import Constants

Constants = Constants()


def collate_fn(insts):
    '''PAD the instance to the max seq length in batch'''
    equ_nodes, com_sns_nodes, captions, equ_matrixs, com_sns_matrixs, scenes = zip(*insts)
    len_equ_nodes = [len(node) for node in equ_nodes]
    len_sns_nodes = [len(node) for node in com_sns_nodes]
    # padding scenes
    max_scene_len = max(len(scene) for scene in scenes)
    batch_scenes = np.array([
        inst + [Constants.PAD] * (max_scene_len-len(inst)) for inst in scenes
    ])
    # padding equation nodes
    max_equ_node_len = max(len_equ_nodes)
    batch_equ_nodes = np.array([
        inst + [Constants.PAD] * (max_equ_node_len-len(inst)) for inst in equ_nodes
    ])
    # padding common sense nodes
    max_sns_node_len = max(len_sns_nodes)
    batch_sns_nodes = np.array([
        inst + [Constants.PAD] * (max_sns_node_len - len(inst)) for inst in com_sns_nodes
    ])

    # padding captions
    lengths_caption = [len(cap) for cap in captions]
    max_caption_len = max(lengths_caption)
    batch_captions = np.array([
        cap_inst + [Constants.PAD] * (max_caption_len-len(cap_inst)) for cap_inst in captions
    ])

    # padding equation matrix
    equ_matrixs_pad = []
    for j, node_one in enumerate(equ_matrixs):
        new_matrix = np.pad(equ_matrixs[j], ((0, max_equ_node_len-len(equ_matrixs[j])),(0, max_equ_node_len-len(equ_matrixs[j]))),'constant', constant_values=(0,0))
        equ_matrixs_pad.append(new_matrix)
    
    # padding common sense matrix
    sns_matrixs_pad = []
    for j, node_one in enumerate(com_sns_matrixs):
        new_matrix = np.pad(com_sns_matrixs[j], ((0, max_sns_node_len-len(com_sns_matrixs[j])),(0, max_sns_node_len-len(com_sns_matrixs[j]))),'constant', constant_values=(0,0))
        sns_matrixs_pad.append(new_matrix)
    return torch.LongTensor(batch_equ_nodes), torch.LongTensor(batch_sns_nodes),torch.FloatTensor(len_equ_nodes), torch.FloatTensor(len_sns_nodes),torch.FloatTensor(equ_matrixs_pad), torch.FloatTensor(sns_matrixs_pad), torch.LongTensor(batch_captions), torch.LongTensor(batch_scenes)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, src_word2idx, tgt_word2idx, node_insts=None, rel_insts=None, node_insts_1=None, rel_insts_1=None,scene_insts=None,tgt_insts=None):
        src_idx2word = {idx:word for word, idx in src_word2idx.items()}
        self._src_word2idx = src_word2idx
        self._src_idx2word = src_idx2word
        self._node_insts = node_insts # equation info
        self._node_insts_1 = node_insts_1 # common sense info

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word
        self._rel_insts = rel_insts # equation info
        self._rel_insts_1 = rel_insts_1 # common sense info
        self._tgt_insts = tgt_insts
        self._scene_insts = scene_insts

    @property
    def n_insts(self):
        '''Property for dataset size'''
        return len(self._node_insts)

    @property
    def src_vocab_size(self):
        '''property for vocab size'''
        return len(self._src_word2idx)

    @property
    def tgt_vocab_size(self):
        '''peoperty for vocab size'''
        return len(self._tgt_word2idx)

    @property
    def src_word2idx(self):
        return self._src_word2idx

    @property
    def tgt_word2idx(self):
        return self._tgt_word2idx

    @property
    def src_idx2word(self):
        '''Property for index dictionary'''
        return self._src_idx2word

    @property
    def tgt_idx2word(self):
        return self._tgt_idx2word

    def __len__(self):
        return self.n_insts
    
    def __getitem__(self, idx):
        # return one data pair (node and caption and adjmatrix)
        one_caption = self._tgt_insts[idx] + [Constants.EOS_WORD]
        one_node = self._node_insts[idx]
        one_rel = self._rel_insts[idx]
        one_scene = self._scene_insts[idx]
        other_node = self._node_insts_1[idx]
        other_rel = self._rel_insts_1[idx]

        one_num_caption = [self._src_word2idx.get(x,Constants.UNK) for x in one_caption]
        one_num_node = [self._src_word2idx.get(x,Constants.UNK) for x in one_node]
        one_num_scene = [self._src_word2idx.get(x, Constants.UNK) for x in one_scene]
        other_num_node = [self._src_word2idx.get(x,Constants.UNK) for x in other_node]
        # build adj matrix
        # this is graph for equation information
        i = 0
        one_inner_dict = {}
        for tok in one_node:
            one_inner_dict[tok] = i
            i += 1
        
        node_num = len(one_inner_dict)
        matrix = np.zeros((node_num, node_num), dtype=int)
        for m in range(0, node_num):
            # add diagonal
            matrix[m][m] = 1
        for r in one_rel:
            head, tail = r
            loc1, loc2 = one_inner_dict[head], one_inner_dict[tail]
            matrix[loc1][loc2], matrix[loc2][loc1] = 1,1
        
        # common sense graph
        ii = 0
        other_inner_dict = {}
        for tok_other in other_node:
            other_inner_dict[tok_other] = ii
            ii += 1
        
        other_node_num = len(other_inner_dict)
        other_matrix = np.zeros((other_node_num, other_node_num), dtype=int)
        for m_other in range(0, other_node_num):
            # add diagonal
            other_matrix[m_other][m_other] = 1
        for r_other in other_rel:
            head_other, tail_other = r_other
            loc1_other, loc2_other = other_inner_dict[head_other], other_inner_dict[tail_other]
            other_matrix[loc1_other][loc2_other], other_matrix[loc2_other][loc1_other] = 1,1
        return one_num_node, other_num_node, one_num_caption, matrix, other_matrix, one_num_scene
