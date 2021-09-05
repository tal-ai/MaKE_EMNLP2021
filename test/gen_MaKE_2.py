import torch
import numpy as np
from queue import Queue
import torch.nn.functional as F
import sys
sys.path.append('..')
from model.Constant import Constants
from model.dual_graph_vae_2 import Graph2seq, ScheduledOptim
import pandas as pd
from dataset_dual import MyDataset, collate_fn

class opt:
    def __init__(self):
        self.model = '../saved_model/dual_dir_fix_0.chkpt'
        self.cuda = True
        self.batch_size=1
opt = opt()
device = torch.device('cuda:2' if opt.cuda else 'cpu')
class Generator(object):
    """Load with trained model and handle the beam search"""
    def __init__(self, opt, mmi_opt=None, mmi_g=10, mmi_lambda=0.1, mmi_gamma=0.1):
        self.opt = opt
        self.device = torch.device('cuda:2' if opt.cuda else 'cpu')
        
        checkpoint = torch.load(opt.model,map_location=lambda storage, loc: storage)
        model_opt = checkpoint['settings']
        self.model_opt = model_opt
        model = Graph2seq(
            vocab_size=model_opt.vocab_size, 
            embedding_dim = model_opt.embedding_dim, 
            hidden_size = model_opt.hidden_size, 
            z_dim =model_opt.z_dim,
            output_size = model_opt.vocab_size,
            n_hop=model_opt.n_hop,
            teacher_forcing=model_opt.teacher_forcing,
            dropout=0.1
        )
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')
        model = model.to(self.device)
        self.model = model
        self.model.eval()
        self.beam_width = 20
        self.output_beam = 5
    
    def translate_one(self, input_nodes, adj_matrix, node_lens, scene):
        dec_ids, attn_weights=self.model.predict(input_nodes, adj_matrix, node_lens,scene,self.device, self.model_opt.max_token_seq_len+1)
        return dec_ids
    
    def translate_with_beam(self, input_equ_nodes, adj_equ_matrix, equ_node_lens, input_sns_nodes, adj_sns_matrix, sns_node_lens, scene):
        attns, res = beam_search_graph_gate_vae(self.model, input_equ_nodes, adj_equ_matrix, equ_node_lens, input_sns_nodes, adj_sns_matrix, sns_node_lens, scene,self.device, self.model_opt.max_token_seq_len+1, self.beam_width, self.output_beam)
        return res

gen = Generator(opt)

data = torch.load('../processed_data/dual_graph_rev.pt')
test_loader = torch.utils.data.DataLoader(
        MyDataset(
            src_word2idx = data['dict']['tgt'],
            tgt_word2idx = data['dict']['tgt'],
            node_insts= data['test']['node_1'],# equation info
            rel_insts = data['test']['edge_1'],
            node_insts_1 = data['test']['node_2'],# common sense info
            rel_insts_1 = data['test']['edge_2'],
            scene_insts = data['test']['scene'],
            tgt_insts = data['test']['ref']
        ),
        num_workers = 4,
        batch_size=opt.batch_size,
        collate_fn=collate_fn,
        shuffle=False)
all_nodes_equ, all_node_lens_equ, all_adj_matrix_equ, all_nodes_sns, al_node_lens_sns, all_adj_matrix_sns,all_scene = [],[],[],[],[],[],[]
for batch in test_loader:
    equ_nodes, sns_nodes, equ_node_lens, sns_node_lens, equ_adj_matrixs, sns_adj_matrixs, tgt_seq, scene = map(lambda x: x.to(device), batch)
    all_nodes_equ.append(equ_nodes)
    all_node_lens_equ.append(equ_node_lens)
    all_adj_matrix_equ.append(equ_adj_matrixs)
    all_scene.append(scene)
    all_nodes_sns.append(sns_nodes)
    al_node_lens_sns.append(sns_node_lens)
    all_adj_matrix_sns.append(sns_adj_matrixs)

all_lex = data['test']['lexs']
idx2word_type = {value:key for key, value in data['dict']['tgt'].items()}

Constants = Constants()
lex_tar = [Constants.eq2_right_num1_WORD, Constants.eq1_right_num1_WORD, Constants.x_entity_WORD, Constants.y_entity_WORD, Constants.head_info_entity_WORD, Constants.jiao_info_entity_WORD,
    Constants.head_info_unit_WORD, Constants.jiao_info_unit_WORD,
    Constants.eq1_y_index_WORD,Constants.eq2_y_index_WORD,
    Constants.eq1_x_index_WORD, Constants.eq2_x_index_WORD, Constants.eq1_right_num2_WORD, Constants.eq2_right_num2_WORD]
lex_tar_id =  [Constants.eq2_right_num1, Constants.eq1_right_num1, Constants.x_entity, Constants.y_entity, 
    Constants.head_info_entity, Constants.jiao_info_entity,
    Constants.head_info_unit, Constants.jiao_info_unit,
    Constants.eq1_y_index,Constants.eq2_y_index,
    Constants.eq1_x_index, Constants.eq2_x_index, Constants.eq1_right_num2, Constants.eq2_right_num2]


lex_tar_map = dict((key, idx) for idx, key in enumerate(lex_tar))
lex_tar_id_map = dict((key, idx) for idx, key in enumerate(lex_tar_id))
def ids2token_2(ids, lex, idx2word):
    res = []
    #print('before delexicalization is:', ' '.join([idx2word[x] for x in ids]))
    for id_ in ids:
        if id_ in lex_tar_id:
            if lex[lex_tar_id_map[id_]] is not None:
                res.append(lex[lex_tar_id_map[id_]])
        else:
            res.append(idx2word[id_])
    return ''.join(res)

def end2end_beam(input_equ_nodes, adj_equ_matrix, equ_node_lens, input_sns_nodes, adj_sns_matrix, sns_node_lens, scene, lex, idx2word, generator):
    output = generator.translate_with_beam(input_equ_nodes, adj_equ_matrix, equ_node_lens, input_sns_nodes, adj_sns_matrix, sns_node_lens, scene)
    all_text = []
    for one_res in output:
        one_text = ids2token_2(one_res, lex, idx2word)
        all_text.append(one_text)
    return all_text

class GateNode(object):
    def __init__(self, hidden, previous_node, decoder_input, attn, log_prob, length):
        self.hidden = hidden
        self.previous_node = previous_node
        self.decoder_input = decoder_input
        self.attn = attn
        self.log_prob = log_prob
        self.length = length

def beam_search_graph_gate_vae(model, input_equ_nodes, adj_equ_matrix, equ_node_lens, input_sns_nodes, adj_sns_matrix, sns_node_lens, scene, device, max_tgt_len, beam_width=3, output_num=3):
    model.eval()
    with torch.no_grad():
        # encode equation input
        equ_node_resp = model.embedding(input_equ_nodes)
        equ_encoder_outputs, equ_encoder_hidden = model.encoder_equ(equ_node_resp, adj_equ_matrix, equ_node_lens)# bs*seq*h, bs*h
        # encode common sense input
        sns_node_resp = model.embedding(input_sns_nodes)
        sns_encoder_outputs, sns_encoder_hidden = model.encoder_sns(sns_node_resp, adj_sns_matrix, sns_node_lens)# bs*seq*h, bs*h
        
        # sample z from normal distribution
        batch_size = equ_node_resp.shape[0]
        # get condition embedding
        cond_embedding = torch.cat([equ_encoder_hidden, sns_encoder_hidden],1)
        prior_embedding = model.prior_fc1(cond_embedding)
        prior_mu, prior_logvar = model.q_mu_prior(prior_embedding), model.q_logvar_prior(prior_embedding)

        # smaple latent z
        latent_sample = model.sample_z(prior_mu, prior_logvar)
        
        # decode
        decoder_input = torch.LongTensor([Constants.BOS]).to(device)
        node = GateNode(torch.cat([cond_embedding,latent_sample], dim=1), None, decoder_input, None, 0,1)
        q = Queue()
        q.put(node)
        end_nodes = []
        while not q.empty():
            candidates = []
            for _ in range(q.qsize()):
                node = q.get()
                decoder_input = node.decoder_input
                prev_y = model.embedding(decoder_input)
                hidden = node.hidden

                if decoder_input.item() == Constants.EOS or node.length >= max_tgt_len:
                    end_nodes.append(node)
                    continue
#                 print("prev_y shape",prev_y.shape)
#                 print('hidden shape', hidden.shape)
#                 print("d_dec shape", d_dec.shape)
                
                log_prob, hidden, attn = model.decoder(prev_y, hidden, equ_encoder_outputs, sns_encoder_outputs, latent_sample, input_equ_node_mask=None, input_sns_node_mask=None)
                log_prob = F.log_softmax(log_prob.squeeze(), dim=-1)
                log_prob, indices = log_prob.topk(beam_width)
                
                for k in range(beam_width):
                    index = indices[k].unsqueeze(0)
                    log_p = log_prob[k].item()
                    child = GateNode(hidden.squeeze(1), node, index, attn, node.log_prob + log_p,node.length+1)
                    candidates.append((node.log_prob + log_p, child))
            candidates = sorted(candidates, key = lambda x: x[0], reverse = True)
            length = min(len(candidates), beam_width)
            for i in range(length):
                q.put(candidates[i][1])
        candidates = []
        for node in end_nodes:
            value = node.log_prob
            candidates.append((value, node))
        candidates = sorted(candidates, key = lambda x: x[0], reverse=True)
        node = [x[1] for x in candidates[:output_num]]
        res = []
        attns = []
        for one_node in node:
            one_res = []
            one_attns = []
            while one_node.previous_node != None:
                one_res.append(one_node.decoder_input.item())
                one_attns.append(one_node.attn.squeeze(0).cpu().numpy().tolist())
                one_node = one_node.previous_node
            res.append(one_res[::-1])
            attns.append(attns[::-1])
    return attns, res

all_graph_gen_beam = []
for i in range(len(all_nodes_equ)):
    all_graph_gen_beam.append(end2end_beam(all_nodes_equ[i], all_adj_matrix_equ[i], all_node_lens_equ[i],
                                        all_nodes_sns[i], all_adj_matrix_sns[i], al_node_lens_sns[i],
                                        all_scene[i], all_lex[i], idx2word_type, gen))
torch.save(all_graph_gen_beam,'MaKE.pt')

# def end2end_generator(input_node, node_len, graph, scene,lex, idx2word, generator):
#     output = generator.translate_one(input_node, graph, node_len, scene)
#     return ids2token_2(output, lex, idx2word)
# graph_gen = []
# for i in range(len(all_nodes)):
#     graph_gen.append(end2end_generator(all_nodes[i], all_node_lens[i], all_adj_matrix[i], all_scene[i], all_lex[i], idx2word_type, gen))
# torch.save(graph_gen,'gggvae_gen.pt')
