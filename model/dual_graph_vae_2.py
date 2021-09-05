import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class Propogator(nn.Module):
    def __init__(self, node_dim):
        super(Propogator, self).__init__()
        self.node_dim = node_dim
        self.reset_gate = nn.Sequential(
            nn.Linear(node_dim*2, node_dim),
            nn.Sigmoid(),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(node_dim*2, node_dim),
            nn.Sigmoid(),
        )
        self.transform = nn.Sequential(
            nn.Linear(node_dim*2, node_dim),
            nn.Tanh()
        )
    
    def forward(self, node_representation, adjmatrixs): # ICLR 2016 fomulas implementation
        a = torch.bmm(adjmatrixs, node_representation)
        joined_input1 = torch.cat((a, node_representation),2) # bs * node_len * hidden
        z = self.update_gate(joined_input1)
        r = self.reset_gate(joined_input1)
        joined_input2 = torch.cat((a, r*node_representation),2)
        h_hat = self.transform(joined_input2)
        output = (1-z) * node_representation + z*h_hat
        return output

class EncoderGGNN(nn.Module):
    def __init__(self, vocab_size, node_dim, hidden_dim, n_hop = 5):
        super(EncoderGGNN, self).__init__()
        self.node_dim = node_dim
        self.n_hop = n_hop
        self.propogator = Propogator(self.node_dim)
        self.out1 = nn.Sequential(
            nn.Linear(self.node_dim * 2, hidden_dim)
        )
        # self.out2 = nn.Sequential(
        #     nn.Linear(self.node_dim * 2, self.node_dim),
        #     nn.Sigmoid()
        # )
        self._initialization()
    
    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
    
    def forward(self, nodes_rep, adjmatrixs, lengths):
        lengths = lengths.view(-1,1)
        # embeddings = self.embed(nodes)
        node_representation = nodes_rep
        init_node_representation = nodes_rep
        for _ in range(self.n_hop):
            node_representation = self.propogator(node_representation, adjmatrixs)
        gate_inputs = torch.cat((node_representation, init_node_representation),2)
        gate_outputs = self.out1(gate_inputs)
        # aggregate all node information as global vector
        features = torch.sum(gate_outputs,1)
        features = features / lengths
        return gate_outputs, features

class Constants:
    def __init__(self):
        self.BOS_WORD = '<s>'
        self.EOS_WORD = '</s>'
        self.PAD_WORD = '<blank>'
        self.UNK_WORD = '<unk>'
        self.head_info_num_WORD = 'head_info_num'
        self.rabbit_entity_WORD = 'rabbit_entity'
        self.rabbit_tou_num_WORD = 'rabbit_tou_num'
        self.rabbit_jiao_num_WORD = 'rabbit_jiao_num'
        self.jiao_info_num_WORD = 'jiao_info_num'
        self.ji_entity_WORD = 'ji_entity'
        self.ji_tou_num_WORD = 'ji_tou_num'
        self.ji_jiao_num_WORD = 'ji_jiao_num'
        
        self.PAD = 0
        self.UNK = 1
        self.BOS = 2
        self.EOS = 3
        self.head_info_num = 4
        self.rabbit_entity = 5
        self.rabbit_tou_num = 6
        self.rabbit_jiao_num = 7
        self.jiao_info_num = 8
        self.ji_entity = 9
        self.ji_tou_num = 10
        self.ji_jiao_num = 11
Constants = Constants()

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduleing'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model,-0.5)

    def step_and_update_lr(self):
        "step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "zero out the gradient by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps,-0.5),
            np.power(self.n_warmup_steps,-1.5)*self.n_current_steps
        ])
    
    def _update_learning_rate(self):
        '''learning rate scheduleing per step'''
        self.n_current_steps+=1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def masked_log_softmax(vector, mask, dim):
	if mask is not None:
		mask = mask.float()
		while mask.dim() < vector.dim():
			mask = mask.unsqueeze(1)
		vector = vector + (mask + 1e-45).log()
	return torch.nn.functional.log_softmax(vector, dim=dim)

class MyGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size,d_size, batch_first=True):
        super(MyGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.reset_gate = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Sigmoid(),
        )
        self.update_gate = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Sigmoid(),
        )
        self.transform = nn.Sequential(
            nn.Linear(hidden_size+input_size, hidden_size),
            nn.Tanh(),
        )
        self.w2h_r = nn.Linear(input_size, d_size)
        self.h2h_r = nn.Linear(hidden_size, d_size)
        self.dc = nn.Linear(d_size, hidden_size, bias=False)
        self.output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
        )
    
    def forward(self, input_t, last_hidden, last_dt,alpha=0.5):
        """
        Do feedforward for one step
        input_t: (batch_size, 1, input_size)
        last_hidden: (bs, hidden_size)
        last_dt: (bs, d_size)
        """
        input_t = input_t.squeeze(1)
        joined_input1 = torch.cat((last_hidden, input_t),-1)
        z = self.update_gate(joined_input1)
        r = self.reset_gate(joined_input1)
        joined_input2 = torch.cat([r*last_hidden, input_t],-1)
        h_hat = self.transform(joined_input2)
        gate_r = torch.sigmoid(self.w2h_r(input_t) + alpha * self.h2h_r(last_hidden))
        dt = gate_r * last_dt # bs * d_size
        hidden = (1-z) * last_hidden + z*h_hat + self.dc(dt)
        output = self.output(hidden)
        return output, hidden, dt

class Attention(nn.Module):
    def __init__(self, hidden_size, z_dim, emb_size=0):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(2*hidden_size+z_dim, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, decoder_state, encoder_outputs,input_node_mask):
        # print('decoder state dimension,',decoder_state.shape)
        # print('encoder_outputs dimension,', encoder_outputs.shape)
        # (bs, seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)
        # (bs, 1(unsqueeze), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)
        # # (bs, max_doc_len, 1) => (bs, max_doc_len)
        u_i = self.vt(torch.tanh(encoder_transform+decoder_transform)).squeeze(-1)
        # softmax with only valid inputs, excluding zero padding parts
        # # log-softmax for a better numerical stability
        if input_node_mask is not None:
            u_i = u_i.masked_fill(input_node_mask, -np.inf)
        else:
            pass
        # log_score = F.log_softmax(u_i, dim=-1)
        log_score = F.softmax(u_i, dim=-1)
        # log_score = masked_log_softmax(u_i, input_node_mask, dim=-1)
        return log_score

class DecoderVAE(nn.Module):
    def __init__(self, embedding_dim, hidden_size, z_dim, output_size, dropout=0.1):
        super(DecoderVAE, self).__init__()
        self.z_dim = z_dim
        # self.rnn = nn.GRU(input_size=hidden_size+z_dim,hidden_size=hidden_size+z_dim, batch_first=True)
        # self.my_rnn_gate = MyGRUCell(input_size = hidden_size+z_dim, hidden_size=hidden_size+z_dim+embedding_dim, d_size=embedding_dim)
        self.my_rnn_gate = nn.GRU(input_size=hidden_size,hidden_size=2*hidden_size+z_dim, batch_first=True)
        self.hidden_dim = hidden_size
        self.attn_equ = Attention(hidden_size,z_dim, embedding_dim)
        self.attn_sns = Attention(hidden_size, z_dim, embedding_dim)
        self.plan_w = nn.Linear(2*hidden_size+z_dim, 2)
        # self.plan_w = ScaledDotProductAttention(d_q=2*hidden_size+z_dim, d_k=hidden_size, d_attn=hidden_size,temperature=1)
        self.linear_map_context = nn.Linear(embedding_dim+hidden_size, hidden_size)
        self.linear_map = nn.Linear(hidden_size*2+z_dim, output_size)
    
    def forward(self, prev_y_batch, prev_h_batch, equ_encoder_outputs, sns_encoder_outputs, z_sample,input_equ_node_mask, input_sns_node_mask):
        """
        A foward path step to a Decoder
        The step operates on one step-slice of the target sequence
        :param prev_y_batch: embedded previous prediction bs*embedding_dim
        :param prev_h_batch: current decoder state: bs * hidden_size(dec_dim)
        :param z_sample: vae sample z: bs * z_dim
        :param *_encoder_outputs" bs * n * hidden_size, bs*m*hidden_size
        equ_global, sns_global leave as future work
        """
        # calcualte attention from current RNN state and equation encoder outputs
        attn_weights_equ = self.attn_equ(prev_h_batch, equ_encoder_outputs, input_equ_node_mask)
        # Apply attention weights to encoder outputs to get weighted average
        # bs*1*seq_len x bs*seq_len*hidden_size -> bs*1*hidden_size
        context_equation = torch.bmm(attn_weights_equ.unsqueeze(1), equ_encoder_outputs)
        
        # calculate attention fom current RNN state and common sense outputs
        attn_weights_sns = self.attn_sns(prev_h_batch, sns_encoder_outputs, input_sns_node_mask)
        # bs*1*seq_len x bs*seq_len*hidden_size -> bs*1*hidden_size
        context_sns = torch.bmm(attn_weights_sns.unsqueeze(1), sns_encoder_outputs)
        
        # calculate plan attention, extract how many info from equ_encoder_outputs, how many info from sns_encoder_outputs
        # plan attention is conducted by softmax(ht-1W + b)
        plan_attn_weight = F.softmax(self.plan_w(prev_h_batch), dim=-1) # bs * 2，这里是不是改一下好一点
        # print('plan attention shape is ',plan_attn_weight.shape)
        combine_context = torch.cat((context_equation, context_sns), 1)
        # print('combine context shape is, ', combine_context.shape)
        context = torch.bmm(plan_attn_weight.unsqueeze(1), combine_context) # bs*1*2 x bs*2*hidden_size -> bs*1*hidden_size

        # context, plan_attn_weight = self.plan_w(prev_h_batch.unsqueeze(1),combine_context,combine_context)
        # combine embedded input word and attented context, run through RNN
        y_ctx = torch.cat((prev_y_batch, context.squeeze(1)), 1) # bs*(hidden_size+embedding_dim)
        rnn_input = self.linear_map_context(y_ctx) # bs * hidden_size
        
        dec_output, dec_hidden = self.my_rnn_gate(rnn_input.unsqueeze(1), prev_h_batch.unsqueeze(0))
        dec_output = self.linear_map(dec_output)
        return dec_output, dec_hidden, plan_attn_weight

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_q, d_k, d_attn,temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.w_qs = nn.Linear(d_q, d_attn)
        nn.init.xavier_normal_(self.w_qs.weight)
        self.w_ks = nn.Linear(d_k, d_attn)
        nn.init.xavier_normal_(self.w_ks.weight)
        
    def forward(self,q,k,v,mask=None):
        q,k = self.w_qs(q), self.w_ks(k)
        attn = torch.bmm(q,k.transpose(1,2))
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        safe_attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        # attn[attn!=attn] = 0.0
        safe_attn = self.dropout(safe_attn)
        # print(attn)
        output = torch.bmm(safe_attn,v)
        return output, safe_attn

class WordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,dropout=0.1, pretrained=None):
        super(WordRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embedding_dim
        self.hidden_size = hidden_dim
        self.dropout = nn.Dropout(p = dropout)
        if pretrained is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim).from_pretrained(pretrained)
        # self.word_encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # no bidirectional trytry
        self.word_encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        # self.Ws1 = nn.Parameter(torch.Tensor(1, 2*hidden_dim, 2*hidden_dim))
        self.Ws1 = nn.Parameter(torch.Tensor(1, hidden_dim, hidden_dim))
        self.Ws1.data = self.Ws1.data.uniform_(-0.1,0.1)
        # self.Ws2 = nn.Parameter(torch.Tensor(1, 1, 2*hidden_dim))
        self.Ws2 = nn.Parameter(torch.Tensor(1, 1, hidden_dim))
        self.Ws2.data = self.Ws2.data.uniform_(-0.1,0.1)
    
    def forward(self, one_doc):
        '''
        one_doc: [[1,2,3,0],[2,3,4,5],[3,0,0,0]], doc_lens:[3,4,1]
        '''
        seq_len = len(one_doc[0])
        max_sent_len = len(one_doc)
        tmp_mask = torch.eq(one_doc,0)
        doc_embedding = self.embedding(one_doc)
        doc_embedding = self.dropout(doc_embedding)
        rnn_out,_ = self.word_encoder(doc_embedding)
        #rnn_out = run_rnn(self.word_encoder,doc_embedding, doc_lens)
        rnn_out = self.dropout(rnn_out)
        final_T = torch.transpose(rnn_out,2,1).contiguous()
        A = torch.tanh(torch.bmm(self.Ws1.repeat(1*max_sent_len,1,1),final_T))
        A = torch.bmm(self.Ws2.repeat(1*max_sent_len,1,1),A)
        A = A.squeeze(1).masked_fill(tmp_mask, -1e12)
        A = F.softmax(A,dim=1).view(1*max_sent_len,-1, seq_len)
        final = torch.bmm(A, rnn_out).squeeze(1)
        return final


class Graph2seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, z_dim, output_size,n_hop=5,teacher_forcing=0.5,dropout=0.1):
        super(Graph2seq, self).__init__()
        # the First encoder to encode equation information
        self.encoder_equ = EncoderGGNN(vocab_size, embedding_dim, hidden_size,n_hop)
        # the second encoder to encode common sense information
        self.encoder_sns = EncoderGGNN(vocab_size, embedding_dim, hidden_size, n_hop)
        # this is sentence encoder for training VAE posterior
        self.out_enc = WordRNN(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_size,dropout=dropout)
        self.decoder = DecoderVAE(embedding_dim, hidden_size, z_dim,output_size,dropout)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.teacher_forcing = teacher_forcing
        # sample mu and logvars
        self.z_dim = z_dim

        self.q_mu_posterior = nn.Linear(3*hidden_size, z_dim)
        self.q_logvar_posterior = nn.Linear(3*hidden_size, z_dim)
        self.prior_fc1 = nn.Sequential(
            nn.Linear(2*hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.q_mu_prior = nn.Linear(hidden_size, z_dim)
        self.q_logvar_prior = nn.Linear(hidden_size, z_dim)
        # set embedding layer with the same parameters
        self.out_enc.embedding.weight = self.embedding.weight
    
    def sample_z(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std*eps; eps ~ N(0,I)
        """
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    def sample_z_prior(self, mbsize, device):
        z = torch.randn(mbsize, self.z_dim).to(device)
        return z
    
    def forward(self, input_equ_nodes, adj_equ_matrix, equ_node_lens, input_sns_nodes, adj_sns_matrix, sns_node_lens, input_target, scene, device):
        # encode equation input
        equ_node_resp = self.embedding(input_equ_nodes)
        equ_encoder_outputs, equ_encoder_hidden = self.encoder_equ(equ_node_resp, adj_equ_matrix, equ_node_lens)# bs*seq*h, bs*h
        # encode common sense input
        sns_node_resp = self.embedding(input_sns_nodes)
        sns_encoder_outputs, sns_encoder_hidden = self.encoder_sns(sns_node_resp, adj_sns_matrix, sns_node_lens)# bs*seq*h, bs*h
        # print('node representation dim is,', nodes_resp.shape)
        cond_embedding = torch.cat([equ_encoder_hidden, sns_encoder_hidden],1)
        output_embedding = self.out_enc(input_target)
        # get posterior mu & logvar
        recog_input = torch.cat([cond_embedding, output_embedding],1)
        recog_mu, recog_logvar = self.q_mu_posterior(recog_input), self.q_logvar_posterior(recog_input)
        # get prior mu & logvar
        prior_embedding = self.prior_fc1(cond_embedding)
        prior_mu, prior_logvar = self.q_mu_prior(prior_embedding), self.q_logvar_prior(prior_embedding)
        # sample latent z during training, sample posterior, inference with prior
        latent_sample = self.sample_z(recog_mu, recog_logvar)
        input_equ_node_mask = input_equ_nodes.eq(0)
        input_sns_node_mask = input_sns_nodes.eq(0)

        # train with schedule sampling
        logits, plan_attns = self.decode_dual(input_target, cond_embedding, equ_encoder_outputs,
            sns_encoder_outputs, latent_sample, self.teacher_forcing, device, input_equ_node_mask, input_sns_node_mask)
        logits = logits.view(-1, logits.size(2))
        return logits, recog_mu, recog_logvar, prior_mu, prior_logvar, plan_attns
    
    def decode_dual(self, dec_input_var, cond_embedding, equ_encoder_outputs, sns_encoder_outputs, \
        z_sample, teacher_forcing_ratio, device, input_equ_node_mask, input_sns_node_mask):
        bs, seq_len = dec_input_var.shape
        dec_hidden = torch.cat([cond_embedding, z_sample], dim=1)
        dec_input = torch.LongTensor([Constants.BOS] * bs).to(device)
        predicted_logits, graph_attntion = [], [] # use this to record the attention score
        # d_dec = d_initial
        for di in range(seq_len):
            if random.random() < self.teacher_forcing:
                prev_y = self.embedding(dec_input) # embedding look up table
                dec_output, dec_hidden, plan_attn = self.decoder(prev_y, dec_hidden, equ_encoder_outputs, sns_encoder_outputs, z_sample,input_equ_node_mask, input_sns_node_mask)
                predicted_logits.append(dec_output.squeeze(1))
                graph_attntion.append(plan_attn)
                dec_input = dec_input_var[:,di]
                dec_hidden = dec_hidden.squeeze(0)
            else:
                prev_y = self.embedding(dec_input) # embedding look up table
                dec_output, dec_hidden, plan_attn = self.decoder(prev_y, dec_hidden, equ_encoder_outputs, sns_encoder_outputs, z_sample,input_equ_node_mask, input_sns_node_mask)
                
                predicted_logits.append(dec_output.squeeze(1))
                graph_attntion.append(plan_attn)
                max_value, max_index = dec_output.squeeze(1).max(dim=-1)
                dec_input = max_index
                dec_hidden = dec_hidden.squeeze(0)
        predicted_logits = torch.stack(predicted_logits, 1)
        planning_probs = torch.stack(graph_attntion, 1)
        return predicted_logits, planning_probs
    
    def predict(self, input_equ_nodes, adj_equ_matrix, equ_node_lens, input_sns_nodes, adj_sns_matrix, sns_node_lens, scene, device, max_tgt_len):
        # encode equation input
        equ_node_resp = self.embedding(input_equ_nodes)
        equ_encoder_outputs, equ_encoder_hidden = self.encoder_equ(equ_node_resp, adj_equ_matrix, equ_node_lens)# bs*seq*h, bs*h
        # encode common sense input
        sns_node_resp = self.embedding(input_sns_nodes)
        sns_encoder_outputs, sns_encoder_hidden = self.encoder_sns(sns_node_resp, adj_sns_matrix, sns_node_lens)# bs*seq*h, bs*h

        # sample z from normal distribution
        batch_size = equ_node_resp.shape[0]

        # get prior mu & logvar
        cond_embedding = torch.cat([equ_encoder_hidden, sns_encoder_hidden],1)
        prior_embedding = self.prior_fc1(cond_embedding)
        prior_mu, prior_logvar = self.q_mu_prior(prior_embedding), self.q_logvar_prior(prior_embedding)

        # smaple latent z
        latent_sample = self.sample_z(prior_mu, prior_logvar)
        
        # decode
        dec_ids, graph_attntion = [],[]
        curr_token = Constants.BOS
        curr_dec_idx = 0
        dec_input_var = torch.LongTensor([curr_token]).to(device)
        dec_hidden = torch.cat([cond_embedding,latent_sample], dim=1)
        
        while (curr_token != Constants.EOS and curr_dec_idx <=max_tgt_len):
            prev_y = self.embedding(dec_input_var)
            # print(prev_y.shape)
            # print(dec_hidden.shape)
            # print(sns_encoder_outputs.shape)
            # print(latent_sample.shape)

            dec_output, dec_hidden, plan_attn = self.decoder(prev_y, dec_hidden, equ_encoder_outputs, sns_encoder_outputs, latent_sample,input_equ_node_mask=None, input_sns_node_mask=None)
            graph_attntion.append(plan_attn.data)
            max_value, max_index = dec_output.squeeze(1).max(dim=-1)
            #max_index = F.softmax(dec_output, dim=-1).squeeze(1).multinomial(1)
            dec_ids.append(max_index.squeeze().item())
            dec_input_var = max_index
            # print(max_index)
            dec_hidden = dec_hidden.squeeze(1)
            curr_dec_idx += 1
            curr_token = max_index.item()
        return dec_ids, graph_attntion
    
    def predict_with_sampling(self, input_nodes, adj_matrix, node_lens, device, max_tgt_len,temp=None, k=None,p=None,m=None):
        """
        some sampling based decoding method:
        tem: temperature
        k: top-k sampling method k
        p: Nucleus sampling method
        m: mass of original dist to interpolate
        """
        # embedding look up table
        nodes_resp = self.embedding(input_nodes)
        encoder_outputs, encoder_hidden = self.encoder(nodes_resp, adj_matrix, node_lens) # bs*seq*h, bs*h
        
        # decode
        dec_ids, attn_weight = [],[]
        curr_token = Constants.BOS
        curr_dec_idx = 0
        dec_input_var = torch.LongTensor([curr_token]).to(device)
        dec_hidden = encoder_hidden
        while (curr_token != Constants.EOS and curr_dec_idx <=max_tgt_len):
            prev_y = self.embedding(dec_input_var)
            decoder_output, dec_hidden, dec_attn = self.decoder(prev_y, dec_hidden, encoder_outputs, input_node_mask=None)
            
            probs = F.softmax(decoder_output, dim=-1).squeeze(1)
            logprobs = F.log_softmax(decoder_output, dim=-1).squeeze(1)
            
            if temp is not None:
                samp_probs = F.softmax(decoder_output.div_(temp), dim=-1).squeeze(1)
            else:
                samp_probs = probs.clone()
            if k is not None:
                indices_to_remove = samp_probs < torch.topk(samp_probs,k)[0][...,-1,None]
                samp_probs[indices_to_remove] = 0.
                if m is not None:
                    samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
                    samp_probs.mul_(1-m)
                    samp_probs.add_(probs.mul_(m))
                next_tokens = samp_probs.multinomial(1)
                next_logprobs = samp_probs.gather(1, next_tokens.view(-1,1)).log()
            elif p is not None:
                sorted_probs, sorted_indices = torch.sort(samp_probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                sorted_samp_probs = sorted_probs.clone()
                sorted_samp_probs[sorted_indices_to_remove] = 0
                if m is not None:
                    sorted_samp_probs.div_(sorted_samp_probs.sum(1).unsqueeze(1))
                    sorted_samp_probs.mul_(1-m)
                    sorted_samp_probs.add_(sorted_probs.mul(m))
                sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
                next_tokens = sorted_indices.gather(1, sorted_next_indices)
                next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()
            else:
                if m is not None:
                    samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
                    samp_probs.mul_(1-m)
                    samp_probs.add_(probs.mul(m))
                next_tokens = samp_probs.multinomial(1)
                next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()
            dec_ids.append(next_tokens.squeeze().item())
            dec_input_var = next_tokens.squeeze(1)
            dec_hidden = dec_hidden.squeeze(0)
            curr_dec_idx += 1
            curr_token = next_tokens.item()
        return dec_ids, attn_weight

if __name__ == "__main__":
    device = torch.device('cpu')
    nodes = torch.LongTensor([[1,2,3,4,5],[2,3,4,5,6],[1,2,3,0,0]]).to(device)
    adjmatrixs = torch.FloatTensor([
        [[0,1,0,0,0],[1,0,1,0,0],[1,0,0,0,1],[1,1,1,1,1],[0,0,0,0,1]],
        [[0,1,0,1,0],[0,0,1,0,0],[1,0,0,0,1],[1,1,0,0,1],[0,0,0,0,1]],
        [[0,1,0,1,0],[0,0,1,0,0],[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    ]).to(device)
    scene = torch.LongTensor([[1,2],[2,3],[3,4]])
    lengths = torch.FloatTensor([5,4,3]).to(device)
    # GGNN = EncoderGGNN(10, 8, 7).to(device)
    # gate_out, features = GGNN(nodes, adjmatrixs, lengths)
    # print('gate out shape is:',gate_out.shape)
    # print('feature out shape is:', features.shape)
    # print(gate_out)
    # print(features)
    input_target = torch.LongTensor([[1,2,3,4,3,2,1,0,0,0],[2,2,3,4,5,6,7,8,9,0],[2,2,3,4,5,6,1,2,3,0]]).to(device)
    nodes_2 = torch.LongTensor([[1,2,3,4,5,6], [2,3,4,5,6,7],[3,4,5,0,0,0]]).to(device)
    adjmatrixs_2 = torch.FloatTensor([
        [[0,1,0,0,0,1],[1,0,1,0,0,1],[1,0,1,0,0,1],[1,1,1,1,1,1],[1,0,0,0,0,1],[1,0,1,0,0,1]],
        [[0,1,0,1,0,1],[0,0,1,0,0,1],[1,0,0,1,0,1],[1,1,0,0,1,1],[0,0,0,0,1,1],[1,0,1,0,0,1]],
        [[0,1,0,1,0,1],[0,0,1,0,0,1],[1,0,1,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
    ]).to(device)
    lengths2 = torch.FloatTensor([6,6,3])
    
    g2seq = Graph2seq(vocab_size = 10, embedding_dim=6, hidden_size=7, z_dim=13, output_size=10).to(device)
    # logits, recog_mu, recog_logvar, prior_mu, prior_logvar = g2seq(nodes, adjmatrixs, lengths, input_target,scene,device)
    # print(logits.shape)

    logits, recog_mu, recog_logvar, prior_mu, prior_logvar, attn_weights = g2seq(nodes, adjmatrixs, lengths, nodes_2, adjmatrixs_2, lengths2,input_target,scene,device)
    print(logits.shape)
    print(attn_weights.shape)
    print(attn_weights)
    ids, _ = g2seq.predict(input_equ_nodes=nodes[0].unsqueeze(0), adj_equ_matrix=adjmatrixs[0].unsqueeze(0), equ_node_lens=lengths[0].unsqueeze(0), \
        input_sns_nodes=nodes_2[0].unsqueeze(0), adj_sns_matrix=adjmatrixs_2[0].unsqueeze(0), sns_node_lens=lengths2[0].unsqueeze(0),\
        scene=scene[0].unsqueeze(0), device=device, max_tgt_len=10)
    # print(ids)









