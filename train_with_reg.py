'''This script handling the training process'''
import argparse
import math
import time

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

from dataset_dual import MyDataset, collate_fn
from model.Constant import Constants
from model.dual_graph_vae_2 import Graph2seq, ScheduledOptim
from utils.cyclical_annealing import frange_cycle_linear
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


Constants = Constants()

def cal_performence(pred, gold, mu_prior, log_var_prior, mu_posterior, log_var_posterior, plan_attns, lambda_kl):
    """
    Apply label smooth if needed
    """
    loss, loss_recon, loss_kl, loss_sparse = cal_loss(pred, gold, mu_prior, log_var_prior, mu_posterior, log_var_posterior, plan_attns, lambda_kl)
    pred = pred.max(1)[1]
    gold = gold.contiguous().view(-1)
    non_pad_mask = gold.ne(Constants.PAD)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()
    return loss, n_correct, loss_recon, loss_kl, loss_sparse
    
def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
    return torch.mean(kld)

def sparse_resularizer(attn_matrix):
    attn_matrix = attn_matrix.view(-1, attn_matrix.shape[-1])
    L_sparse = 0.5 * torch.sum(torch.abs(torch.sum(attn_matrix**2,1)-1),0)
    return L_sparse

# def sparse_resularizer(attn_matrix):
#     L_sparse = 0.5 * torch.mean(torch.abs(torch.sum(attn_matrix**2, dim=-1)-1))
#     return L_sparse

def cal_loss(pred, gold, mu_prior, log_var_prior, mu_posterior, log_var_posterior, plan_attns, lambda_kl):
    """
    Calculate cross entropy loss, apply label smoothing if needed
    """
    gold = gold.contiguous().view(-1)
    loss_recon = F.cross_entropy(pred, gold, ignore_index=Constants.PAD, reduction='sum')
    # loss_kl = lambda_kl*-0.5 * torch.sum(1 + log_var - mu.pow(2)-log_var.exp())
    loss_kl = lambda_kl*gaussian_kld(mu_posterior, log_var_posterior, mu_prior, log_var_prior)
    loss_sparse = 0.05*sparse_resularizer(plan_attns)
    return loss_recon + loss_kl+ loss_sparse, loss_recon, loss_kl, loss_sparse

def train_epoch(model, training_data, optimizer, scheduler,device, smoothing, lambda_kl):
    '''Epoch operation in training phase'''
    model.train()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    total_loss_recon, total_loss_kl, total_loss_sparse = 0,0,0
    total_sen = 0

    for batch in tqdm(training_data, mininterval=2, desc = ' -(Training) ', leave=False):

        # prepare data
        equ_nodes, sns_nodes, equ_node_lens, sns_node_lens, equ_adj_matrixs, sns_adj_matrixs, tgt_seq, scene = map(lambda x: x.to(device), batch)
        #print('src_seq shape:', src_seq.shape)
        #print('src_pos shape:', src_pos.shape)
        #print('tgt_seq shape:', tgt_seq.shape)
        #print('tgt_pos shape:', tgt_pos.shape)
        gold = tgt_seq
        # forward
        optimizer.zero_grad()
        
        pred, recog_mu, recog_logvar, prior_mu, prior_logvar, plan_attns = model(equ_nodes, equ_adj_matrixs, equ_node_lens, sns_nodes,sns_adj_matrixs,sns_node_lens,tgt_seq,scene,device)
        #print('pred shape', pred.shape)
        #print('gold shape', gold.shape)

        # backward
        loss, n_correct, loss_recon, loss_kl, loss_sparse = cal_performence(pred, gold, prior_mu, prior_logvar, recog_mu, recog_logvar, plan_attns, lambda_kl)
        #print(loss)
        #print(n_correct)
        loss.backward()

        # update parameters
        optimizer.step()
        scheduler.step()

        # note keeping
        total_loss += loss.item()
        total_loss_kl += loss_kl.item()
        total_loss_recon += loss_recon.item()
        total_loss_sparse += loss_sparse.item()
        total_sen += len(equ_nodes)

        non_pad_mask = gold.ne(Constants.PAD)
        n_word = non_pad_mask.sum().item()
        n_word_total += n_word
        n_word_correct += n_correct
    loss_per_word = total_loss /n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy,total_loss_recon/n_word_total, total_loss_kl/total_sen, total_loss_sparse/total_sen


def eval_epoch(model, validation_data, device):
    '''Epoch operation in evaluation phase'''
    model.eval()
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0
    total_loss_recon, total_loss_kl, total_loss_sparse = 0, 0, 0
    total_sen=0
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc=' -(Validation) ',leave = False):
            # prepare data
            equ_nodes, sns_nodes, equ_node_lens, sns_node_lens, equ_adj_matrixs, sns_adj_matrixs, tgt_seq, scene = map(lambda x: x.to(device), batch)
            gold = tgt_seq

            # forward
            pred, recog_mu, recog_logvar, prior_mu, prior_logvar, plan_attns = model(equ_nodes, equ_adj_matrixs, equ_node_lens, sns_nodes,sns_adj_matrixs,sns_node_lens,tgt_seq,scene,device)
            loss, n_correct, loss_recon, loss_kl, loss_sparse = cal_performence(pred, gold, prior_mu, prior_logvar, recog_mu, recog_logvar, plan_attns, lambda_kl=1)

            # note keeping
            total_loss += loss.item()

            non_pad_mask = gold.ne(Constants.PAD)
            n_word = non_pad_mask.sum().item()
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss_kl += loss_kl.item()
            total_sen += len(equ_nodes)
            total_loss_recon += loss_recon.item()
            total_loss_sparse += loss_sparse.item()
        
    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    return loss_per_word, accuracy, total_loss_recon/n_word_total, total_loss_kl/total_sen, total_loss_sparse/total_sen


def train(model, training_data, validation_data, optimizer, scheduler, device, idx2word, opt):
    '''Start training'''
    log_train_file = None
    log_valid_file = None
    beta_epochs = frange_cycle_linear(start=0.0, stop=1.0, n_epoch=opt.epoch)

    if opt.log:
        log_train_file = opt.log + '.train.log'
        log_valid_file = opt.log + '.valid.log'

        print('[Info] Training performence will be written to file: {} and {}'.format(log_train_file, log_valid_file))
        with open(log_train_file, 'w') as log_tf, open(log_valid_file,'w') as log_vf:
            log_tf.write('epoch,loss,ppl,accuracy\n')
            log_vf.write('epoch,loss,ppl,accuracy\n')
    valid_accus = []
    for epoch_i in range(opt.epoch):
        beta_this_epoch = beta_epochs[epoch_i]
        print('[ Epoch',epoch_i,' ]')
        start = time.time()
        train_loss, train_accu, train_loss_recon, train_loss_kl, train_loss_sparse = train_epoch(
            model, training_data, optimizer, scheduler,device, smoothing= opt.label_smoothing, lambda_kl=beta_this_epoch
        )
        print(' -(Trianing) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f}, train_loss_recon: {recon: 8.5f}, train_loss_kl:{kl: 8.5f}, train_loss_sparse:{sp: 8.5f}, elapse: {elapse:3.3f} min'.format(ppl=math.exp(min(train_loss,100)),accu=100*train_accu,
                                                                                                            recon=train_loss_recon, kl=train_loss_kl, sp=train_loss_sparse,elapse=(time.time()-start)/60))
        start = time.time()
        valid_loss, valid_accu, valid_loss_recon, valid_loss_kl, valid_loss_sparse = eval_epoch(model,validation_data, device)
        print(' -(Validation) ppl: {ppl: 8.5f}, accuracy: {accu:3.3f}, valid_loss_recon: {recon: 8.5f}, valid_loss_kl:{kl: 8.5f}, valid_loss_sparse: {sp: 8.5f}, elapse: {elapse:3.3f} min'.format(ppl=math.exp(min(valid_loss,100)),accu=100*valid_accu,
                                                                                                            recon=valid_loss_recon, kl=valid_loss_kl, sp=valid_loss_sparse,elapse=(time.time()-start)/60))
        valid_accus += [valid_accu]
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': opt,
            'epoch': epoch_i
        }
        if opt.save_model:
            if opt.save_mode == 'all':
                model_name = opt.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
                torch.save(checkpoint, model_name)
            if opt.save_mode == 'best':
                model_name = opt.save_model + '.chkpt'
                if valid_accu >= max(valid_accus):
                    torch.save(checkpoint, model_name)
                    print(' -[Info] The check point file has been updated.')
                    sample_generation(model, training_data, idx2word,device)
        
        if log_train_file and log_valid_file:
            with open(log_train_file,'a') as log_tf, open(log_valid_file,'a') as log_vf:
                log_tf.write('{epoch}, {loss: 8.5f},{ppl: 8.5f},{accu: 3.3f}\n'.format(
                    epoch = epoch_i, loss=train_loss, ppl=math.exp(min(train_loss, 100)), accu=100*train_accu
                ))
                log_vf.write('{epoch}, {loss: 8.5f},{ppl: 8.5f},{accu: 3.3f}\n'.format(
                    epoch = epoch_i, loss=valid_loss, ppl=math.exp(min(valid_loss, 100)), accu=100*valid_accu
                ))
                
def sample_generation(model, train_loader, idx2word, device):
    for batch in train_loader:
        equ_nodes, sns_nodes, equ_node_lens, sns_node_lens, equ_adj_matrixs, sns_adj_matrixs, tgt_seq, scene = map(lambda x: x.to(device), batch)
    print('show case during training')
    show_case, show_attn = [],[]
    with torch.no_grad():
        for i in range(3):
            dec_ids, attn_matrix = model.predict(
                input_equ_nodes=equ_nodes[i].unsqueeze(0), adj_equ_matrix=equ_adj_matrixs[i].unsqueeze(0), equ_node_lens=equ_node_lens[i].unsqueeze(0), \
            input_sns_nodes=sns_nodes[i].unsqueeze(0), adj_sns_matrix=sns_adj_matrixs[i].unsqueeze(0), sns_node_lens=sns_node_lens[i].unsqueeze(0),\
            scene=scene[i].unsqueeze(0), device=device, max_tgt_len=50)
            show_case.append(''.join(idx2word[x] for x in dec_ids))
    print('one attention matrix is {}'.format(torch.stack(attn_matrix,1)))
    print(show_case[0]+'\n')
    print(show_case[1] + '\n')
    print(show_case[2] + '\n')

def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('-data',default='./processed_data/dual_graph_rev_res.pt')
    parser.add_argument('-epoch', type=int, default=200)
    parser.add_argument('-batch_size',type=int, default=16)
    
    parser.add_argument('-embedding_dim',type=int, default=256) #node dim same as this
    parser.add_argument('-n_hop',type=int, default=3)
    parser.add_argument('-hidden_size',type=int, default=512)
    parser.add_argument('-z_dim', type=int,default=128)
    parser.add_argument('-teacher_forcing',type=float, default=0.5)

    parser.add_argument('-n_warmup_steps',type=int, default=2000)
    
    parser.add_argument('-dropout',type=float,default=0.1)
    
    parser.add_argument('-log',default=None)
    parser.add_argument('-save_model',default=None)
    parser.add_argument('-save_mode', type=str, choices=['all','best'], default='best')

    parser.add_argument('-no_cuda',action='store_true')
    parser.add_argument('-label_smoothing',action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    #====== Loading Dataset =====#
    data = torch.load(opt.data)
    opt.max_token_seq_len = max(len(x) for x in data['train']['ref'])
    
    training_data, validation_data = prepare_dataloaders(data,opt)
    opt.vocab_size = training_data.dataset.src_vocab_size

    #======= Preparing model ====#
    print(opt)
    device = torch.device('cuda:3' if opt.cuda else 'cpu')
    # device = torch.device('cpu')
    graph2seq = Graph2seq(
        vocab_size=opt.vocab_size, 
        embedding_dim = opt.embedding_dim, 
        hidden_size = opt.hidden_size, 
        z_dim =opt.z_dim,
        output_size = opt.vocab_size,
        n_hop=opt.n_hop,
        teacher_forcing=opt.teacher_forcing,
        dropout=0.1).to(device)

    # optimizer = ScheduledOptim(
    #     optim.Adam(
    #         filter(lambda x: x.requires_grad, graph2seq.parameters()),
    #         betas=(0.9,0.98),eps=1e-09),
    #     opt.hidden_size, opt.n_warmup_steps
    # )
    optimizer = AdamW(graph2seq.parameters(),
                lr = 1e-3, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
    # Number of training epochs (authors recommend between 2 and 4)
    epochs = opt.epoch

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(training_data) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)
    # train(model, tokenizer, train_loader, valid_loader, optimizer, scheduler, device, opt)
    
    idx2word = {value:item for item, value in data['dict']['tgt'].items()}
    train(graph2seq, training_data, validation_data, optimizer, scheduler, device,idx2word,opt)

def prepare_dataloaders(data, opt):
    # =====Prepareing DataLoader=====
    train_loader = torch.utils.data.DataLoader(
        MyDataset(
            src_word2idx = data['dict']['tgt'],
            tgt_word2idx = data['dict']['tgt'],
            node_insts= data['train']['node_1'],# equation info
            rel_insts = data['train']['edge_1'],
            node_insts_1 = data['train']['node_2'],# common sense info
            rel_insts_1 = data['train']['edge_2'],
            scene_insts = data['train']['scene'],
            tgt_insts = data['train']['ref']
        ),
        num_workers = 4,
        batch_size=opt.batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        MyDataset(
            src_word2idx = data['dict']['tgt'],
            tgt_word2idx = data['dict']['tgt'],
            node_insts= data['dev']['node_1'],# equation info
            rel_insts = data['dev']['edge_1'],
            node_insts_1 = data['dev']['node_2'],# common sense info
            rel_insts_1 = data['dev']['edge_2'],
            scene_insts = data['dev']['scene'],
            tgt_insts = data['dev']['ref']
        ),
        num_workers = 4,
        batch_size=opt.batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )
    return train_loader, valid_loader




if __name__ == "__main__":
    main()




