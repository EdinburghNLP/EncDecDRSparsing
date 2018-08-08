    # -*- coding: utf-8 -*-
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import os
import sys
import types

from joint_mask import OuterMask
from joint_mask import RelationMask
from joint_mask import VariableMask


use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.manual_seed_all(12345678)

torch.manual_seed(12345678)

dev_out_dir = "output_dev/"
tst_out_dir = "output_tst/"
model_dir = "output_model/"

class EncoderRNN(nn.Module):
    def __init__(self, word_size, word_dim, pretrain_size, pretrain_dim, pretrain_embeddings, lemma_size, lemma_dim, input_dim, hidden_dim, n_layers=1, dropout_p=0.0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.word_embeds = nn.Embedding(word_size, word_dim)
        self.pretrain_embeds = nn.Embedding(pretrain_size, pretrain_dim)
        self.pretrain_embeds.weight = nn.Parameter(pretrain_embeddings, False)
        self.lemma_embeds = nn.Embedding(lemma_size, lemma_dim)
        self.dropout = nn.Dropout(self.dropout_p)

        self.embeds2input = nn.Linear(word_dim + pretrain_dim + lemma_dim, input_dim)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.n_layers, bidirectional=True)

    def forward(self, sentence, hidden, train=True):
        word_embedded = self.word_embeds(sentence[0])
        pretrain_embedded = self.pretrain_embeds(sentence[1])
        lemma_embedded = self.lemma_embeds(sentence[2])

        if train:
            word_embedded = self.dropout(word_embedded)
            lemma_embedded = self.dropout(lemma_embedded)
            self.lstm.dropout = self.dropout_p

        embeds = self.tanh(self.embeds2input(torch.cat((word_embedded, pretrain_embedded, lemma_embedded), 1))).view(len(sentence[0]),1,-1)
        output, hidden = self.lstm(embeds, hidden)
        return output, hidden

    def initHidden(self):
        if use_cuda:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda(),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)).cuda())
            return result
        else:
            result = (Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)),
                Variable(torch.zeros(2*self.n_layers, 1, self.hidden_dim)))
            return result

class AttnDecoderRNN(nn.Module):
    def __init__(self, outer_mask_pool, rel_mask_pool, var_mask_pool, tags_info, tag_dim, input_dim, feat_dim, encoder_hidden_dim, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.outer_mask_pool = outer_mask_pool
        self.rel_mask_pool = rel_mask_pool
        self.var_mask_pool = var_mask_pool
        self.total_rel = 0

        self.tags_info = tags_info
        self.tag_size = tags_info.tag_size
        self.all_tag_size = tags_info.all_tag_size

        self.tag_dim = tag_dim
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.hidden_dim = encoder_hidden_dim * 2

        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(self.dropout_p)
        self.tag_embeds = nn.Embedding(self.tags_info.all_tag_size, self.tag_dim)

        self.struct2rel = nn.Linear(self.hidden_dim, self.tag_dim)
        self.rel2var = nn.Linear(self.hidden_dim, self.tag_dim)

        #self.struct_lstm = nn.LSTM(self.tag_dim, self.hidden_dim, num_layers= self.n_layers)
        #self.rel_lstm = nn.LSTM(self.tag_dim, self.hidden_dim, num_layers= self.n_layers)
        #self.var_lstm = nn.LSTM(self.tag_dim, self.hidden_dim, num_layers= self.n_layers)
        self.lstm = nn.LSTM(self.tag_dim, self.hidden_dim, num_layers= self.n_layers)

        self.feat = nn.Linear(self.hidden_dim + self.tag_dim, self.feat_dim)
        self.feat_tanh = nn.Tanh()

        #self.out_struct = nn.Linear(self.feat_dim, self.tag_size)
        #self.out_rel = nn.Linear(self.feat_dim, self.tag_size)
        #self.out_var = nn.Linear(self.feat_dim, self.tag_size)
	self.out = nn.Linear(self.feat_dim, self.tag_size)

        self.selective_matrix = Variable(torch.randn(1, self.hidden_dim, self.hidden_dim))
        if use_cuda:
            self.selective_matrix = self.selective_matrix.cuda()

    def forward(self, sentence_variable, inputs, hidden, encoder_output, least, train, mask_variable, opt):

        if opt == 1:
            return self.forward_1(inputs, hidden, encoder_output, train, mask_variable)
        elif opt == 2:
            return self.forward_2(sentence_variable, inputs, hidden, encoder_output, least, train, mask_variable)
        elif opt == 3:
            return self.forward_3(inputs, hidden, encoder_output, train, mask_variable)
        else:
            assert False, "unrecognized option"
    def forward_1(self, input, hidden, encoder_output, train, mask_variable):
        if train:
            self.lstm.dropout = self.dropout_p
            embedded = self.tag_embeds(input).unsqueeze(1)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)
            
            attn_weights = F.softmax(torch.bmm(output.transpose(0,1), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0),-1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0),encoder_output.transpose(0,1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded.transpose(0,1)), 2).view(output.size(0),-1)))

            global_score = self.out(feat_hiddens)

            log_softmax_output = F.log_softmax(global_score + (mask_variable - 1) * 1e10, 1)

            return log_softmax_output, output
        else:

            self.lstm.dropout = 0.0
            tokens = []
            self.outer_mask_pool.reset()
            hidden_rep = []
            while True:
                mask = self.outer_mask_pool.get_step_mask()
                mask_variable = Variable(torch.FloatTensor(mask), requires_grad = False, volatile=True).unsqueeze(0)
                mask_variable = mask_variable.cuda() if use_cuda else mask_variable
                
                embedded = self.tag_embeds(input).view(1, 1, -1)
                output, hidden = self.lstm(embedded, hidden)
                hidden_rep.append(output)

                attn_weights = F.softmax(torch.bmm(output, encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1), 1)
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
                feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0),-1)))

                global_score = self.out(feat_hiddens)

                score = global_score + (mask_variable - 1) * 1e10

                _, input = torch.max(score,1)
                idx = input.view(-1).data.tolist()[0]

                tokens.append(idx)
                self.outer_mask_pool.update(-2, idx)

                if idx == tags_info.tag_to_ix[tags_info.EOS]:
                    break
            return Variable(torch.LongTensor(tokens),volatile=True), torch.cat(hidden_rep,0), hidden

    def forward_2(self, sentence_variable, inputs, hidden, encoder_output, least, train, mask_variable):

        if train:
            self.lstm.dropout = self.dropout_p
            List = []
            for condition, input in inputs:
                List.append(self.struct2rel(condition).view(1, 1, -1))
                if type(input) == types.NoneType:
                    pass
                else:
                    List.append(self.tag_embeds(input).unsqueeze(1))
            embedded = torch.cat(List, 0)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)

            selective_score = torch.bmm(torch.bmm(output.transpose(0,1), self.selective_matrix), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1)

            attn_weights = F.softmax(torch.bmm(output.transpose(0,1), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0),-1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0),encoder_output.transpose(0,1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded.transpose(0,1)), 2).view(output.size(0),-1)))

            global_score = self.out(feat_hiddens)

            total_score = torch.cat((global_score, selective_score), 1)

            log_softmax_output = F.log_softmax(total_score + (mask_variable - 1) *1e10, 1)

            return log_softmax_output, output

        else:
            self.lstm.dropout = 0.0
            tokens = []
            rel = 0
            hidden_reps = []

            mask_variable_true = Variable(torch.FloatTensor(rel_mask_pool.get_step_mask(True)), requires_grad = False)
            mask_variable_false = Variable(torch.FloatTensor(rel_mask_pool.get_step_mask(False)), requires_grad = False)
            if use_cuda:
                mask_variable_true = mask_variable_true.cuda()
                mask_variable_false = mask_variable_false.cuda()
            embedded = self.struct2rel(inputs).view(1, 1,-1)

            while True:
                
                output, hidden = self.lstm(embedded, hidden)
                hidden_reps.append(output)

                selective_score = torch.bmm(torch.bmm(output, self.selective_matrix), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1)
                attn_weights = F.softmax(torch.bmm(output, encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1), 1)
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
                feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0),-1)))

                global_score = self.out(feat_hiddens)

                total_score = torch.cat((global_score, selective_score), 1)

                if least:
                    output = total_score + (mask_variable_true - 1) * 1e10
                    least = False
                else:
                    output = total_score + (mask_variable_false - 1) * 1e10

                _, input = torch.max(output,1)
                idx = input.view(-1).data.tolist()[0]

                if idx >= tags_info.tag_size:
                    ttype = idx - tags_info.tag_size
                    idx = sentence_variable[2][ttype].view(-1).data.tolist()[0]
                    idx += tags_info.tag_size
                    tokens.append(idx)
                    input = Variable(torch.LongTensor([idx]), volatile=True)
                    if use_cuda:
                        input = input.cuda()
                else:
                    tokens.append(idx)

                if idx == tags_info.tag_to_ix[tags_info.EOS]:
                    break
                elif rel > 61 or self.total_rel > 121:
                    embedded = self.tag_embeds(input).view(1, 1, -1)
                    output, hidden = self.lstm(embedded, hidden)
                    hidden_reps.append(output)
                    break
                rel += 1
                self.total_rel += 1
                embedded = self.tag_embeds(input).view(1, 1, -1)
            return Variable(torch.LongTensor(tokens), volatile=True), torch.cat(hidden_reps,0), hidden

    def forward_3(self, inputs, hidden, encoder_output, train, mask_variable):
        if train:
            self.lstm.dropout = self.dropout_p

            List = []
            for condition, input in inputs:
                List.append(self.rel2var(condition).view(1, 1, -1))
                List.append(self.tag_embeds(input).unsqueeze(1))
            embedded = torch.cat(List, 0)
            embedded = self.dropout(embedded)

            output, hidden = self.lstm(embedded, hidden)

            attn_weights = F.softmax(torch.bmm(output.transpose(0,1), encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0),-1), 1)
            attn_hiddens = torch.bmm(attn_weights.unsqueeze(0),encoder_output.transpose(0,1))
            feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded.transpose(0,1)), 2).view(output.size(0),-1)))

            global_score = self.out(feat_hiddens)

            score = global_score

            log_softmax_output = F.log_softmax(score + (mask_variable - 1) *1e10, 1)

            return log_softmax_output, output
        else:
            self.lstm.dropout = 0.0
            tokens = []
            embedded = self.rel2var(inputs).view(1, 1,-1)
            while True:
                output, hidden = self.lstm(embedded, hidden)

                attn_weights = F.softmax(torch.bmm(output, encoder_output.transpose(0,1).transpose(1,2)).view(output.size(0), -1), 1)
                attn_hiddens = torch.bmm(attn_weights.unsqueeze(0), encoder_output.transpose(0, 1))
                feat_hiddens = self.feat_tanh(self.feat(torch.cat((attn_hiddens, embedded), 2).view(embedded.size(0),-1)))

                global_score = self.out(feat_hiddens)

                mask = self.var_mask_pool.get_step_mask()
                mask_variable = Variable(torch.FloatTensor(mask), volatile=True)
                mask_variable = mask_variable.cuda() if use_cuda else mask_variable

                score = global_score + (mask_variable - 1) * 1e10

                _, input = torch.max(score, 1)
                embedded = self.tag_embeds(input).view(1, 1, -1)

                idx = input.view(-1).data.tolist()[0]
                assert idx < tags_info.tag_size
                if idx == tags_info.tag_to_ix[tags_info.EOS]:
                    break
                    
                tokens.append(idx)
                self.var_mask_pool.update(idx)
                
            return Variable(torch.LongTensor(tokens), volatile=True), hidden


def train(sentence_variable, input_variables, gold_variables, mask_variables, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, back_prop=True):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length1 = 0
    target_length2 = 0
    target_length3 = 0

    encoder_hidden = encoder.initHidden()
    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)

    ################structure
    decoder_hidden1 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
    decoder_input1 = input_variables[0]
    decoder_output1, hidden_rep1 = decoder(None, decoder_input1, decoder_hidden1, encoder_output, least=None, train=True, mask_variable=mask_variables[0], opt=1)
    gold_variable1 = gold_variables[0]
    loss1 = criterion(decoder_output1, gold_variable1)
    target_length1 += gold_variable1.size(0)

    ################ relation
    decoder_hidden2 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
    decoder_input2 = []
    structs = input_variables[0].view(-1).data.tolist() # SOS DRS( P1( DRS( P2( DRS( ) ) ) ) )
    p = 0
    for i in range(len(structs)):
        if structs[i] == 5 or structs[i] == 6:
            decoder_input2.append((hidden_rep1[i], input_variables[1][p]))
            p += 1
    assert p == len(input_variables[1])
    decoder_output2, hidden_rep2 = decoder(sentence_variable, decoder_input2, decoder_hidden2, encoder_output, least=None, train=True, mask_variable=mask_variables[1], opt=2)
    loss2 = criterion(decoder_output2, gold_variables[1])
    target_length2 += gold_variables[1].size(0)

    ################ variable
    decoder_hidden3 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
    decoder_input3 = []
    ##### decoder hidden is like
    #####   DRS( india( say( CAUSE( TOPIC( 
    #####   DRS( THING( security( increase( PATIENT( ATTRIBUTE( FOR( 
    #####   DRS( TOPIC( possible( TOPIC( militant( strike( country( in( thwart( AGENT( THEME(
    i = 0
    p = 0
    for j in range(len(input_variables[1])):
        i += 1
        if type(input_variables[1][j]) == types.NoneType:
            pass
        else:
            for k in range(input_variables[1][j].size(0)) :
                decoder_input3.append((hidden_rep2[i], input_variables[2][p]))
                i += 1
                p += 1
    assert p == len(input_variables[2])
    decoder_output3, hidden_rep3 = decoder(None, decoder_input3, decoder_hidden3, encoder_output, least=None, train=True, mask_variable=mask_variables[2], opt=3)
    loss3 = criterion(decoder_output3, gold_variables[2])
    target_length3 += gold_variables[2].size(0)

    loss = loss1 + loss2 + loss3
    if back_prop:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss1.data[0] / target_length1, loss2.data[0] / target_length2, loss3.data[0] / target_length3

def decode(sentence_variable, encoder, decoder):
    encoder_hidden = encoder.initHidden()
    encoder_output, encoder_hidden = encoder(sentence_variable, encoder_hidden)
    
    ####### struct
    decoder_hidden1 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))

    decoder_input1 = Variable(torch.LongTensor([0]), volatile=True)
    decoder_input1 = decoder_input1.cuda() if use_cuda else decoder_input1
    decoder_output1, hidden_rep1, decoder_hidden1 = decoder(None, decoder_input1, decoder_hidden1, encoder_output, least=None, train=False, mask_variable=None, opt=1)
    structs = decoder_output1.view(-1).data.tolist()

    ####### relation
    decoder_hidden2 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
    decoder.rel_mask_pool.reset(sentence_variable[0].size(0))
    decoder.total_rel = 0
    relations = []
    hidden_rep2_list = []
    for i in range(len(structs)):
        if structs[i] == 5 or structs[i] == 6: # prev output, and hidden_rep1[i+1] is the input representation of prev output.
            least = False
            if structs[i] == 5 or (structs[i] == 6 and structs[i+1] == 4):
                least = True
            decoder.rel_mask_pool.set_sdrs(structs[i] == 5)
            decoder_output2, hidden_rep2, decoder_hidden2 = decoder(sentence_variable, hidden_rep1[i+1], decoder_hidden2, encoder_output, least=least, train=False, mask_variable=None, opt=2)
            relations.append(decoder_output2.view(-1).data.tolist())
            hidden_rep2_list.append(hidden_rep2)
    ####### variable
    decoder_hidden3 = (torch.cat((encoder_hidden[0][-2], encoder_hidden[0][-1]), 1).unsqueeze(0),torch.cat((encoder_hidden[1][-2], encoder_hidden[1][-1]), 1).unsqueeze(0))
    #p_max
    p_max = 0
    for tok in structs:
        if tok >= decoder.tags_info.p_rel_start and tok < decoder.tags_info.k_tag_start:
            p_max += 1
    #user_k
    user_k = []
    stack = []
    for tok in structs:
        if tok == 4:
            if stack[-1][0] == 5:
                user_k.append(stack[-1][1])
            stack.pop()
        else:
            if tok >= decoder.tags_info.k_rel_start and tok < decoder.tags_info.p_rel_start:
                stack[-1][1].append(tok - decoder.tags_info.k_rel_start)
            stack.append([tok,[]])
    decoder.var_mask_pool.reset(p_max, k_use=True)
    structs_p = 0
    user_k_p = 0
    struct_rel_tokens = []
    var_tokens = []
    for i in range(len(structs)):
        if structs[i] == 1: # EOS
            continue
        decoder.var_mask_pool.update(structs[i])
        struct_rel_tokens.append(structs[i])
        if structs[i] == 5 or structs[i] == 6:
            if structs[i] == 5:
                assert len(user_k[user_k_p]) >= 2
                decoder.var_mask_pool.set_k(user_k[user_k_p])
                user_k_p += 1

            for j in range(len(relations[structs_p])):
                if relations[structs_p][j] == 1: # EOS
                    continue
                decoder.var_mask_pool.update(relations[structs_p][j])
                struct_rel_tokens.append(relations[structs_p][j])
                decoder_output3, decoder_hidden3= decoder(None, hidden_rep2_list[structs_p][j+1], decoder_hidden3, encoder_output, least=None, train=False, mask_variable=None, opt=3)
                var_tokens.append(decoder_output3.view(-1).data.tolist())
                decoder.var_mask_pool.update(4)
                struct_rel_tokens.append(4)
            structs_p += 1
    assert structs_p == len(relations)

    return struct_rel_tokens, var_tokens

######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(trn_instances, dev_instances, tst_instances, encoder, decoder, print_every=100, evaluate_every=1000, learning_rate=0.001):
    print_loss_total = 0.0 # Reset every print_every
    print_loss_total1 = 0.0
    print_loss_total2 = 0.0
    print_loss_total3 = 0.0

    criterion = nn.NLLLoss()

    check_point = {}
    if len(sys.argv) == 4:
        check_point = torch.load(sys.argv[3])
        encoder.load_state_dict(check_point["encoder"])
        decoder.load_state_dict(check_point["decoder"])
        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()

    encoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate, weight_decay=1e-4)
    decoder_optimizer = optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=learning_rate, weight_decay=1e-4)

    if len(sys.argv) == 4:
        encoder_optimizer.load_state_dict(check_point["encoder_optimizer"])
        decoder_optimizer.load_state_dict(check_point["decoder_optimizer"])

        for state in encoder_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        for state in decoder_optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

    #=============================== training_data

    sentence_variables = []

    input1_variables = []
    input2_variables = []
    input3_variables = []

    gold1_variables = []
    gold2_variables = []
    gold3_variables = []

    mask1_variables = []
    mask2_variables = []
    mask3_variables = []

    for instance in trn_instances:
        #print len(sentence_variables)
        sentence_variables.append([])
        if use_cuda:
            sentence_variables[-1].append(Variable(instance[0]).cuda())
            sentence_variables[-1].append(Variable(instance[1]).cuda())
            sentence_variables[-1].append(Variable(instance[2]).cuda())
        else:
            sentence_variables[-1].append(Variable(instance[0]))
            sentence_variables[-1].append(Variable(instance[1]))
            sentence_variables[-1].append(Variable(instance[2]))

        if use_cuda:
            input1_variables.append(Variable(torch.LongTensor([0] + instance[3])).cuda())
            gold1_variables.append(Variable(torch.LongTensor(instance[3] + [1])).cuda())
        else:
            input1_variables.append(Variable(torch.LongTensor([0] + instance[3])))
            gold1_variables.append(Variable(torch.LongTensor(instance[3] + [1])))

        p = 0
        all_relations = []
        input2_variable = []
        for i in range(len(instance[3])):
            idx = instance[3][i]
            if idx == 5 or idx == 6:
                all_relations = all_relations + instance[4][p]
                all_relations.append([-2, 1])
                if len(instance[4][p]) == 0:
                    input2_variable.append(None)
                elif use_cuda:
                    input2_variable.append(Variable(torch.LongTensor([x[1] for x in instance[4][p]])).cuda())
                else:
                    input2_variable.append(Variable(torch.LongTensor([x[1] for x in instance[4][p]])))
                p += 1
        input2_variables.append(input2_variable)

        sel_gen_relations = []
        for type, idx in all_relations:
            if type == -2:
                sel_gen_relations.append(idx)
            else:
                sel_gen_relations.append(type+decoder.tags_info.tag_size)
        if use_cuda:
            gold2_variables.append(Variable(torch.LongTensor(sel_gen_relations)).cuda())
        else:
            gold2_variables.append(Variable(torch.LongTensor(sel_gen_relations)))
        assert p == len(instance[4])

        p = 0
        all_variables = []
        input3_variable = []
        for i in range(len(instance[5])):
            idx = instance[5][i]
            if (idx >= 13 and idx < decoder.tags_info.k_rel_start) or idx >= decoder.tags_info.tag_size:
                all_variables = all_variables + instance[6][p]
                all_variables.append(1)
                if use_cuda:
                    input3_variable.append(Variable(torch.LongTensor(instance[6][p])).cuda())
                else:
                    input3_variable.append(Variable(torch.LongTensor(instance[6][p])))
                p += 1
        input3_variables.append(input3_variable)
        if use_cuda:
            gold3_variables.append(Variable(torch.LongTensor(all_variables)).cuda())
        else:
            gold3_variables.append(Variable(torch.LongTensor(all_variables)))
        assert p == len(instance[6])


        ##### mask1
        p_max = 0
        decoder.outer_mask_pool.reset()
        mask1 = []
        mask1.append(decoder.outer_mask_pool.get_step_mask())
        for idx in instance[3]:
            assert mask1[-1][idx] == decoder.outer_mask_pool.need
            if idx >= decoder.tags_info.p_rel_start and idx < decoder.tags_info.k_tag_start:
                p_max += 1
            decoder.outer_mask_pool.update(-2, idx)
            mask1.append(decoder.outer_mask_pool.get_step_mask())

        ##### mask2
        decoder.rel_mask_pool.reset(len(instance[0]))
        mask2 = []
        p = 0
        for i in range(len(instance[3])):
            idx = instance[3][i]
            if idx == 5 or idx == 6:
                least = False
                if idx == 5 or (idx == 6 and instance[3][i+1] == 4):
                    least = True
                decoder.rel_mask_pool.set_sdrs(idx == 5)
                temp_mask = decoder.rel_mask_pool.get_all_mask(len(instance[4][p]), least)
                for k in range(len(instance[4][p])):
                    temp_idx = instance[4][p][k][0]
                    if temp_idx == -2:
                        temp_idx = instance[4][p][k][1]
                    else:
                        temp_idx += decoder.tags_info.tag_size
                    assert temp_mask[k][temp_idx] == decoder.rel_mask_pool.need
                mask2 = mask2 + temp_mask

                p += 1
        assert p == len(instance[4])

        #### mask3
        decoder.var_mask_pool.reset(p_max)
        mask3 = []
        p = 0
        for i in range(len(instance[5])):
            idx = instance[5][i]

            decoder.var_mask_pool.update(idx)
            if (idx >= 13 and idx < decoder.tags_info.k_rel_start) or idx >= decoder.tags_info.tag_size:
                for idxx in instance[6][p]:
                    #print idxx
                    mask3.append(decoder.var_mask_pool.get_step_mask())
                    decoder.var_mask_pool.update(idxx)
                    assert mask3[-1][idxx] == decoder.var_mask_pool.need
                    #decoder.mask_pool._print_state()
                mask3.append(decoder.var_mask_pool.get_step_mask())
                p += 1
        assert p == len(instance[6])

        if use_cuda:
            mask1_variables.append(Variable(torch.FloatTensor(mask1), requires_grad=False).cuda())
            mask2_variables.append(Variable(torch.FloatTensor(mask2), requires_grad=False).cuda())
            mask3_variables.append(Variable(torch.FloatTensor(mask3), requires_grad=False).cuda())
        else:
            mask1_variables.append(Variable(torch.FloatTensor(mask1), requires_grad=False))
            mask2_variables.append(Variable(torch.FloatTensor(mask2), requires_grad=False))
            mask3_variables.append(Variable(torch.FloatTensor(mask3), requires_grad=False))

#==================================
    dev_sentence_variables = []

    dev_input1_variables = []
    dev_input2_variables = []
    dev_input3_variables = []

    dev_gold1_variables = []
    dev_gold2_variables = []
    dev_gold3_variables = []

    dev_mask1_variables = []
    dev_mask2_variables = []
    dev_mask3_variables = []

    for instance in dev_instances:
        #print len(sentence_variables)
        dev_sentence_variables.append([])
        if use_cuda:
            dev_sentence_variables[-1].append(Variable(instance[0], volatile=True).cuda())
            dev_sentence_variables[-1].append(Variable(instance[1], volatile=True).cuda())
            dev_sentence_variables[-1].append(Variable(instance[2], volatile=True).cuda())
        else:
            dev_sentence_variables[-1].append(Variable(instance[0], volatile=True))
            dev_sentence_variables[-1].append(Variable(instance[1], volatile=True))
            dev_sentence_variables[-1].append(Variable(instance[2], volatile=True))

        if use_cuda:
            dev_input1_variables.append(Variable(torch.LongTensor([0] + instance[3]), volatile=True).cuda())
            dev_gold1_variables.append(Variable(torch.LongTensor(instance[3] + [1]), volatile=True).cuda())
        else:
            dev_input1_variables.append(Variable(torch.LongTensor([0] + instance[3]), volatile=True))
            dev_gold1_variables.append(Variable(torch.LongTensor(instance[3] + [1]), volatile=True))

        p = 0
        all_relations = []
        dev_input2_variable = []
        for i in range(len(instance[3])):
            idx = instance[3][i]
            if idx == 5 or idx == 6:
                all_relations = all_relations + instance[4][p]
                all_relations.append([-2, 1])
                if len(instance[4][p]) == 0:
                    dev_input2_variable.append(None)
                elif use_cuda:
                    dev_input2_variable.append(Variable(torch.LongTensor([x[1] for x in instance[4][p]]), volatile=True).cuda())
                else:
                    dev_input2_variable.append(Variable(torch.LongTensor([x[1] for x in instance[4][p]]), volatile=True))
                p += 1
        dev_input2_variables.append(dev_input2_variable)
        sel_gen_relations = []
        for type, idx in all_relations:
            if type == -2:
                sel_gen_relations.append(idx)
            else:
                sel_gen_relations.append(type+decoder.tags_info.tag_size)
        if use_cuda:
            dev_gold2_variables.append(Variable(torch.LongTensor(sel_gen_relations), volatile=True).cuda())
        else:
            dev_gold2_variables.append(Variable(torch.LongTensor(sel_gen_relations), volatile=True))
        assert p == len(instance[4])

        p = 0
        all_variables = []
        dev_input3_variable = []
        for i in range(len(instance[5])):
            idx = instance[5][i]
            if (idx >= 13 and idx < decoder.tags_info.k_rel_start) or idx >= decoder.tags_info.tag_size:
                all_variables = all_variables + instance[6][p]
                all_variables.append(1)
                if use_cuda:
                    dev_input3_variable.append(Variable(torch.LongTensor(instance[6][p]), volatile=True).cuda())
                else:
                    dev_input3_variable.append(Variable(torch.LongTensor(instance[6][p]), volatile=True))
                p += 1
        dev_input3_variables.append(dev_input3_variable)
        if use_cuda:
            dev_gold3_variables.append(Variable(torch.LongTensor(all_variables), volatile=True).cuda())
        else:
            dev_gold3_variables.append(Variable(torch.LongTensor(all_variables), volatile=True))
        assert p == len(instance[6])


        ##### mask1
        p_max = 0
        decoder.outer_mask_pool.reset()
        mask1 = []
        mask1.append(decoder.outer_mask_pool.get_step_mask())
        for idx in instance[3]:
            assert mask1[-1][idx] == decoder.outer_mask_pool.need
            if idx >= decoder.tags_info.p_rel_start and idx < decoder.tags_info.k_tag_start:
                p_max += 1
            decoder.outer_mask_pool.update(-2, idx)
            mask1.append(decoder.outer_mask_pool.get_step_mask())

        #### mask2
        decoder.rel_mask_pool.reset(len(instance[0]))
        mask2 = []
        p = 0
        for i in range(len(instance[3])):
            idx = instance[3][i]
            if idx == 5 or idx == 6:
                least = False
                if idx == 5 or (idx == 6 and instance[3][i+1] == 4):
                    least = True
                decoder.rel_mask_pool.set_sdrs(idx == 5)
                temp_mask = decoder.rel_mask_pool.get_all_mask(len(instance[4][p]), least)
                for k in range(len(instance[4][p])):
                    temp_idx = instance[4][p][k][0]
                    if temp_idx == -2:
                        temp_idx = instance[4][p][k][1]
                    else:
                        temp_idx += decoder.tags_info.tag_size
                    assert temp_mask[k][temp_idx] == decoder.rel_mask_pool.need
                mask2 = mask2 + temp_mask
                p += 1
        assert p == len(instance[4])

        #### mask3
        decoder.var_mask_pool.reset(p_max)
        mask3 = []
        p = 0
        for i in range(len(instance[5])):
            idx = instance[5][i]

            decoder.var_mask_pool.update(idx)
            if (idx >= 13 and idx < decoder.tags_info.k_rel_start) or idx >= decoder.tags_info.tag_size:
                for idxx in instance[6][p]:
                    #print idxx
                    mask3.append(decoder.var_mask_pool.get_step_mask())
                    decoder.var_mask_pool.update(idxx)
                    assert mask3[-1][idxx] == decoder.var_mask_pool.need
                    #decoder.mask_pool._print_state()
                mask3.append(decoder.var_mask_pool.get_step_mask())
                p += 1
        assert p == len(instance[6])

        if use_cuda:
            dev_mask1_variables.append(Variable(torch.FloatTensor(mask1), volatile=True).cuda())
            dev_mask2_variables.append(Variable(torch.FloatTensor(mask2), volatile=True).cuda())
            dev_mask3_variables.append(Variable(torch.FloatTensor(mask3), volatile=True).cuda())
        else:
            dev_mask1_variables.append(Variable(torch.FloatTensor(mask1), volatile=True))
            dev_mask2_variables.append(Variable(torch.FloatTensor(mask2), volatile=True))
            dev_mask3_variables.append(Variable(torch.FloatTensor(mask3), volatile=True))

#====================================== test
    tst_sentence_variables = []

    for instance in tst_instances:
        tst_sentence_variable = []
        if use_cuda:
            tst_sentence_variable.append(Variable(instance[0], volatile=True).cuda())
            tst_sentence_variable.append(Variable(instance[1], volatile=True).cuda())
            tst_sentence_variable.append(Variable(instance[2], volatile=True).cuda())
            
        else:
            tst_sentence_variable.append(Variable(instance[0], volatile=True))
            tst_sentence_variable.append(Variable(instance[1], volatile=True))
            tst_sentence_variable.append(Variable(instance[2], volatile=True))
            
        tst_sentence_variables.append(tst_sentence_variable)

#======================================
    idx = -1
    iter = 0
    if len(sys.argv) >= 4:
        iter = check_point["iter"]
        idx = check_point["idx"]

    while True:
        if use_cuda:
            torch.cuda.empty_cache()
        idx += 1
        iter += 1
        if idx == len(trn_instances):
            idx = 0       
        loss1, loss2, loss3 = train(sentence_variables[idx], (input1_variables[idx], input2_variables[idx], input3_variables[idx]), (gold1_variables[idx], gold2_variables[idx], gold3_variables[idx]), (mask1_variables[idx], mask2_variables[idx], mask3_variables[idx]), encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total1 += loss1
        print_loss_total2 += loss2
        print_loss_total3 += loss3
        print_loss_total += (loss1 + loss2 + loss3)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0

            print_loss_avg1 = print_loss_total1 / print_every
            print_loss_total1 = 0
            print_loss_avg2 = print_loss_total2 / print_every
            print_loss_total2 = 0
            print_loss_avg3 = print_loss_total3 / print_every
            print_loss_total3 = 0
            print('epoch %.6f : %.10f s1: %.10f s2: %.10f s3: %.10f' % (iter*1.0 / len(trn_instances), print_loss_avg, print_loss_avg1, print_loss_avg2, print_loss_avg3 ))

        if iter % evaluate_every == 0:
            dev_idx = 0
            dev_loss = 0.0
            dev_loss1 = 0.0
            dev_loss2 = 0.0
            dev_loss3 = 0.0
            torch.save({"iter": iter, "idx":idx,  "encoder":encoder.state_dict(), "decoder":decoder.state_dict(), "encoder_optimizer": encoder_optimizer.state_dict(), "decoder_optimizer": decoder_optimizer.state_dict()}, model_dir+str(int(iter/evaluate_every))+".model")
            while dev_idx < len(dev_instances):
                if use_cuda:
                    torch.cuda.empty_cache()
                a, b, c = train(dev_sentence_variables[dev_idx], (dev_input1_variables[dev_idx], dev_input2_variables[dev_idx], dev_input3_variables[dev_idx]), (dev_gold1_variables[dev_idx], dev_gold2_variables[dev_idx], dev_gold3_variables[dev_idx]), (dev_mask1_variables[dev_idx], dev_mask2_variables[dev_idx], dev_mask3_variables[dev_idx]), encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, back_prop=False)
                dev_loss1 += a
                dev_loss2 += b
                dev_loss3 += c
                dev_loss += (a+b+c)
                dev_idx += 1
            print('dev loss %.10f, s1: %.10f, s2: %.10f, s3: %.10f ' % (dev_loss/len(dev_instances), dev_loss1/len(dev_instances), dev_loss2/len(dev_instances), dev_loss3/len(dev_instances)))
            evaluate(dev_sentence_variables, encoder, decoder, dev_out_dir+str(int(iter/evaluate_every))+".drs")
            evaluate(tst_sentence_variables, encoder, decoder, tst_out_dir+str(int(iter/evaluate_every))+".drs")

def evaluate(sentence_variables, encoder, decoder, path):
    out = open(path,"w")
    for idx in range(len(sentence_variables)):
        if use_cuda:
            torch.cuda.empty_cache()
        
        structs, tokens = decode(sentence_variables[idx], encoder, decoder)

        p = 0
        output = []
        for i in range(len(structs)):
            if structs[i] < decoder.tags_info.tag_size:
                output.append(decoder.tags_info.ix_to_tag[structs[i]])
            else:
                output.append(decoder.tags_info.ix_to_lemma[structs[i] - decoder.tags_info.tag_size])
            if (structs[i] >= 13 and structs[i] < decoder.tags_info.k_rel_start) or structs[i] >= decoder.tags_info.tag_size:
                for idx in tokens[p]:
                    output.append(decoder.tags_info.ix_to_tag[idx])
                p += 1
        assert p == len(tokens)
        out.write(" ".join(output)+"\n")
        out.flush()
    out.close()

def test(dev_instances, tst_instances, encoder, decoder):
    #====================================== test
    dev_sentence_variables = []

    for instance in dev_instances:
        dev_sentence_variable = []
        if use_cuda:
            dev_sentence_variable.append(Variable(instance[0], volatile=True).cuda())
            dev_sentence_variable.append(Variable(instance[1], volatile=True).cuda())
            dev_sentence_variable.append(Variable(instance[2], volatile=True).cuda())
            
        else:
            dev_sentence_variable.append(Variable(instance[0], volatile=True))
            dev_sentence_variable.append(Variable(instance[1], volatile=True))
            dev_sentence_variable.append(Variable(instance[2], volatile=True))
            
        dev_sentence_variables.append(dev_sentence_variable)

    #====================================== test
    tst_sentence_variables = []

    for instance in tst_instances:
        tst_sentence_variable = []
        if use_cuda:
            tst_sentence_variable.append(Variable(instance[0], volatile=True).cuda())
            tst_sentence_variable.append(Variable(instance[1], volatile=True).cuda())
            tst_sentence_variable.append(Variable(instance[2], volatile=True).cuda())
            
        else:
            tst_sentence_variable.append(Variable(instance[0], volatile=True))
            tst_sentence_variable.append(Variable(instance[1], volatile=True))
            tst_sentence_variable.append(Variable(instance[2], volatile=True))
            
        tst_sentence_variables.append(tst_sentence_variable)

    check_point = {}
    if len(sys.argv) >= 4:
        check_point = torch.load(sys.argv[3])
        encoder.load_state_dict(check_point["encoder"])
        decoder.load_state_dict(check_point["decoder"])
        if use_cuda:
            encoder = encoder.cuda()
            decoder = decoder.cuda()
    evaluate(dev_sentence_variables, encoder, decoder, dev_out_dir+"dev.drs")
    evaluate(tst_sentence_variables, encoder, decoder, tst_out_dir+"tst.drs")

#####################################################################################
#####################################################################################
#####################################################################################
# main

from utils import readfile
from utils import data2instance
from utils import readpretrain
from tag import Tag
#from mask import Mask

trn_file = "data/train.input"
dev_file = "data/dev.input"
tst_file = "data/test.input"
pretrain_file = "data/sskip.100.vectors"
tag_info_file = "data/tag.info"
#trn_file = "train.input.part"
#dev_file = "dev.input.part"
#tst_file = "test.input.part"
#pretrain_file = "sskip.100.vectors.part"
UNK = "<UNK>"

trn_data = readfile(trn_file)
word_to_ix = {UNK:0}
lemma_to_ix = {UNK:0}
ix_to_lemma = [UNK]
ix_to_word = [UNK]
for sentence, _, lemmas, tags in trn_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
	    ix_to_word.append(word)
    for lemma in lemmas:
        if lemma not in lemma_to_ix:
            lemma_to_ix[lemma] = len(lemma_to_ix)
            ix_to_lemma.append(lemma)
#############################################
## tags
tags_info = Tag(tag_info_file, ix_to_lemma)
SOS = tags_info.SOS
EOS = tags_info.EOS
outer_mask_pool = OuterMask(tags_info)
rel_mask_pool = RelationMask(tags_info)
var_mask_pool = VariableMask(tags_info)
##############################################
##
#mask_info = Mask(tags)
#############################################
pretrain_to_ix = {UNK:0}
pretrain_embeddings = [ [0. for i in range(100)] ] # for UNK 
pretrain_data = readpretrain(pretrain_file)
for one in pretrain_data:
    pretrain_to_ix[one[0]] = len(pretrain_to_ix)
    pretrain_embeddings.append([float(a) for a in one[1:]])
print "pretrain dict size:", len(pretrain_to_ix)

dev_data = readfile(dev_file)
tst_data = readfile(tst_file)

print "word dict size: ", len(word_to_ix)
print "lemma dict size: ", len(lemma_to_ix)
print "global tag (w/o variables) dict size: ", tags_info.k_rel_start
print "global tag (w variables) dict size: ", tags_info.tag_size

WORD_EMBEDDING_DIM = 64
PRETRAIN_EMBEDDING_DIM = 100
LEMMA_EMBEDDING_DIM = 32
TAG_DIM = 128
INPUT_DIM = 100
ENCODER_HIDDEN_DIM = 256
DECODER_INPUT_DIM = 128
ATTENTION_HIDDEN_DIM = 256

out_word_ix = open("word.list", "w")
out_lemma_ix = open("lemma.list", "w")

for item in ix_to_word:
    out_word_ix.write(item+"\n")
out_word_ix.flush()
out_word_ix.close()
for item in ix_to_lemma:
    out_lemma_ix.write(item+"\n")
out_lemma_ix.flush()
out_lemma_ix.close()
encoder = EncoderRNN(len(word_to_ix), WORD_EMBEDDING_DIM, len(pretrain_to_ix), PRETRAIN_EMBEDDING_DIM, torch.FloatTensor(pretrain_embeddings), len(lemma_to_ix), LEMMA_EMBEDDING_DIM, INPUT_DIM, ENCODER_HIDDEN_DIM, n_layers=2, dropout_p=0.1)
attn_decoder = AttnDecoderRNN(outer_mask_pool, rel_mask_pool, var_mask_pool, tags_info, TAG_DIM, DECODER_INPUT_DIM, ENCODER_HIDDEN_DIM, ATTENTION_HIDDEN_DIM, n_layers=1, dropout_p=0.1)

###########################################################
# prepare training instance
trn_instances = data2instance(trn_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
print "trn size: " + str(len(trn_instances))
###########################################################
# prepare development instance
dev_instances = data2instance(dev_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
print "dev size: " + str(len(dev_instances))
###########################################################
# prepare test instance
tst_instances = data2instance(tst_data, [(word_to_ix,0), (pretrain_to_ix,0), (lemma_to_ix,0), tags_info])
print "tst size: " + str(len(tst_instances))

print "GPU", use_cuda
if use_cuda:
    encoder = encoder.cuda()
    attn_decoder = attn_decoder.cuda()

if len(sys.argv) == 5 and sys.argv[-1] == "test":
    test(dev_instances, tst_instances, encoder, attn_decoder)
else:
    trainIters(trn_instances, dev_instances, tst_instances, encoder, attn_decoder, print_every=1000, evaluate_every=50000, learning_rate=0.0005)

