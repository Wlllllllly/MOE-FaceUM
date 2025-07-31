# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import math
from typing import List
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from .parallel_experts import ParallelExperts

@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)
    index_sorted_experts = nonzeros[_index_sorted_experts]
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts



class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, head_size, num_experts, k,
                 cvloss=0.01, switchloss=0.1, zloss=1e-4,
                 bias=False, gating_activation=None,
                 activation=nn.ReLU(), noisy_gating=True, usage_mem = 10000,
                 acc_aux_loss=True):
        super(MoE, self).__init__()

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias)
        self.k = min(k, self.num_experts)
        self.cvloss = cvloss#0.01
        self.switchloss = switchloss#0.1
        self.zloss = zloss#1e-4
        self.activation = activation
        # self.usage = np.random.randint(num_experts, size=(usage_mem, k))
        # self.cur = 0

        self.acc_aux_loss = acc_aux_loss
        if self.acc_aux_loss:
            self.init_aux_statistics()

        if True:
            if gating_activation is None:
                gating_activation = nn.ReLU()
            self.f_gate = nn.Sequential(
                # nn.Linear(input_size, input_size),
                # gating_activation,
                nn.Linear(input_size,
                          2 * num_experts if noisy_gating else num_experts,
                          bias=False)
            )
            nn.init.zeros_(self.f_gate[-1].weight)
        else:
            self.f_gate = nn.Linear(input_size, num_experts, bias=False)
            nn.init.zeros_(self.f_gate.weight)


    def extra_repr(self):
        return 'k={}, cvloss={}, switchloss={}, zloss={}, noisy_gating={}'.format(
            self.k, self.cvloss, self.switchloss, self.zloss, self.noisy_gating)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def init_aux_statistics(self):
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.



    def update_aux_statistics(self, logits, probs, gates):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.000001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

    def get_aux_loss_and_clear(self):
        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        # cvloss = self.acc_gates.mean() / 10000.0
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)
        # loss = (self.cvloss * cvloss)
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss)


        self.init_aux_statistics()
        return loss



    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return (1.0 - loss.sum()) * self.num_experts


    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def top_k_gating(self, x, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.f_gate(x)
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        assert sample_topk == 0
        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

       # top_k_indecis: [batch, K]


        top_k_gates = top_k_gates

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)

        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        self.acc_aux_loss=False
        if self.acc_aux_loss:
            # if self.training:
            self.update_aux_statistics(logits, probs, gates)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        return loss

    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # y_ = self.forward_(x, skip_mask, sample_topk, multiply_by_gates)
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        # assert torch.allclose(y, y_)
        return y, loss

    def forward_(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # FOR DEBUGGING: naive forward
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        logits = self.f_gate(x)
        probs = torch.softmax(logits, dim=1)

        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)
        assert not self.bias
        hs = [torch.einsum('li,ij->lj', x, self.experts.w[i]) for i in range(self.num_experts)]
        hs = [self.activation(h) for h in hs]
        expert_outputs = [
            torch.einsum('li,ij->lj', hs[i], self.output_experts.w[i]) for i in range(self.num_experts)
        ]
        y = sum(probs[..., i][..., None] * expert_outputs[i] for i in range(self.num_experts))
        y = y.view(bsz, length, self.input_size)

        # if multiply_by_gates:
        #     expert_outputs = expert_outputs * self.batch_gates[:, None]
        return y

    def map(self, x, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class TaskMoE_Face_sixR(MoE):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self,  input_size, head_size, num_experts, k, w_MI=0, limit_k=0, w_topk_loss=0.0, task_num=2, noisy_gating=False, gating_activation=None, **kwargs):
        self.task_num = task_num
        self.w_topk_loss = w_topk_loss
        self.w_MI = w_MI

        self.limit_k = max(k, limit_k)

        super(TaskMoE_Face_sixR, self).__init__(input_size, head_size, num_experts, k, noisy_gating=noisy_gating, gating_activation=gating_activation, **kwargs)
        
        if gating_activation is None:
            gating_activation = nn.ReLU()

        self.shared_mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(input_size, input_size * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(input_size * 4, input_size))
        ]))

        self.tot_expert =[2 *num_experts if noisy_gating else num_experts for i in range(task_num)] 
        self.top_k_gates_task_1 = [torch.Size([])] * 6 #torch.zeros(task_num, self.num_experts).cuda()
        self.top_k_indices_task_1 = [torch.Size([])] * 6 #torch.zeros(task_num, self.num_experts).cuda()
        self.probs_bias_1 = [torch.Size([])] * 6
        self.top_k_gates_task_2 = [torch.Size([])] * 6 #torch.zeros(task_num, self.num_experts).cuda()
        self.top_k_indices_task_2= [torch.Size([])] * 6 #torch.zeros(task_num, self.num_experts).cuda()
        self.f_gate_1 = nn.ModuleList([nn.Sequential(
                                        # nn.Linear(input_size, input_size),
                                        # gating_activation,
                                        nn.Linear(input_size,
                                                  2 * num_experts if noisy_gating else num_experts,
                                                  bias=False)
                                    ) for i in range(task_num)])
        self.f_gate_2 = nn.ModuleList([nn.Sequential(
                                        # nn.Linear(input_size, input_size),
                                        # gating_activation,
                                        nn.Linear(input_size,
                                                  2 * num_experts if noisy_gating else num_experts,
                                                  bias=False)
                                    ) for i in range(task_num)])
        self.expert_biases_1= [nn.Parameter(torch.zeros(2 * num_experts if noisy_gating else num_experts).cuda()) for i in range(task_num)]
        self.expert_biases_2= [nn.Parameter(torch.zeros(2 * num_experts if noisy_gating else num_experts).cuda()) for i in range(task_num)]
        
        self.record_expert_usage = True  
        self.expert_usage_counter = torch.zeros(self.task_num, self.num_experts)  # shape = [T, E]

        for i in range(task_num):
            nn.init.zeros_(self.f_gate_1[i][-1].weight)
            nn.init.zeros_(self.f_gate_2[i][-1].weight)
    
    def init_aux_statistics(self, clear=True):
        """Initialize auxiliary statistics for tracking expert routing performance.
        
        Args:
            clear (bool): If True, resets task-specific gate frequencies and top-k accuracy.
        """
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

        if clear:
            self.task_gate_freq = [0] * self.task_num
            self.topk_acc_probs = 0.

        self.MI_task_gate = torch.zeros(self.task_num, self.num_experts).cuda()

    def update_aux_statistics(self, logits, probs, gates, task_bh):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.0001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

        self.topk_acc_probs = self.topk_acc_probs + probs.mean(0)

        self.task_gate_freq[task_bh] = self.task_gate_freq[task_bh]*0.95 + ((gates > 0).float().sum(0)).detach()*0.05

        self.MI_task_gate[task_bh] = self.MI_task_gate[task_bh] + probs.sum(0)

    def get_topk_loss_and_clear(self):
        top_k_probs, top_k_indices = self.topk_acc_probs.topk(self.limit_k, dim=0)
        zeros = torch.zeros_like(self.topk_acc_probs)
        gates = zeros.scatter(0, top_k_indices, top_k_probs)
        topk_loss = ((self.topk_acc_probs - gates) * (self.topk_acc_probs - gates)).sum()

        self.topk_acc_probs = 0.
        return topk_loss * self.w_topk_loss # 0.004 * 12 * 2 = 0.09

    def get_aux_loss_and_clear(self):
        '''
            acc_gates: sum of topk soft score
            acc_freq: the number of being chosen
            acc_probs: sum of probs (probs = softmax(score))
        '''

        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)

        tot = self.acc_freq.sum() / self.k
        self.MI_task_gate = self.MI_task_gate / (tot+0.0001)
        P_TI = torch.sum(self.MI_task_gate, dim=1, keepdim=True) + 0.0001
        P_EI = torch.sum(self.MI_task_gate, dim=0, keepdim=True) + 0.0001

        MI_loss = -(self.MI_task_gate * torch.log(self.MI_task_gate / P_TI / P_EI + 0.0001)).sum()
        # print(f"MI_loss:{MI_loss}")
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss +
                self.w_MI * MI_loss
                )
        # print(f"loss:{loss}")

        self.init_aux_statistics(clear=False)
        return loss


    def top_k_gating_roi2(self, x, task_bh, skip_mask=None, sample_topk=0, noise_epsilon=1e-2, is_roi=None, treshold=0.1):
        
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        if is_roi:
            roi_index=0
            clean_logits = self.f_gate_1[task_bh](x)
            clean_logits_bias = self.f_gate_1[task_bh](x)+ self.expert_biases_1[task_bh]
        else:
            roi_index=1
            clean_logits = self.f_gate_2[task_bh](x)
            clean_logits_bias = self.f_gate_2[task_bh](x)+ self.expert_biases_2[task_bh]
        
        # if self.noisy_gating and self.training:
        # # if self.noisy_gating:
        #     # clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
        #     # noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
        #     # eps = torch.randn_like(clean_logits)
        #     # noisy_logits = clean_logits + eps * noise_stddev
        #     # logits = noisy_logits
        #     logits = clean_logits
        #     logits_bias = clean_logits_bias
        # elif self.noisy_gating:
        #     logits, _ = clean_logits.chunk(2, dim=-1)
        # else:
        logits = clean_logits
        logits_bias = clean_logits_bias


        probs_bias = torch.softmax(logits_bias, dim=1) + 1e-4
        if skip_mask is not None:
            probs_bias = torch.masked_fill(probs_bias, skip_mask, 0)

        probs = torch.softmax(logits, dim=1) + 1e-4
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)



        # if self.training and (sample_topk > 0):
        #     # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
        #     # top_k_gates = torch.gather(probs, 1, top_k_indices)
        #     assert sample_topk <= self.k

        #     _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
        #     # ic(top_km1_indices.shape)
        #     masked_probs = probs + 1e-6
        #     masked_probs[torch.arange(probs.size(0)).unsqueeze(
        #         1), top_km1_indices] = 0
        #     k_indices = torch.multinomial(masked_probs, sample_topk)
        #     top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
        #     top_k_gates = torch.gather(probs, 1, top_k_indices)
        # else:
        if is_roi:
            selected_token = probs_bias > treshold
            selected_token_sum = selected_token.sum(1, keepdim=True)
            selected_token_mean = np.mean(selected_token_sum.detach().cpu().numpy())
            # max_selected = selected_token_sum.max().item()
            self.k = min(int(round(selected_token_mean))+2,6) #max(2,int(round(selected_token_mean)))
            # print(f"probs_bias :{probs_bias}")
        else:
            self.k = 2


        top_k_gates, top_k_indices = probs_bias.topk(self.k, dim=1)

        # top_k_indecis: [batch, K]  
       
        if is_roi:
            self.top_k_gates_task_1[task_bh] = top_k_gates
            self.top_k_indices_task_1[task_bh]  = top_k_indices
            self.probs_bias_1[task_bh] = clean_logits
        else:
            self.top_k_gates_task_2[task_bh] = top_k_gates
            self.top_k_indices_task_2[task_bh]  = top_k_indices

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        # self.acc_aux_loss=True
        # raise Exception(f"self.acc_aux_loss:{self.acc_aux_loss}")
        # if self.acc_aux_loss:
        #     self.update_aux_statistics(logits, probs, gates, task_bh)
        # else:
        #     loss += self.cvloss * self.compute_cvloss(gates)
        #     loss += self.switchloss * \
        #         self.compute_switchloss(probs, self.expert_size)
        #     loss += self.zloss * self.compute_zloss(logits)

        #if not self.training and self.record_expert_usage:
        # top_k_indices: shape = [B, k]
            #with torch.no_grad():
                #expert_counts = torch.bincount(top_k_indices.view(-1), minlength=self.task_num)
                #self.expert_usage_counter[task_bh] += expert_counts.to(self.expert_usage_counter.device)

        return loss


    def forward(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True,patch_indices_all = None, split_point_all = None):

        length,bsz,emb_size = x.shape
        patch_tokens = x[1:, :, :]  # (B, C, H*W)
        cls_token = x[0, :, :].unsqueeze(0)  # (1, C, H*W)


        def create_masked_input(input_tensor, indices_all, keep_roi=True):
            masked = input_tensor.clone()
            for i in range(bsz):
                mask = torch.ones_like(input_tensor[:,i,:])
                # print("mask shape:", mask.shape)

                if keep_roi:
                    # keep ROI, so mask non-ROI
                    indices = indices_all[i][split_point_all[i]:]
                else:
                    # keep non-ROI, so mask ROI
                    indices = indices_all[i][:split_point_all[i]]
                # print("first index example:", indices[0])
                if len(indices) > 0:
                    indices_tensor = torch.tensor(indices, dtype=torch.long, device=mask.device)
                    mask[indices_tensor, :] = 0
                masked[:,i,:] = input_tensor[:,i,:] * mask
            return masked.cuda()

        roi_inp = create_masked_input(patch_tokens, patch_indices_all, keep_roi=True).permute(1, 0, 2)
        nonroi_inp = create_masked_input(patch_tokens, patch_indices_all, keep_roi=False).permute(1, 0, 2)

        def get_roi_MOE(input_tensor,skip_mask=None,multiply_by_gates=None, is_roi=None):
            if is_roi:
                self.k = 4
            else:
                self.k = 2
            
            input_tensor = input_tensor.reshape(-1, emb_size)
            if skip_mask is not None:
                skip_mask = skip_mask.view(-1, 1)

            loss = self.top_k_gating_roi2(input_tensor, task_bh, skip_mask,  sample_topk=sample_topk, is_roi=is_roi)

            expert_inputs = input_tensor[self.batch_index]
            h = self.experts(expert_inputs, self.expert_size)
            h = self.activation(h)
            expert_outputs = self.output_experts(h, self.expert_size)

            if multiply_by_gates:
                expert_outputs = expert_outputs * self.batch_gates[:, None]
            else:
                expert_outputs = expert_outputs

            zeros = torch.zeros((bsz * (length-1), self.input_size), 
                dtype=expert_outputs.dtype, device=expert_outputs.device)
            input_tensor  = zeros.index_add(0, self.batch_index, expert_outputs)

            input_tensor = input_tensor.view( bsz, length-1,  self.input_size)

            return input_tensor

        core_out_roi = get_roi_MOE(roi_inp,skip_mask=skip_mask, multiply_by_gates=multiply_by_gates, is_roi=True).permute(1, 0, 2)
        core_out_nonroi = get_roi_MOE(nonroi_inp,skip_mask=skip_mask, multiply_by_gates=multiply_by_gates, is_roi=False).permute(1, 0, 2)

        full_output = torch.zeros_like(patch_tokens)
        for i in range(bsz):
            roi_indices = patch_indices_all[i][:split_point_all[i]]
            non_roi_indices = patch_indices_all[i][split_point_all[i]:]
            if len(roi_indices) > 0:
                ri = torch.tensor(roi_indices, dtype=torch.long, device=full_output.device)
             
                full_output[:,i,:][ri] = core_out_roi[:,i,:][ri]
            if len(non_roi_indices) > 0:
                nri = torch.tensor(non_roi_indices, dtype=torch.long, device=full_output.device)
                full_output[:,i,:][nri] = core_out_nonroi[:,i,:][nri]
        
        cls_output = self.shared_mlp(cls_token)
        y = torch.cat([cls_output, full_output], dim=0)

        return y

    def map(self, x, task_bh, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)


        zeros = torch.zeros((bsz * length * self.k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)

        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y