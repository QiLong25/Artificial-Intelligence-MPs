'''
This is one of the modules you'll submit to the autograder. The functions here, combined, 
implements the multi-head attention mechanisms of the Transformer encoder and decoder layers
'''

'''
Note:
Please do not modify any variable name given to you for code completion, especially those that have trainable parameters in torch
'''


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class MultiHeadAttention(nn.Module):
    '''
    We implement the multi-head attention mechanism, as a torch.nn.Module. In the __init__ function, we define some trainable parameters and hyperparameters - do not modify the variable names of the trainable parameters!
    The forward function has been completed for you, but you need to complete  compute_scaled_dot_product_attention and compute_mh_qkv_transformation.
    '''
    ## The __init__ function is given; you SHOULD NOT modify it
    def __init__(self, d_model, num_heads):
        '''
        Initialize a multihead attention module
        d_model (int) - dimension of the multi-head attention module, which is the dimension of the input Q, K and V before and after linear transformation;

        num_heads (int) - number of attention heads
        '''
        
        super(MultiHeadAttention, self).__init__()
        
        # Set model dimension, attention head count, and attention dimension
        self.d_model = d_model
        self.num_heads = num_heads 
        self.d_k = d_model // num_heads

        assert (self.d_k * self.num_heads) == self.d_model
        
        # Query, key, value and output linear transformation matrices; note that d_model = d_k * num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    ## You need to implement the missing lines in compute_scaled_dot_product_attention below
    def compute_scaled_dot_product_attention(self, query, key, value, key_padding_mask = None, attention_mask = None):
        '''
        This function calculates softmax(Q K^T / sqrt(d_k))V for the attention heads; further, a key_padding_mask is given so that padded regions are not attended, and an attention_mask is provided so that we can disallow attention for some part of the sequence
        Input:
        query (torch.Tensor) - Query; torch tensor of size B x num_heads x T_q x d_k, where B is the batch size, T_q is the number of time steps of the query (aka the target sequence), num_head is the number of attention heads, and d_k is the feature dimension;

        key (torch.Tensor) - Key; torch tensor of size B x num_head x T_k x d_k, where in addition, T_k is the number of time steps of the key (aka the source sequence);

        value (torch.Tensor) - Value; torch tensor of size B x num_head x T_v x d_k; where in addition, T_v is the number of time steps of the value (aka the source sequence);, and we assume d_v = d_k
        Note: We assume T_k = T_v as the key and value come from the same source in the Transformer implementation, in both the encoder and the decoder.

        key_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_k, where for each key_padding_mask[b] for the b-th source in the batch, the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

        attention_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size 1 x T_q x T_k or B x T_q x T_k, where again, T_q is the length of the target sequence, and T_k is the length of the source sequence. An example of the attention_mask is used for decoder self-attention to enforce auto-regressive property during parallel training; suppose the maximum length of a batch is 5, then the attention_mask for any input in the batch will look like this for each input of the batch.
        0 1 1 1 1
        0 0 1 1 1
        0 0 0 1 1
        0 0 0 0 1
        0 0 0 0 0
        As the key_padding_mask, the non-zero positions will be ignored and disallowed for attention while the zero positions will be allowed for attention.

        
        Output:
        x (torch.Tensor) - torch tensor of size B x T_q x d_model, which is the attended output

        '''
        min_val = torch.finfo(query.dtype).min
        ##### YOUR CODE STARTS HERE #####
        ## Use min_val defined above if you want to fill in certain parts of a tensor with the mininum value of a specific data type; use the torch.Tensor.masked_fill function in PyTorch to fill in a bool type mask; You will likely need to use torch.softmax, torch.matmul, torch.Tensor.transpose in your implementation as well, so make sure you look up the definitions in the PyTorch documentation (via a Google Search); also, note that broadcasting in PyTorch works similarly to the behavior in numpy

        B = query.shape[0]              # batch size
        T_q = query.shape[2]
        T_k = key.shape[2]

        # ## multi-head qkv
        # query_use = torch.transpose(query, 1, 2)                # (B, T_q, num_heads, d_k)
        # query_use = query_use.contiguous().view(B, T_q, self.num_heads * self.d_k)              # (B, T_q, d_model)
        # key_use = torch.transpose(key, 1, 2)                    # (B, T_k, num_heads, d_k)
        # key_use = key_use.contiguous().view(B, T_k, self.num_heads * self.d_k)                  # (B, T_k, d_model)
        # value_use = torch.transpose(value, 1, 2)                # (B, T_v, num_heads, d_k)
        # value_use = value_use.contiguous().view(B, T_k, self.num_heads * self.d_k)              # (B, T_v, d_model)
        # q, k, v = self.compute_mh_qkv_transformation(query_use, key_use, value_use)             # (B, num_heads, T, d_k)

        ## before softmax
        k_use = torch.transpose(key, 2, 3)                    # (B, num_heads, d_k, T_k)
        S = torch.matmul(query, k_use) / (self.d_k ** 0.5)                  # S = qk / sqrt(dk), (B, num_heads, T_q, T_k)

        ## key_padding_mask
        if key_padding_mask is not None:
            key_padding_mask_use = (key_padding_mask != 0).view(B, 1, 1, -1)                    # (B, 1, 1, T_k), non-zero entry = True
            S = S.masked_fill(key_padding_mask_use, min_val)                        # non-zero entry = True -> set to -inf

        ## attention_mask
        if attention_mask is not None:
            if attention_mask.shape[0] == 1:
                attention_mask_use = (attention_mask == 1).view(1, 1, T_q, T_k)             # (1, 1, T_q, T_k), one entry = True
                S = S.masked_fill(attention_mask_use, min_val)              # one entry = True -> set to -inf
            else:
                attention_mask_use = (attention_mask == 1).view(B, 1, T_q, T_k)             # (B, 1, T_q, T_k), one entry = True
                S = S.masked_fill(attention_mask_use, min_val)              # one entry = True -> set to -inf

        ## softmax
        softmax = nn.Softmax(dim=3)             # unify over T_k
        att = softmax(S)

        ## cross product with value
        x = torch.matmul(att, value)                # (B, num_heads, T_q, d_k)
        x = torch.transpose(x, 1, 2)            # (B, T_q, num_heads, d_k)

        ##### YOUR CODE ENDS HERE #####

        x = x.contiguous().view(B, -1, self.num_heads * self.d_k)       # (B, T_q, d_model)

        return x
    
    ## You need to implement the missing lines in compute_mh_qkv_transformation below
    def compute_mh_qkv_transformation(self, Q, K, V):
        """Transform query, key and value using W_q, W_k, W_v and split 

        Input:
            Q (torch.Tensor) - Query tensor of size B x T_q x d_model.
            K (torch.Tensor) - Key tensor of size B x T_k x d_model.
            V (torch.Tensor) - Value tensor of size B x T_v x d_model. Note that T_k = T_v.

        Output:
            q (torch.Tensor) - Transformed query tensor B x num_heads x T_q x d_k.
            k (torch.Tensor) - Transformed key tensor B x num_heads x T_k x d_k.
            v (torch.Tensor) - Transformed value tensor B x num_heads x T_v x d_k. Note that T_k = T_v
            Note that d_k * num_heads = d_model

        """
        ##### YOUR CODE STARTS HERE #####

        q = self.W_q(Q)             # (B, T_q, d_model)
        k = self.W_k(K)             # (B, T_k, d_model)
        v = self.W_v(V)             # (B, T_v, d_model)

        q = q.contiguous().view(Q.shape[0], Q.shape[1], self.num_heads, self.d_k)           # (B, T_q, num_heads, d_k)
        k = k.contiguous().view(K.shape[0], K.shape[1], self.num_heads, self.d_k)           # (B, T_k, num_heads, d_k)
        v = v.contiguous().view(V.shape[0], V.shape[1], self.num_heads, self.d_k)           # (B, T_v, num_heads, d_k)

        q = torch.transpose(q, 1, 2)                # (B, num_heads, T_q, d_k)
        k = torch.transpose(k, 1, 2)                # (B, num_heads, T_k, d_k)
        v = torch.transpose(v, 1, 2)                # (B, num_heads, T_v, d_k)

        ##### YOUR CODE ENDS HERE #####

        return q, k, v
    
    ## The below function is given; you DO NOT need to modify it
    def forward(self, query, key, value, key_padding_mask = None, attention_mask = None):
        """Compute scaled dot product attention.

        Args:
            Q (torch.Tensor) - Query tensor of size B x T_q x d_model.
            K (torch.Tensor) - Key tensor of size B x T_k x d_model.
            V (torch.Tensor) - Value tensor of size B x T_v x d_model. Note that T_k = T_v.

            key_padding_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size B x T_k, where for each key_padding_mask[b] for the b-th source in the batch, the non-zero positions will be ignored as they represent the padded region during batchify operation in the dataloader (i.e., disallowed for attention) while the zero positions will be allowed for attention as they are within the length of the original sequence

            attention_mask (None/torch.Tensor) - If it is not None, then it is a torch.IntTensor/torch.LongTensor of size 1 x T_q x T_k or B x T_q x T_k,where again, T_q is the length of the target sequence, and T_k is the length of the source sequence. An example of the attention_mask is used for decoder self-attention to enforce auto-regressive property during parallel training; suppose the maximum length of a batch is 5, then the attention_mask for any input in the batch will look like this for each input of the batch.
            0 1 1 1 1
            0 0 1 1 1
            0 0 0 1 1
            0 0 0 0 1
            0 0 0 0 0
            As the key_padding_mask, the non-zero positions will be ignored and disallowed for attention while the zero positions will be allowed for attention.


        Output:
            torch.Tenso - Output tensor of size B x T_q x d_model.

        """
        q, k, v = self.compute_mh_qkv_transformation(query, key, value)
        return self.W_o(self.compute_scaled_dot_product_attention(q, k, v, key_padding_mask = key_padding_mask, attention_mask = attention_mask))


