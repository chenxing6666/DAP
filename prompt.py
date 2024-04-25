import torch
import torch.nn as nn

class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',tasklength=10):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.tasklength = tasklength 
        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            generalpromt = (top_k, length, embed_dim)

            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                self.taskprompt = nn.ParameterList([nn.Parameter(torch.zeros(top_k, length, embed_dim)) for _ in range(tasklength)]) # this is for taskid
                self.generalprompt = nn.Parameter(torch.zeros(generalpromt)) 

            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
                self.taskprompt = nn.ParameterList([nn.Parameter(torch.zeros(top_k, length, embed_dim)) for _ in range(tasklength)]) # this is for taskid
                for tp in self.taskprompt:
                    nn.init.uniform_(tp, -1, 1)
                self.generalprompt = nn.Parameter(torch.randn(generalpromt))
                nn.init.uniform_(self.generalprompt, -1, 1)

        if prompt_key:

            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:

            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None,taskid=None):

        out = dict()

        top_k, length, c = self.taskprompt[taskid].shape
        batched_task_prompt_raw = self.taskprompt[taskid].reshape(top_k * length, c)
        batched_task_prompt = batched_task_prompt_raw.unsqueeze(0).expand(x_embed.shape[0], -1, -1)

        batched_general_prompt_raw = self.generalprompt.reshape(top_k * length, c)
        batched_general_prompt = batched_general_prompt_raw.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        

        out['total_prompt_len'] = batched_task_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_task_prompt, x_embed], dim=1)

        out['gen_total_prompt_len'] = batched_general_prompt.shape[1]
        out['gen_prompted_embedding'] = torch.cat([batched_general_prompt, x_embed], dim=1)
        return out
