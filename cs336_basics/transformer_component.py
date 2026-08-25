import torch
from einops import einsum, rearrange
import einx
from jaxtyping import Bool, Float, Int
from torch import Tensor
from math import sqrt

class transformer_lm(torch.nn.Module):
    def __init__(
            self,
            vocab_size : int,
            context_length : int,
            num_layers : int,
            d_model: int,
            d_ff: int,
            rope_theta: float,
            num_heads: int,
            weights: dict[str, Tensor] | None = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.embedding_layer = Embedding(self.vocab_size, d_model)
        self.rms_norm_last = Rms_Norm(d_model)
        self.transformer_block_list = torch.nn.ModuleList()
        self.output_linear_layer = Linear(d_model, self.vocab_size)
        if weights is not None:
            self.embedding_layer.weight.data = weights.get("token_embeddings.weight")
            for i in range(self.num_layers):
                block_i_weights = self._generate_weights(weights, i)
                transformer_block_i = transformer_block(d_model, num_heads, d_ff, rope_theta, context_length, block_i_weights)
                self.transformer_block_list.append(transformer_block_i)
            self.rms_norm_last.weight.data = weights.get("ln_final.weight")     
            self.output_linear_layer.weight.data = weights.get("lm_head.weight")
        else :
            for i in range(self.num_layers):
                transformer_block_i = transformer_block(d_model, num_heads, d_ff, rope_theta, context_length)
                self.transformer_block_list.append(transformer_block_i)
            

    def forward(
            self,
            in_indices: Int[Tensor, "batch_size seq_len"]
    ) -> Float[Tensor,"batch_size seq_len vocab_size"]:
        x = self.embedding_layer.forward(in_indices) # in_features initialization
        for i in range(self.num_layers):
            x = self.transformer_block_list[i].forward(x)
        x = self.rms_norm_last.forward(x)
        x = self.output_linear_layer.forward(x)
        return x

    def _generate_weights(
            self,
            weights: dict[str, Tensor],
            layer_index) -> dict[str, Tensor]:
        block_weights = {k.replace(f"layers.{layer_index}.", ""): v for k, v in weights.items() if f"layers.{layer_index}." in k}
        return block_weights
    
class transformer_block(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float,
            max_seq_len: int,
            weights: dict[str, Tensor] | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.weights = weights

        self.rms_norm_layer1 = Rms_Norm(d_model)
        
        self.rms_norm_layer1_cal = self.rms_norm_layer1.forward

        self.rms_norm_layer2 = Rms_Norm(d_model)
        
        self.rms_norm_layer2_cal = self.rms_norm_layer2.forward

        if weights is not None:
            self.rms_norm_layer1.weight.data = self.weights["ln1.weight"]
            self.rms_norm_layer2.weight.data = self.weights["ln2.weight"]
            self.ffn = FFN_SwiGLU(d_model, d_ff, 
                                  self.weights["ffn.w1.weight"],
                                  self.weights["ffn.w2.weight"],
                                  self.weights["ffn.w3.weight"])
            self.mha = MutiHeadAttention(self.d_model, self.num_heads,
                                         max_seq_len,
                                         self.weights["attn.q_proj.weight"],
                                         self.weights["attn.k_proj.weight"],
                                         self.weights["attn.v_proj.weight"],
                                         self.weights["attn.output_proj.weight"],
                                         self.theta)
        else:
            self.ffn = FFN_SwiGLU(d_model, d_ff)
            self.mha = MutiHeadAttention(self.d_model, self.num_heads, max_seq_len, theta = self.theta)
        
        self.ffn_cal = self.ffn.forward

        

    def forward(
            self,
            x: Float[Tensor, "... seq_len d_model"]
    ):
        mha_res = x + self._MHA_layer(x)
        res = mha_res + self._FFN_layer(mha_res)
        return res

    def _FFN_layer(
            self, 
            x: Float[Tensor, "... seq_len d_model"]
    )  -> Tensor:
        x = self.rms_norm_layer2_cal(x)
        return self.ffn_cal(x)

    def _MHA_layer(
            self,
            x : Float[Tensor, "... seq_len d_model"]
    ) -> Tensor:
        x = self.rms_norm_layer1_cal(x)
        return self.mha.forward(x, torch.arange(x.shape[-2]), True)


class Linear(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int, 
            device : torch.device | None = None, 
            dtype : torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, device=self.device, dtype=self.dtype))
        self.reset_parameter(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum( x, self.weight,"... d_in, d_out d_in -> ... d_out")

    def reset_parameter(
            self,
            x : torch.Tensor, 
            expectation: torch.Tensor | None = None
            ) -> torch.Tensor:
        if expectation == None:
            delta = sqrt(2 / (self.in_features + self.out_features))
            torch.nn.init.trunc_normal_(x, mean = 0.0, std = delta, a = -3.0 * delta, b = 3.0 * delta)
        else : 
            x.data.copy_(expectation.to(device=self.device, dtype= self.dtype))

class Embedding(torch.nn.Module):
    def __init__(
            self,
            num_embeddings : int,
            embedding_dim : int,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.randn(num_embeddings, embedding_dim, dtype = self.dtype, device = self.device))
        torch.nn.init.trunc_normal_(self.weight, mean = 0, std = 1, a = -3, b = 3)

    def forward(
            self,
            token_ids: torch.Tensor
    ) -> torch.Tensor:
       return self.weight[token_ids] 

class Rms_Norm(torch.nn.Module):
    def __init__(
            self,
            d_model: int, 
            eps : float = 1e-5,
            device : torch.device | None= None,
            dtype : torch.dtype | None = None
    ): 
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = torch.nn.Parameter(torch.ones(d_model, dtype = self.dtype, device = self.device))
    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        # assert x.shape is [... self.d_model], "Incorrect Tensor Shape"
        in_type = x.dtype
        x = x.to(torch.float32)
        
        rms_ratio = torch.sqrt(einx.mean("... [d]", x.pow(2)) + self.eps)
        rms_ratio = rms_ratio.unsqueeze(-1)
        # print(f"the shape of x:{x.shape}\nthe shape of self weight:{self.weight.shape}\n")
        results =  x / rms_ratio * self.weight
        return results.to(in_type)

class FFN_SwiGLU(torch.nn.Module):
    def __init__(
            self, 
        d_model : int,
        d_ff : int,
        w1_weight: Float[Tensor, "d_ff d_model"] | None = None,
        w2_weight: Float[Tensor, "d_model d_ff"] | None = None,
        w3_weight: Float[Tensor, "d_ff d_model"] | None = None, 
        device_w1 : torch.device | None = None,
        device_w2 : torch.device | None = None,
        device_w3 : torch.device | None = None,
        dtype_w1 : torch.dtype | None = None,
        dtype_w2 : torch.dtype | None = None,
        dtype_w3 : torch.dtype | None = None
        ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(self.d_model, self.d_ff, device_w1, dtype_w1)
        self.w2 = Linear(self.d_ff, self.d_model, device_w2, dtype_w2)
        self.w3 = Linear(self.d_model, self.d_ff, device_w3, dtype_w3)

        if w1_weight is not None:
            self.w1.weight.data = w1_weight
        if w2_weight is not None:
            self.w2.weight.data = w2_weight
        if w3_weight is not None:
            self.w3.weight.data = w3_weight
    def forward(
            self,
            x: torch.Tensor
            ) -> torch.Tensor:
        w1_x = self.w1.forward(x)
        silu_w1_x = SiLU(w1_x)
        w3_x = self.w3.forward(x)
        return self.w2.forward(silu_w1_x * w3_x)

def SiLU(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
    return x * torch.sigmoid(x)

class RoPE(torch.nn.Module):
    def __init__(
            self,
            theta: float,
            d_k : int,
            max_seq_len: int,
            device: torch.device | None = None
    ):
        super().__init__()
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device
        self.dim = d_k
        token_pos = torch.arange(max_seq_len, device = device).unsqueeze(-1) # (max_seq_len, 1)
        embedding_component = torch.arange(d_k // 2, device = device).unsqueeze(0) # (1, d_k // 2)
        freq = token_pos * (theta ** ( -2 * embedding_component / d_k )) # broadcast -> (max_seq_len, d_k // 2)
        self.register_buffer("cos_cached", torch.cos(freq))
        self.register_buffer("sin_cached", torch.sin(freq))
        
    def forward(
            self,
            x : Float[torch.Tensor,"... seq_len d_k"],
            token_positions: torch.Tensor
            ) -> torch.Tensor:
        x_reshape = rearrange(x, "... (d_k two) -> ... d_k two", two = 2)
        x_rotate_half = self._reverse(x_reshape)
        cos = self.cos_cached[token_positions].unsqueeze(-1)
        sin = self.sin_cached[token_positions].unsqueeze(-1)
        return rearrange((cos * x_reshape + sin * x_rotate_half), "... d_k two -> ... (d_k two)")

    def _reverse(
            self,
            x : Float[Tensor,"... half_d 2"] 
    ) -> Float[Tensor, "... half_d 2"]:
        x1 = x[..., 0]
        x2 = x[..., 1]
        return torch.stack([-1 * x2, x1], dim = -1)
        
def softmax(
        x: torch.Tensor,
        i: int
):
    max_vals = torch.amax(x, dim = i, keepdim=True)
    x_stable = x - max_vals
    return torch.exp(x_stable) / torch.exp(x_stable).sum(dim = i, keepdim=True)

def attention(
        query : Float[Tensor, "batch_size ... seq_len_n d_k"],
        key : Float[Tensor, "batch_size ... seq_len_m d_k"],
        value : Float[Tensor, "batch_size ... seq_len_m d_v"],
        mask : Bool[Tensor, "seq_len_n seq_len_m"] | None = None
) -> Float[Tensor, "batch_size ... seq_len_n d_v"]:
    dot_results = einsum(query, key,"... seq_len_n d_k, ... seq_len_m d_k -> ... seq_len_n seq_len_m") / sqrt(query.shape[-1])
    if mask is not None:
        delta = torch.where(mask, 0.0, float("-inf"))
        dot_results += delta
    return einsum(softmax(dot_results, -1), value, " ... seq_len_n seq_len_m, ... seq_len_m d_v -> ... seq_len_n d_v")

class MutiHeadAttention(torch.nn.Module):
    def __init__(
            self,
            d_models: int,
            num_heads: int,
            max_seq_len: int | None = None,
            q_proj_weight: Float[Tensor, " d_model d_model"] | None = None,
            k_proj_weight: Float[Tensor, " d_model d_model"] | None = None,
            v_proj_weight: Float[Tensor, " d_model d_model"] | None = None,
            o_proj_weight: Float[Tensor, " d_model d_model"] | None = None,
            theta: int | None = None
    ) : 
        super().__init__()

        self.d_models = d_models
        self.num_heads = num_heads
        self.d_k = d_models // num_heads

        self.register_buffer("mask", None, persistent=False)

        self.cal_q = Linear(d_models, d_models)
        if q_proj_weight is not None:
            self.cal_q.weight.data = q_proj_weight
        self.cal_k = Linear(d_models, d_models)
        if k_proj_weight is not None:
            self.cal_k.weight.data = k_proj_weight
        self.cal_v = Linear(d_models, d_models)
        if v_proj_weight is not None:
            self.cal_v.weight.data = v_proj_weight
        self.cal_o = Linear(d_models, d_models)
        if o_proj_weight is not None:
            self.cal_o.weight.data = o_proj_weight
        self.theta = theta
        if max_seq_len is not None:
            self.rope_cal = RoPE(self.theta, self.d_k, max_seq_len)
    def forward(
            self,
            x : Float[Tensor, "batch_size ... seq_len d_model"],
            token_positions: Int[Tensor, "batch_size ... seq_len"] | None = None,
            is_rope = False,
    ) -> Float[Tensor, "batch_size ... seq_len_n d_v"] :

        query = self.cal_q.forward(x)
        key = self.cal_k.forward(x)
        value = self.cal_v.forward(x)

        query_multihead = self._devide(query)
        key_multihead = self._devide(key)
        value_multihead = self._devide(value)

        if self.mask is None or self.mask.shape[-1] != x.shape[-2]:
            mask_cond = torch.ones((x.shape[-2], x.shape[-2]), dtype = torch.bool, device= x.device)
            mask_cond = torch.triu(mask_cond, diagonal = 1)
            self.mask = torch.where(mask_cond, False, True)
        if is_rope:  
            assert token_positions is not None, "Without passing `token_positions`!"

            query_multihead = self.rope_cal.forward(query_multihead, token_positions)
            key_multihead = self.rope_cal.forward(key_multihead, token_positions)
            
        multi_head_results =  self._concat(attention(query_multihead, key_multihead, value_multihead, mask = self.mask))
        return self.cal_o.forward(multi_head_results)
        
    def _devide(
            self,
            x : Float[Tensor,"batch_size ... n d_model"]
    ) -> Float[Tensor, "batch_size ... head n d"]: 
        return rearrange(x,"... n (head d) -> ... head n d", head = self.num_heads)

    def _concat(
            self,
            x : Float[Tensor,"batch_size ... head n d"]
    ) -> Float[Tensor, "batch_size ... n d"]:
        return rearrange(x, " ... head n d -> ... n (head d)")