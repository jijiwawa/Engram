"""
================================================================================
[Engram Architecture Demo Implementation]

DISCLAIMER:
1. Demo Purpose Only: 
   This code is a demonstration version intended to illustrate the core logic and 
   data flow of the Engram module.

2. Production Readiness: 
   This implementation requires further optimization for actual production use 
   (e.g., custom CUDA kernels, distributed training support).

3. Simplifications: 
   Standard components (Normalization, Attention, MoE) and complex Hyper-connection 
   mechanisms are omitted or mocked in this version to focus exclusively on the 
   Engram module implementation.
================================================================================
"""

"""
pip install torch numpy transformers sympy
"""

## built-in
from typing import List, Optional
from dataclasses import dataclass, field
import math
import json
import time

## Chrome Trace Utilities
class ChromeTracer:
    """Chrome Trace Event Profiler - outputs JSON format for chrome://tracing"""
    def __init__(self):
        self.events = []
        self._start_time = time.perf_counter()

    def _get_timestamp_us(self) -> int:
        """Get timestamp in microseconds"""
        return int((time.perf_counter() - self._start_time) * 1e6)

    def begin(self, name: str, category: str = "Engram", args: Optional[dict] = None):
        """Begin a trace event"""
        self.events.append({
            "name": name,
            "cat": category,
            "ph": "B",  # Begin
            "ts": self._get_timestamp_us(),
            "pid": 0,
            "tid": 0,
            "args": args or {}
        })

    def end(self, name: str, category: str = "Engram", args: Optional[dict] = None):
        """End a trace event"""
        self.events.append({
            "name": name,
            "cat": category,
            "ph": "E",  # End
            "ts": self._get_timestamp_us(),
            "pid": 0,
            "tid": 0,
            "args": args or {}
        })

    def instant(self, name: str, category: str = "Engram", args: Optional[dict] = None):
        """Add an instant event"""
        self.events.append({
            "name": name,
            "cat": category,
            "ph": "I",  # Instant
            "ts": self._get_timestamp_us(),
            "pid": 0,
            "tid": 0,
            "args": args or {}
        })

    def counter(self, name: str, value: float, category: str = "Engram"):
        """Add a counter event"""
        self.events.append({
            "name": name,
            "cat": category,
            "ph": "C",  # Counter
            "ts": self._get_timestamp_us(),
            "pid": 0,
            "tid": 0,
            "args": {"value": value}
        })

    def save(self, filepath: str):
        """Save trace to JSON file for Chrome trace viewer"""
        with open(filepath, 'w') as f:
            json.dump({"traceEvents": self.events}, f, indent=2)
        print(f"✅ Trace saved to: {filepath}")

    def clear(self):
        """Clear all events"""
        self.events = []

# Global tracer instance
tracer = ChromeTracer()

class trace_event:
    """Context manager for automatic trace begin/end"""
    def __init__(self, name: str, category: str = "Engram", args: Optional[dict] = None):
        self.name = name
        self.category = category
        self.args = args

    def __enter__(self):
        tracer.begin(self.name, self.category, self.args)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _ = exc_type, exc_val, exc_tb  # Unused but required by context manager protocol
        tracer.end(self.name, self.category)


def trace_init(category: str = "Initialization"):
    """Decorator for tracing __init__ methods"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            class_name = self.__class__.__name__
            trace_name = f"{class_name}.__init__"
            tracer.begin(trace_name, category, args={"class": class_name})
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                tracer.end(trace_name, category)
        return wrapper
    return decorator


def trace_method(category: str = "Engram"):
    """Decorator for tracing arbitrary methods"""
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            class_name = self.__class__.__name__
            trace_name = f"{class_name}.{func.__name__}"
            tracer.begin(trace_name, category)
            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                tracer.end(trace_name, category)
        return wrapper
    return decorator

## third-party
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30
    
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

class CompressedTokenizer:
    @trace_init(category="Tokenizer")
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        t = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
        print("init Compressed Tokenizer Vocab size:", self.tokenizer.vocab_size)
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        with trace_event("compress.index", category="Tokenizer"):
            arr = np.asarray(input_ids, dtype=np.int64)
            pos_mask = arr >= 0
            out = arr.copy()
            valid_ids = arr[pos_mask]
        with trace_event("compress.lookup_table", category="Tokenizer"):
            out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        with trace_event("CompressedTokenizer.compress", category="Tokenizer"):
            return self._compress(input_ids)
            
class ShortConv(nn.Module):
    @trace_init(category="Engram")
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) 
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        with trace_event("ShortConv.forward", category="Engram"):
            B, T, G, C = x.shape

            assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

            with trace_event("ShortConv.normalize", category="Engram"):
                normed_chunks = []
                for i in range(G):
                    chunk = x[:, :, i, :]
                    normed_chunks.append(self.norms[i](chunk))

            with trace_event("ShortConv.convolution", category="Engram"):
                x_norm = torch.cat(normed_chunks, dim=-1)
                x_bct = x_norm.transpose(1, 2)
                y_bct = self.conv(x_bct)
                y_bct = y_bct[..., :T]

            if self.activation:
                with trace_event("ShortConv.activation", category="Engram"):
                    y_bct = self.act_fn(y_bct)
            y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()

            return y
    
def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    @trace_init(category="Hash")
    def __init__(
        self,
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        with trace_event("NgramHashMapping.hash", category="Hash"):
            with trace_event("NgramHashMapping.compress_tokens", category="Hash"):
                input_ids = self.compressed_tokenizer(input_ids)
            hash_ids_for_all_layers = {}
            for layer_id in self.layer_ids:
                with trace_event(f"NgramHashMapping.layer_{layer_id}", category="Hash"):
                    hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
            return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):
    @trace_init(category="Engram")
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        with trace_event("MultiHeadEmbedding.forward", category="Engram"):
            shifted_input_ids = input_ids + self.offsets
            output = self.embedding(shifted_input_ids)
            return output
    
class Engram(nn.Module):
    @trace_init(category="Engram")
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size = engram_cfg.max_ngram_size,
            n_embed_per_ngram = engram_cfg.n_embed_per_ngram,
            n_head_per_ngram = engram_cfg.n_head_per_ngram,
            layer_ids = engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id = engram_cfg.pad_id,
            seed = engram_cfg.seed,
        )
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        self.short_conv = ShortConv(
            hidden_size = backbone_config.hidden_size,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
            hc_mult     = backbone_config.hc_mult,
        )
        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size,backbone_config.hidden_size)
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size,backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
    
    def forward(self,hidden_states,input_ids):
        """
        hidden_states: [B, L, HC_MULT, D]
        input_ids: [B, L]
        """
        with trace_event(f"Engram.layer_{self.layer_id}", category="Engram", args={"layer_id": self.layer_id}):
            with trace_event("Engram.hash_lookup", category="Engram"):
                hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])

            with trace_event("Engram.embedding_lookup", category="Engram"):
                embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)

            with trace_event("Engram.gate_computation", category="Engram"):
                gates = []
                for hc_idx in range(backbone_config.hc_mult):
                    key = self.key_projs[hc_idx](embeddings)
                    normed_key = self.norm1[hc_idx](key)
                    query = hidden_states[:,:,hc_idx,:]
                    normed_query = self.norm2[hc_idx](query)
                    gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
                    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
                    gate = gate.sigmoid().unsqueeze(-1)
                    gates.append(gate)
                gates = torch.stack(gates,dim=2)

            with trace_event("Engram.value_computation", category="Engram"):
                value = gates * self.value_proj(embeddings).unsqueeze(2)

            with trace_event("Engram.short_conv", category="Engram"):
                output = value + self.short_conv(value)

            return output 

class TransformerBlock(nn.Module):
    @trace_init(category="Model")
    def __init__(self, layer_id):
        super().__init__()
        self.attn = lambda x:x
        self.moe  = lambda x:x
        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id)
    
    def forward(self,input_ids,hidden_states):
        if self.engram is not None:
            hidden_states = self.engram(hidden_states=hidden_states,input_ids=input_ids) + hidden_states
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states

if __name__ == '__main__':
    # Clear any previous trace events
    tracer.clear()

    with trace_event("Model_Initialization", category="Model"):
        LLM = [
            nn.Embedding(backbone_config.vocab_size,backbone_config.hidden_size),
            *[TransformerBlock(layer_id=layer_id) for layer_id in range(backbone_config.num_layers)],
            nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)
        ]

    # Generate random input_ids with length 2048
    seq_len = 2048
    batch_size = 1

    # text = "Only Alexander the Great could tame the horse Bucephalus."
    with trace_event("Tokenizer", category="Model", args={"seq_len": seq_len}):
    # with trace_event("Tokenizer", category="Model"):
        tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path,trust_remote_code=True)
    #     input_ids = tokenizer(text,return_tensors='pt').input_ids
        vocab_size = len(tokenizer)
        # Generate random token IDs from 0 to vocab_size-1
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
        print(f"Generated random input_ids shape: {input_ids.shape}")

    B, L = input_ids.shape

    with trace_event("Model_Forward_Pass", category="Model", args={"batch_size": B, "seq_len": L}):
        for idx, layer in enumerate(LLM):
            if idx == 0:
                with trace_event("Embedding_Layer", category="Model"):
                    hidden_states = LLM[0](input_ids)
                    ## mock hyper-connection
                    hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1)
            elif idx == len(LLM)-1:
                ## mock hyper-connection
                hidden_states = hidden_states[:,:,0,:]
                with trace_event("Output_Layer", category="Model"):
                    output = layer(hidden_states)
            else:
                with trace_event(f"TransformerBlock.layer_{idx}", category="Model"):
                    hidden_states = layer(input_ids=input_ids,hidden_states=hidden_states)

    print("✅ Forward Complete!")
    print(f"{input_ids.shape=}\n{output.shape=}")

    # Save trace to JSON file for Chrome trace viewer
    tracer.save("engram_trace.json")
    print("\n💡 To view the trace:")
    print("   1. Open Chrome browser")
    print("   2. Navigate to chrome://tracing")
    print("   3. Click 'Load' and select 'engram_trace.json'")
            