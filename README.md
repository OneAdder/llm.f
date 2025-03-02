# llm.f

f is for Fortran. Here I'm implementing language models
from scratch in Fortran. Inspired by 
[llm.c](https://github.com/karpathy/llm.c)

I contribute general purpose layers to
[neural-fortran](https://github.com/modern-fortran/neural-fortran)
deep learning library. Therefore, it is among this project deps
(but it's still tiny).

## Installation

```bash
fpm build
```

## Progress
### ML
| Layer           | Status   | Forward | Backward | Llama   | Qwen  |
|-----------------|----------|---------|----------|---------|-------|
| Llama Attention | ✅        | ✅       | ✅        | ✅       | ✅     |
| Silu MLP        | ✅        | ✅       | ✅        | ✅       | ✅     |
| RMSNorm         | ✅        | ✅       | ✅        | ✅       | ✅     |
| Decoder Layer   | ✅        | ✅       | ✅        | ✅       | ✅     |
| Llama Model     | ⌛        | ⌛       | ❌        |         |       |
| KV Caching      | ❌        | ❌       | -        |         |       |
| Text Generation | ❌        | ❌       | -        |         |       |
| Training        | ❌        | ❌       | ❌        |         |       |

### Infrastructure
1. Code fo loading safetensors. Will need to make Rust and Fortran friends
2. BPE Tokenizer. In progress
3. Graphics cards. Need to choose between C CUDA, Fortran CUDA and OpenACC
4. CMakeList
