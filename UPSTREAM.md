# Upstream components

SecJev-0.8B adapts the LoRA parameters and pointer decision head of Kev-0.8B.

| Component | Pinned revision | Terms |
|---|---|---|
| [Kev source](https://github.com/jaredpalmer/kev) | `90990a5fac2995b9faa3190f7d437e84f2067768` | Apache-2.0 |
| [Kev-0.8B initialization](https://huggingface.co/jaredpalmer/kev-0.8b) | `54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8` | Apache-2.0 |
| [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` | Apache-2.0 |

The inference implementation and training loss use pinned Kev source. SecJev adds security adaptation, batching, the training/evaluation workflow, and the security corpus. See SecJev-Corpus/SOURCES.md for data attribution and source-specific conditions. Dataset terms are not replaced by the code license.
