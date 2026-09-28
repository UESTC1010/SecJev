# Third-party notices

The SecJev Use and Distribution Agreement applies only to the rights held by SecJev contributors. The following upstream licenses remain in effect and are not replaced by that agreement.

| Component | Attribution | Applicable terms |
|---|---|---|
| Kev source and initial Kev-0.8B adapter/head | Jared Palmer and Kev contributors | Apache-2.0; see licenses/Apache-2.0.txt and UPSTREAM.md |
| Qwen3.5-0.8B-Base and Qwen3.5-2B-Base backbones | Qwen team / Alibaba and applicable contributors | Apache-2.0; see licenses/Apache-2.0.txt and UPSTREAM.md |
| Twins / Streamlet simulator | Twins simulator contributors | licenses/Twins-LICENSE.txt |
| ByzFL | EPFL and ByzFL contributors | licenses/ByzFL-LICENSE.txt |
| FoolsGold | FoolsGold contributors | licenses/FoolsGold-LICENSE.txt |
| AgentDojo | ETH Zurich SPY Lab and contributors | licenses/AgentDojo-LICENSE.txt |
| InjecAgent | UIUC Kang Lab and contributors | licenses/InjecAgent-LICENSE.txt |
| VeReMi NextGen data | Authors listed in SOURCES.md | licenses/VeReMi-LICENSE.txt; CC BY 4.0 |
| LANL source observations | Alexander D. Kent / Los Alamos National Laboratory | licenses/LANL.md; public-domain source dedication |
| CICIoT2023 and ToN-IoT data | Authors listed in SOURCES.md | Official source-specific conditions linked in SOURCES.md |

SecJev modifies the Kev adapter and decision head by security fine-tuning and probability calibration. The backbone is an upstream dependency. SecJev-Corpus transformations and newly generated experiments are described in SOURCES.md; using a simulator does not imply its software license automatically covers every output or input dataset. Full data attribution and required publications are retained in SOURCES.md.

SecJev-2B trains its own LoRA/head from the pinned Qwen2B backbone with Kev's decision recipe before security adaptation; the initial Kev-0.8B adapter applies only to the 0.8B model. Kev general training source records retain their upstream attribution and conditions.

## SecJev-4B

Initialized from jaredpalmer/kev-4b@485ace8703592fcf405488b262449990824cfed1 on Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b, then adapted for three epochs on SecJev-Corpus v1.0.0. Existing SecJev and upstream terms remain applicable.

## SecJev-9B

Initialized from jaredpalmer/kev-9b@2629c06a5aeb0feb3b9783bafed17ed8f39ecf5c on Qwen/Qwen3.5-9B-Base@68c46c4b3498877f3ef123c856ecfde50c39f404, then trained for three epochs on SecJev-Corpus v1.0.0. The existing SecJev agreement and applicable upstream terms continue to apply.
