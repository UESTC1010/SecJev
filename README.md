<div align="center">

# SecJev

**Jev-like System One decision models for security**

[中文](README.zh-CN.md) · [Models](#models) · [Usage](#usage) · [SecJev-Corpus](#secjev-corpus) · [Results](#results) · [License](#license)

</div>

Give SecJev a log, a tool response, or a set of votes, along with the question you want answered. It scores the candidate answers and returns a judgment, a choice, or a rating with probabilities.

The current **SecJev-0.8B** is fine-tuned from Kev-0.8B on the Qwen3.5-0.8B-Base backbone. We train it on **SecJev-Corpus**, a dataset covering prompt injection, network attack labels, federated learning, consensus protocols, authentication logs, and vehicle messages.

We plan to release the weights, training and inference code, and SecJev-Corpus together. The 0.8B model and corpus are ready; uploads are pending.

## Models

| Model | Size | Status |
|---|---:|---|
| **SecJev-0.8B** | 0.8B | Trained and tested; upload pending |
| SecJev-2B | 2B | Planned |
| SecJev-4B | 4B | Planned |
| SecJev-9B | 9B | Planned |
| SecJev-27B | 27B | Planned |

The 0.8B package contains a LoRA adapter, decision head, and calibration parameters. It requires the Qwen3.5-0.8B-Base backbone. See [TRAINING.md](TRAINING.md) for the training recipe.

## Usage

The interface follows [Kev](https://github.com/jaredpalmer/kev): `state` holds the evidence and `questions` holds the questions. One request can contain all three question types:

| Type | Use | Output |
|---|---|---|
| `noul` | Test whether a condition holds | Probability of true |
| `choice` | Choose from named options | Selected option and option probabilities |
| `score` | Rate on an ordered scale | Category probabilities and expected level |

For example, three validators support block A, but one also voted for B in the same round. The policy requires rejecting a certificate with a conflicting voter.

```json
{
  "state": {
    "policy": "Reject if a candidate voter also voted for another block in the same round. Otherwise accept with at least three distinct voters; otherwise wait.",
    "candidate": "A",
    "round": 7,
    "observed_votes": {"v1": ["A"], "v2": ["A", "B"], "v3": ["A"]}
  },
  "questions": {
    "certificate_action": {
      "type": "choice",
      "instructions": "Which action follows the policy?",
      "criteria": {
        "accept": "Accept the certificate",
        "reject": "Reject the conflicting certificate",
        "wait": "Wait for more votes"
      }
    }
  }
}
```

The labeled answer for this example is `reject`.

Once you have the model files and have followed the [environment setup](TRAINING.md#environment-and-inputs), run the included example:

```bash
python infer.py \
  --model /path/to/SecJev-0.8B \
  --request examples/certificate.json \
  --device cuda
```

The script prints the model's actual answer and probabilities. See [INFERENCE.md](INFERENCE.md) for output fields and loading a local backbone.

## SecJev-Corpus

We built SecJev-Corpus to train and evaluate SecJev. **Version 1.0.0 contains 170,185 questions from 115,324 scenes, covering 8 sources and 14 tasks.**

We preserve source annotations and turn logs, traffic features, and experimental observations into labeled decision questions. For policy tasks, the rules are part of the input. LANL questions, for example, ask how to handle a logon given recent failures; Twins questions ask whether the observed votes justify accepting a certificate.

### Sources

| Source | Security decision tasks | Questions across all splits |
|---|---|---:|
| **CICIoT2023** | Attack-capture label prediction from traffic features | 48,000 |
| **ToN-IoT** | Attack/normal label prediction from connection features | 40,000 |
| **Twins / Streamlet** | Observed vote conflicts, quorum, certificate policy actions, and evidence levels | 29,588 |
| **VeReMi NextGen** | Message-manipulation labels and ten-way attack/normal classification | 22,498 |
| **ByzFL-based controlled experiments** | Identifying attack-generated updates from client-update summaries | 17,000 |
| **LANL authentication events** | Prior-failure thresholds and explicit-policy logon triage | 7,902 |
| **AgentDojo** | Embedded prompt injection and trust-boundary actions in tool outputs | 4,143 |
| **InjecAgent** | Classifying the operation requested by an injected instruction | 1,054 |

The Twins and ByzFL data come from controlled experiments we ran. Other sources are processed from their original records and annotations. Versions, transformations, and citations are listed in [SOURCES.md](SOURCES.md).

Each record has a `state` and one or more labeled `questions`. The corpus contains **137,067 Boolean, 25,721 choice, and 7,397 ordered-category questions**. The package includes JSONL requests and a tabular format. Labels and source metadata are excluded from model inputs.

### Splits

| Split | Scenes | Questions | Role |
|---|---:|---:|---|
| **Training** | 81,410 | 118,818 | Update model parameters |
| **Calibration** | 9,333 | 14,821 | Fit the probability-calibration temperature |
| **Development** | 9,911 | 14,912 | Compare and select checkpoints |
| **Test** | 14,670 | 21,634 | Report per-task and overall performance |

The security results below use the **21,634 test questions** in the last row. Calibration adjusts probabilities; development selects the checkpoint. These are separate splits.

We remove duplicate inputs and conflicting labels, split by source groups, and check class distributions. Within each task and split, every class has at least 50 questions and no class exceeds 80%. The largest cross-split difference in a class's share is 8.58 percentage points.

## Results

SecJev-0.8B was trained for three epochs. Epoch 3 was selected by development macro-task balanced accuracy. We compare it with original Kev-0.8B on the **SecJev-Corpus v1.0.0 test split**, using the same questions and scoring method. Both models fit their temperatures only on the calibration split.

| Metric | Original Kev-0.8B | SecJev-0.8B |
|---|---:|---:|
| **Macro-average accuracy (equal task weights)** | 50.94% | **90.34%** |
| Macro-average balanced accuracy (equal task weights) | 51.86% | **88.00%** |
| Micro accuracy (all questions pooled) | 40.53% | **85.03%** |
| Expected calibration error (15 bins; lower is better) | 17.07% | **3.34%** |

### Results by task

| Task | Test questions | Kev accuracy | SecJev accuracy | SecJev balanced accuracy |
|---|---:|---:|---:|---:|
| AgentDojo · Trust-boundary action | 426 | 51.88% | 80.28% | 78.04% |
| AgentDojo · Embedded prompt injection | 426 | 32.86% | 79.81% | 79.58% |
| ByzFL · Attack-generated client update | 800 | 79.62% | 92.62% | 91.88% |
| CICIoT2023 · Attack-capture label | 6,000 | 25.00% | 91.80% | 88.07% |
| InjecAgent · Requested operation | 124 | 44.35% | 100.00% | 100.00% |
| LANL · Logon policy triage | 540 | 72.41% | 98.89% | 98.58% |
| LANL · Prior-failure threshold | 540 | 80.74% | 99.81% | 99.88% |
| ToN-IoT · Connection attack label | 3,500 | 25.00% | 98.03% | 97.01% |
| Twins/Streamlet · Evidence level | 659 | 41.12% | 98.79% | 98.66% |
| Twins/Streamlet · Certificate policy action | 659 | 47.65% | 98.48% | 98.33% |
| Twins/Streamlet · Quorum | 659 | 53.41% | 99.70% | 99.76% |
| Twins/Streamlet · Observed vote conflict | 659 | 64.04% | 99.24% | 98.74% |
| VeReMi · Message-manipulation label | 3,321 | 47.76% | 63.11% | 63.55% |
| VeReMi · Ten-way attack/normal label | 3,321 | 47.24% | 64.14% | 39.88% |
| **Macro-average · 14 tasks** | — | **50.94%** | **90.34%** | **88.00%** |

**Macro-average** is the arithmetic mean of the 14 task accuracies, with equal weight per task. **Micro accuracy** pools all questions. **Macro-average balanced accuracy** averages semantic-class recall within each task, then averages across tasks. Accuracy for `score` questions uses the highest-probability level.

The two VeReMi tasks remain weak, particularly ten-way classification at **39.88% balanced accuracy**. The [evaluation report](evaluation/README.md) has the full metrics and original task IDs; [task-results.json](evaluation/task-results.json) contains the per-task values. We also provide the [supplementary comparison of all three checkpoints](evaluation/checkpoint-audit/README.md).

These research splits were inspected and balanced during construction. Samples from a shared capture or experiment may be correlated. Traffic and vehicle questions predict source labels, and LANL questions follow the stated policy, so these scores should not be read as attack-detection rates in a live deployment.

### General capability after fine-tuning

We also tested both models on the official Kev suites:

| Test suite | Kev-0.8B | SecJev-0.8B |
|---|---:|---:|
| decision-v7 (in-domain tasks) | 83.50% | 83.08% |
| transfer-v4 (new sources) | 68.45% | 66.01% |

Accuracy was nearly unchanged on decision-v7 and fell by 2.44 percentage points on transfer-v4. Transfer calibration also worsened: with each model's stored temperature, ECE rose from **2.56% to 21.10%**. The fine-tuned model is more prone to overconfidence on these tasks. See the [retention evaluation](evaluation/retention/README.md) for details. These ECE values use 10 bins; the security table uses 15.

## Next steps

- Publish SecJev-0.8B and SecJev-Corpus.
- Train and evaluate the 2B, 4B, 9B, and 27B models.
- Add inference speed, memory measurements, and deployment instructions.

## Acknowledgments

SecJev builds on [Kev](https://github.com/jaredpalmer/kev) and the Qwen backbone. Thanks to the authors of the datasets, tools, and simulators used in this project. See [SOURCES.md](SOURCES.md) and [third-party notices](THIRD_PARTY_NOTICES.md) for the full list.

## License

Use and distribution of the models, data, and code are governed by the [SecJev Use and Distribution Agreement v1.0](LICENSE) ([中文](LICENSE.zh-CN.md)) and the applicable upstream terms. For example, commercial use of ToN-IoT data requires separate permission from its authors. See the agreement and [source register](SOURCES.md) for details.
