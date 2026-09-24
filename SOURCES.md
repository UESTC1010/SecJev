# Sources and attribution

SecJev-Corpus v1.0.0 transforms upstream observations into structured decision
questions. Transformations include feature projection, identifier anonymization,
history-window construction, explicit-policy labeling, deduplication, option
permutation, and grouped splitting. Source annotations are not relabeled as
proven real-world attacks. Newly run simulations are identified as SecJev
experiments rather than results reported by the upstream authors.

## CICIoT2023 — 48,000 questions

E. C. P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, and A. A. Ghorbani.
*CICIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment* (2023).

- [Dataset and paper](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- [CIC redistribution and citation policy, under Common questions](https://www.unb.ca/cic/datasets/index.html)
- Mirror used: `bencorn/CIC-IoT-2023`, revision `dac6afc7c1f6242202abf6059932e4c4d166e662`.

CIC explicitly permits redistribution and mirroring with dataset and paper attribution.
Capture labels remain capture-level annotations; incidental background traffic is possible.

## ToN-IoT — 40,000 questions

[Official dataset and usage conditions](https://research.unsw.edu.au/projects/toniot-datasets).
Free academic research use is granted by the authors; commercial use requires
separate author permission. The source requests citation of the following eight
publications, linked from its official page:

1. Nour Moustafa (2021), network dataset study, *Sustainable Cities and Society*, 102994.
2. Tim M. Booij, Irina Chiscop, Erik Meeuwissen, Nour Moustafa, and Frank T. H. den Hartog (2021), ToN-IoT heterogeneity and feature standardization, *IEEE Internet of Things Journal*.
3. Abdullah Alsaedi, Nour Moustafa, Zahir Tari, Abdun Mahmood, and Adnan Anwar (2020), TON-IoT telemetry dataset, *IEEE Access* 8, 165130–165150.
4. Nour Moustafa, M. Keshk, E. Debie, and H. Janicke (2020), federated Windows datasets, *TrustCom*, 848–855. DOI: 10.1109/TrustCom50675.2020.00114.
5. Nour Moustafa, M. Ahmed, and S. Ahmed (2020), ToN-IoT Linux dataset evaluation, *TrustCom*, 727–735. DOI: 10.1109/TrustCom50675.2020.00100.
6. Nour Moustafa (2019), TON-IoT dataset presentation, *eResearch Australasia*.
7. Nour Moustafa (2019), IoT–Fog–Cloud architecture review. [arXiv:1906.01055](https://arxiv.org/abs/1906.01055).
8. Javed Ashraf et al. (2021), IoTBoT-IDS, *Sustainable Cities and Society*, 103041.

Mirror used: `codymlewis/TON_IoT_network`, revision
`b953d9388d2db3bc392ea7150476e4f219a2a1cb`. Features derive from the fixed network CSV.

## VeReMi NextGen — 22,498 questions

Artur Hermann, Jan-Niklas Remmers, Dennis Eisermann, Benjamin Erb, and Frank Kargl.
*VeReMi NextGen: A Dataset for Evaluating Misbehavior Detection Systems in VANETs*.
[Zenodo record / DOI 10.5281/zenodo.19665762](https://zenodo.org/records/19665762).
License: CC BY 4.0; full notice in `licenses/VeReMi-LICENSE.txt`.

Only validated observation categories enter this corpus. Sender claims are
projected into a common representation. The official independent test boundary is retained.

## Twins / Streamlet — 29,588 questions

[Twins simulator](https://github.com/asonnino/twins-simulator), revision
`3a3215addf3f79cb7b1dd2b0677f4d0896943d0b`; Apache-2.0 notice included.
These are newly sampled controlled experiments using the upstream implementation.
Answers follow the explicit observer certificate policy and visible message history.
A validator ID follows the simulator's physical-author representation; no hidden
attacker identity is inferred.

## ByzFL / FoolsGold / MNIST — 17,000 questions

- [ByzFL](https://github.com/LPD-EPFL/byzfl), EPFL, revision `5830978d991a0900748f7ce901dc3ca532081b26`; MIT.
- [FoolsGold](https://github.com/DistributedML/FoolsGold), revision `0aa55114296a2d3c2bcb6f544a6fae31e8e7b8b4`; MIT.
- Yann LeCun, Corinna Cortes, and Christopher J. C. Burges, MNIST handwritten-digit data, underlying the controlled gradient experiments.

The corpus contains measured update summaries from 80 fresh runs, with separate
MNIST pools across splits and explicit attacker configurations. Original images
are not redistributed. The earlier simulated aggregation-selection tasks are
not part of this release.

## LANL — 7,902 questions

Alexander D. Kent (2015), *Comprehensive, Multi-Source Cyber-Security Events*.
Los Alamos National Laboratory. DOI: 10.17021/1179829.
Related publication: A. D. Kent, *Cybersecurity Data Sources for Dynamic Network
Research*, in *Dynamic Networks in Cybersecurity*, 2015.

[Official data, citation and public-domain dedication](https://csr.lanl.gov/data/cyber1/).
Our questions use visible authentication histories and explicit triage rules.
Unlabeled events are not assigned a confirmed-benign label. The source's
successful-authentication selection bias remains applicable.

## AgentDojo — 4,143 questions

[AgentDojo](https://github.com/ethz-spylab/agentdojo), ETH Zurich SPY Lab and contributors,
revision `089ed468cf3ed0322acc66b0211f26d9d90dbf60`; MIT notice included.
Records are drawn from published tool outputs and attack/control runs. Task and
shared-response components are grouped; shared environment or attack templates
are not a claim of unseen-template evaluation.

## InjecAgent — 1,054 questions

[InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent), UIUC Kang Lab and contributors,
revision `f19c9f2c79a41046eb13c03c51a24c567a8ffa07`; MIT notice included.
This release uses the declared objectives of injected instructions. The
all-positive injection-detection task is excluded.
