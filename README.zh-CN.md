<div align="center">

# SecJev

**面向安全决策的 Jev-like System One 模型**

[English](README.md) · [模型](#模型) · [使用示例](#使用示例) · [SecJev-Corpus](#secjev-corpus) · [测试结果](#测试结果) · [协议](#使用协议)

</div>

给模型一段日志、一份工具响应或一组投票记录，再告诉它要判断什么。SecJev 会为候选答案打分，返回判断、选项或等级，以及相应的概率。

**SecJev-0.8B** 从 Kev-0.8B 微调而来。**SecJev-2B** 先从 Qwen3.5-2B-Base 学习 Kev 的通用决策任务，再进行同样的三轮安全领域训练。训练使用我们整理的 **SecJev-Corpus**，包含提示注入、网络攻击标签、联邦学习、共识协议、认证日志和车联网消息等任务。

SecJev-2B 已发布在 [v1.1.0](https://github.com/UESTC1010/SecJev/releases/tag/v1.1.0)。0.8B 模型和原版数据集继续保留在 [v1.0.0](https://github.com/UESTC1010/SecJev/releases/tag/v1.0.0)。

| 下载 | 内容 |
|---|---|
| [SecJev-0.8B](https://github.com/UESTC1010/SecJev/releases/download/v1.0.0/SecJev-0.8B-v1.0.0.tar.gz) | 训练好的 LoRA 权重、决策头、校准参数和分词器 |
| [SecJev-2B](https://github.com/UESTC1010/SecJev/releases/download/v1.1.0/SecJev-2B-v1.1.0.tar.gz) | 训练好的 LoRA 权重、决策头、校准参数和推理入口 |
| [SecJev-Corpus](https://github.com/UESTC1010/SecJev/releases/download/v1.0.0/SecJev-Corpus-v1.0.0.tar.gz) | 全部 170,185 道题、四个数据划分和来源信息 |
| [代码](https://github.com/UESTC1010/SecJev/releases/download/v1.1.0/SecJev-code-v1.1.0.tar.gz) | 推理、三轮训练、检查点选择、校准和评估 |

## 模型

| 模型 | 参数规模 | 状态 |
|---|---:|---|
| **SecJev-0.8B** | 0.8B | [已发布](https://github.com/UESTC1010/SecJev/releases/tag/v1.0.0) |
| **SecJev-2B** | 2B | [已发布](https://github.com/UESTC1010/SecJev/releases/tag/v1.1.0) |
| SecJev-4B | 4B | 计划中 |
| SecJev-9B | 9B | 计划中 |
| SecJev-27B | 27B | 计划中 |

0.8B 版本包含 LoRA 适配器、决策头和校准参数，运行时还需要 Qwen3.5-0.8B-Base。训练配置见 [TRAINING.md](TRAINING.md)。

## 使用示例

接口沿用 [Kev](https://github.com/jaredpalmer/kev) 的格式：`state` 放待分析的内容，`questions` 放问题。一份输入可以同时包含三种题型：

| 类型 | 用途 | 返回内容 |
|---|---|---|
| `noul` | 判断某个条件是否成立 | 条件成立的概率 |
| `choice` | 从给定选项中选择 | 选项及各选项的概率 |
| `score` | 按有序等级评分 | 等级分布及期望等级 |

例如，检查一份投票证书：虽然已有三名验证者支持 A，但其中一人也在同一轮投给了 B。规则要求拒绝这种证书。

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

这个示例的标注答案是 `reject`。

拿到模型文件并按[安装说明](TRAINING.md#environment-and-inputs)准备好环境后，可以运行仓库中的示例：

```bash
python infer.py \
  --model /path/to/SecJev-0.8B \
  --request examples/certificate.json \
  --device cuda
```

脚本会打印模型的实际答案和概率。输出字段、本地底座路径等用法见 [INFERENCE.md](INFERENCE.md)。

## SecJev-Corpus

SecJev-Corpus 是我们为训练和测试 SecJev 整理的数据集。**v1.0.0 共 170,185 道题，来自 115,324 个场景，覆盖 8 个来源、14 项任务。**

我们保留了原始数据的标注，并把日志、流量特征和实验观察整理成模型可以直接读取的决策题。需要策略判断的任务，会把规则写进输入。例如，LANL 的题目要求根据历史失败次数决定如何分流登录事件；Twins 的题目要求根据投票记录判断证书是否应被接受。

### 数据从哪里来

| 来源 | 组织成的安全决策任务 | 全语料题数 |
|---|---|---:|
| **CICIoT2023** | 根据流量特征预测攻击捕获标签 | 48,000 |
| **ToN-IoT** | 根据连接特征预测攻击/正常标签 | 40,000 |
| **Twins / Streamlet** | 可见投票冲突、法定票数、证书策略动作与证据等级 | 29,588 |
| **VeReMi NextGen** | 车辆消息操纵标签与十类攻击/正常标签识别 | 22,498 |
| **基于 ByzFL 的受控实验** | 根据客户端更新摘要识别攻击过程生成的更新 | 17,000 |
| **LANL 认证事件** | 历史失败次数阈值判断与明确登录策略下的分流 | 7,902 |
| **AgentDojo** | 工具输出中的嵌入式提示注入与信任边界动作 | 4,143 |
| **InjecAgent** | 注入请求所要求的操作类别识别 | 1,054 |

其中，Twins 和 ByzFL 数据来自我们运行的受控实验，其余来源按原始记录和标注处理。具体版本、处理方法和引用信息见 [SOURCES.md](SOURCES.md)。

每条记录包含一个 `state` 和一个或多个带答案标签的 `questions`。全语料有 **137,067 道判断题、25,721 道选择题和 7,397 道等级题**。数据包提供 JSONL 请求格式和表格格式；标签与来源信息不作为模型输入。

### 训练和测试怎么划分

| 划分 | 场景数 | 题数 | 用途 |
|---|---:|---:|---|
| **训练集** | 81,410 | 118,818 | 更新模型参数 |
| **校准集** | 9,333 | 14,821 | 拟合概率校准温度 |
| **开发验证集** | 9,911 | 14,912 | 比较并选择 checkpoint |
| **测试集** | 14,670 | 21,634 | 报告模型的逐任务与总体表现 |

下文的安全测试使用最后一行的 **21,634 道题**。校准集用来调整概率，开发验证集用来选择 checkpoint，两者分开使用。

整理数据时，我们去掉了重复输入和矛盾标签，按来源分组划分集合，并检查了类别分布。每项任务在每个集合内，每类至少有 50 道题，任何一类不超过 80%；同类别跨集合的比例差最大为 8.58 个百分点。

## 测试结果

SecJev-0.8B 训练了三轮，按开发验证集的任务平均平衡准确率选中了第 3 轮。下面比较它与原始 Kev-0.8B 在 **SecJev-Corpus v1.0.0 测试集**上的表现。两者使用相同的题目和评分方法，温度参数均只在校准集上拟合。

| 指标 | 原始 Kev-0.8B | SecJev-0.8B |
|---|---:|---:|
| **Macro-average accuracy（任务等权）** | 50.94% | **90.34%** |
| Macro-average balanced accuracy（任务等权） | 51.86% | **88.00%** |
| Micro accuracy（全部题目合并） | 40.53% | **85.03%** |
| 期望校准误差（15 个区间，越低越好） | 17.07% | **3.34%** |

### 分任务结果

| 任务 | 测试题数 | Kev 准确率 | SecJev 准确率 | SecJev 平衡准确率 |
|---|---:|---:|---:|---:|
| AgentDojo · 信任边界动作 | 426 | 51.88% | 80.28% | 78.04% |
| AgentDojo · 嵌入式提示注入 | 426 | 32.86% | 79.81% | 79.58% |
| ByzFL · 攻击生成的客户端更新 | 800 | 79.62% | 92.62% | 91.88% |
| CICIoT2023 · 攻击捕获标签 | 6,000 | 25.00% | 91.80% | 88.07% |
| InjecAgent · 注入请求的操作类别 | 124 | 44.35% | 100.00% | 100.00% |
| LANL · 登录策略分流 | 540 | 72.41% | 98.89% | 98.58% |
| LANL · 历史失败次数阈值 | 540 | 80.74% | 99.81% | 99.88% |
| ToN-IoT · 连接攻击标签 | 3,500 | 25.00% | 98.03% | 97.01% |
| Twins/Streamlet · 证据等级 | 659 | 41.12% | 98.79% | 98.66% |
| Twins/Streamlet · 证书策略动作 | 659 | 47.65% | 98.48% | 98.33% |
| Twins/Streamlet · 法定票数 | 659 | 53.41% | 99.70% | 99.76% |
| Twins/Streamlet · 可见投票冲突 | 659 | 64.04% | 99.24% | 98.74% |
| VeReMi · 消息操纵标签 | 3,321 | 47.76% | 63.11% | 63.55% |
| VeReMi · 十类攻击/正常标签 | 3,321 | 47.24% | 64.14% | 39.88% |
| **Macro-average · 14 项任务** | — | **50.94%** | **90.34%** | **88.00%** |

**Macro-average** 是 14 项任务准确率的算术平均，每项任务权重相同；**Micro accuracy** 把全部题目放在一起计算。**Macro-average balanced accuracy** 先平均任务内各语义类别的召回率，再对任务取平均。`score` 题的准确率按最高概率等级计算。

VeReMi 的两项任务目前表现较弱，十类标签识别的平衡准确率为 **39.88%**。完整指标和原始任务名见[评测报告](evaluation/README.md)，逐任务数据可从 [task-results.json](evaluation/task-results.json) 读取。[三个 checkpoint 的补测结果](evaluation/checkpoint-audit/README.zh-CN.md)也已保留。

这些数据在整理过程中经过检查和平衡，属于研究测试集；同一捕获或实验中的样本可能相关。流量和车辆消息题预测的是来源标签，LANL 题按给定策略作答，不能把这些成绩直接解释成真实环境中的攻击检出率。

### 通用能力保留了多少

我们还用 Kev 官方测试集比较了微调前后的表现：

| 测试集 | Kev-0.8B | SecJev-0.8B |
|---|---:|---:|
| decision-v7（常见任务） | 83.50% | 83.08% |
| transfer-v4（跨领域任务） | 68.45% | 66.01% |

常见任务准确率基本持平，跨领域下降了 2.44 个百分点。跨领域的概率校准也变差了：使用各自现有的温度参数，ECE 从 **2.56% 升至 21.10%**。这意味着模型在这些任务上更容易过度自信。完整结果见[通用能力测试](evaluation/retention/README.zh-CN.md)。这里的 ECE 使用 10 个区间，安全测试表使用 15 个区间。

### SecJev-2B

2B 使用相同的 SecJev-Corpus 划分，验证集选中第 3 轮，三个 checkpoint 均已测试。它先从 Qwen3.5-2B-Base 按 Kev 的方法和数据训练通用决策能力，再进行三轮安全领域训练；本次没有使用官方 Kev-2B 权重。

| 模型 | 总准确率 | 任务宏平均准确率 | 任务宏平均平衡准确率 |
|---|---:|---:|---:|
| SecJev-0.8B | 85.03% | 90.34% | 88.00% |
| **SecJev-2B** | **86.48%** | **92.98%** | **90.31%** |

在原始 Kev 通用测试的 clean 子集上，2B 的 decision-v7 准确率为 **84.50%**，transfer-v4 为 **75.91%**。详见 [14 项任务与三轮 checkpoint 的结果](evaluation/secjev2b/README.zh-CN.md)、[2B 训练方法](training/secjev2b/README.md)和[推理说明](INFERENCE.md#secjev-2b)。


## 接下来

- 训练并评测 4B、9B、27B 版本。
- 补充推理速度、显存占用和部署说明。

## 致谢

SecJev 基于 [Kev](https://github.com/jaredpalmer/kev) 的决策模型实现和 Qwen 底座。感谢各数据集、工具与模拟器的作者，完整列表见 [SOURCES.md](SOURCES.md) 和[第三方声明](THIRD_PARTY_NOTICES.md)。

## 使用协议

模型、数据和代码的使用与分发，请遵守 [SecJev 使用与分发协议 v1.0](LICENSE.zh-CN.md)（[英文](LICENSE)）及相应的上游条款。例如，ToN-IoT 数据用于商用需要另行取得作者许可。具体条款见协议和[来源说明](SOURCES.md)。
