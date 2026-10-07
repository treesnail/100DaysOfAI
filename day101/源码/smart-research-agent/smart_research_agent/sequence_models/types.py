"""``sequence_models`` 的口径表（day094 / M8-D5）.

一次性把这一课的"名词表"写全：**两种循环单元 / 四个门 / 七条性质 / 十条笔记 /
五条边界 / 一张 PyTorch 对照表 / 四张公式**。全部是常量，因此可以被测试逐键检查。

```text
两种循环单元  rnn（一个状态 h） / lstm（一个状态 h 加一条记忆 c）
四个门        input(i) / forget(f) / output(o) / candidate(g)   —— LSTM 的三门 + 一个候选
七条性质      定义 1 条 + 门 1 条 + 时间轴 2 条 + 梯度 2 条 + 长程对比 1 条
```

## 这一课与 day093 的边界：**权重共享换了一个维度**

```text
day093 卷积  同一组核被用到**空间的每一个位置**   ⇒ 参数从 O(H·W) 降到 O(k²)
day094 循环  同一组权重被用到**时间轴的每一个时刻** ⇒ 参数与序列长度 T **完全无关**
```

两句话是同一件事的两个实例。区别在于：卷积共享的是"位置"，循环共享的是"时刻"；
卷积的每一步彼此**独立**（可以并行算全图），循环的每一步**依赖上一步**
（必须按顺序算，因此天生慢）。这条"并行 vs 串行"的对比是第 10 章与
day075 的 Transformer 对照的落点。

因此本课复用 day074 的数值差分当尺子、复用 day089 的 LCG 初始化与激活前向、
复用 day090 的 ``elementwise_backward`` 做激活的反向导数、
复用 day092 的 ``TrainingOptimizer`` 做一次真的训练——**循环本身只有一份实现**。
"""

from __future__ import annotations

from smart_research_agent.neural_basics.types import ACTIVATIONS
from smart_research_agent.sequence_models.errors import RecurrentError

# --------------------------------------------------------------------------- #
# 闭合表 1：两种循环单元
# --------------------------------------------------------------------------- #

CELL_RNN = "rnn"
CELL_LSTM = "lstm"

#: 两种循环单元（顺序 = 从一个状态到两个状态）.
CELL_TYPES: tuple[str, ...] = (CELL_RNN, CELL_LSTM)

#: 每种循环单元的一句话解释.
CELL_DESCRIPTIONS: dict[str, str] = {
    CELL_RNN: "朴素循环单元：一个隐状态 h_t = act(W_h·h_{t−1} + W_x·x_t + b)，信息只有一条通道",
    CELL_LSTM: "长短期记忆单元：多出一条**记忆通道** c，靠三个门决定'写多少 / 忘多少 / 读多少'",
}

#: 每种循环单元的状态形状（一个状态 vs 一对状态）.
CELL_STATE_SHAPES: dict[str, str] = {
    CELL_RNN: "(h,)：长度为 H 的一个向量",
    CELL_LSTM: "(h, c)：两条长度均为 H 的向量（h 是给外部的读数，c 是给未来的记忆）",
}

# --------------------------------------------------------------------------- #
# 闭合表 2：四个门（只有 LSTM 有）
# --------------------------------------------------------------------------- #

GATE_INPUT = "input"
GATE_FORGET = "forget"
GATE_OUTPUT = "output"
GATE_CANDIDATE = "candidate"

#: 四个门（顺序 = 方程里出现的顺序 = i / f / o / g；写进权重的行块顺序也是它）.
GATES: tuple[str, ...] = (GATE_INPUT, GATE_FORGET, GATE_OUTPUT, GATE_CANDIDATE)

#: 每个门的一句话解释.
GATE_DESCRIPTIONS: dict[str, str] = {
    GATE_INPUT: "输入门 i：这一时刻**写多少**新信息进记忆（σ，取值 0~1）",
    GATE_FORGET: "遗忘门 f：旧记忆**保留多少**（σ，取值 0~1；f=1 表示原样带回）",
    GATE_OUTPUT: "输出门 o：这一时刻**读出多少**记忆给外部（σ，取值 0~1）",
    GATE_CANDIDATE: "候选 g：准备写进记忆的**内容**（tanh，取值 −1~1）",
}

#: 每个门的激活（**写下来**：门的激活与候选的激活不是同一个函数）.
GATE_ACTIVATIONS: dict[str, str] = {
    GATE_INPUT: "sigmoid",
    GATE_FORGET: "sigmoid",
    GATE_OUTPUT: "sigmoid",
    GATE_CANDIDATE: "tanh",
}

if not (
    set(GATES)
    == set(GATE_DESCRIPTIONS)
    == set(GATE_ACTIVATIONS)
):  # pragma: no cover - 导入期不变式
    raise RecurrentError(
        "四个门的两张表不一致：GATES / GATE_DESCRIPTIONS / GATE_ACTIVATIONS 必须逐键对齐——"
        "少一个键的门会在实现里静默地没有激活，而'忘了激活'与'门算对了'在读表时长得一样。"
    )

#: 权重矩阵里四个门各自的**行块**（``wx`` / ``wh`` / ``b`` 都是 ``4H`` 行，按这个顺序切）.
GATE_ROW_BLOCKS: dict[str, str] = {
    GATE_INPUT: "行 [0H, 1H)",
    GATE_FORGET: "行 [H, 2H)",
    GATE_OUTPUT: "行 [2H, 3H)",
    GATE_CANDIDATE: "行 [3H, 4H)",
}

# --------------------------------------------------------------------------- #
# 闭合表 3：四张公式
# --------------------------------------------------------------------------- #

#: 朴素循环单元的那一行（本课全部内容的起点）.
RNN_RECURRENCE_FORMULA = "h_t = act(W_h·h_{t−1} + W_x·x_t + b)"

#: LSTM 的五条方程（**门的顺序就是权重行块的顺序**）.
LSTM_GATE_FORMULA = (
    "i = σ(W_xi·x_t + W_hi·h_{t−1} + b_i)；f = σ(…_f)；o = σ(…_o)；g = tanh(…_g)；"
    "c_t = f ⊙ c_{t−1} + i ⊙ g；h_t = o ⊙ tanh(c_t)"
)

#: 两种单元的参数量公式（**都不含 T**——这就是"时间轴上的权重共享"）.
RNN_PARAM_FORMULA = "params(rnn) = H·D + H·H + H = H·(D + H + 1)（与序列长度 T 无关）"
LSTM_PARAM_FORMULA = "params(lstm) = 4·(H·D + H·H + H) = 4H·(D + H + 1)（与序列长度 T 无关）"

#: BPTT 的那一条纪律（与 day090 的 "= 还是 +=" 是同一句话，只是对象换成了时间）.
BPTT_FORMULA = "dW ← Σ_{t=1..T} dz_t ⊗ x_t   （**累加**，不是赋值）"

#: 循环单元可用的激活名（**是 day089 那张 ACTIVATIONS 表的子集**）.
#: 刻意不含 ``softmax``：它是"按行归一成概率"的激活，作用在隐状态上会破坏
#: "每个单元一个状态"这件事；也不含 ``gelu``：它没有被用作循环激活的先例，
#: 而这一课要的是"最经典的两种"（tanh 是原版，relu 是常见的简化变体）。
RNN_ACTIVATIONS: tuple[str, ...] = ("tanh", "relu")

if not set(RNN_ACTIVATIONS) <= set(ACTIVATIONS):  # pragma: no cover - 导入期不变式
    raise RecurrentError(
        "循环单元可用的激活名必须是 day089 ACTIVATIONS 表的子集："
        "自造一个激活名会让它在 forward 里静默地走不到任何分支。"
    )

#: LSTM 的默认遗忘门偏置（**PyTorch / fastai 都把遗忘门偏置初始化成 1**）.
#: 理由很具体：``f = σ(b_f)``，b_f = 0 时 f ≈ 0.5，记忆每一步被砍掉一半；
#: b_f = 1 时 f ≈ 0.731，记忆通道一开始就是"基本连通"的——这是 LSTM 的第一课。
DEFAULT_FORGET_BIAS = 1.0

# --------------------------------------------------------------------------- #
# 闭合表 4：七条性质
# --------------------------------------------------------------------------- #

PROPERTY_RNN_CELL_MATCHES_MANUAL = "rnn_cell_matches_manual_recurrence"
PROPERTY_LSTM_GATES_CONTROL_STATE = "lstm_gates_control_state"
PROPERTY_WEIGHT_SHARING_ACROSS_TIME = "weight_sharing_across_time"
PROPERTY_SEQUENCE_ALIGNMENT = "sequence_alignment_matches_input"
PROPERTY_LSTM_CELL_GRADIENT_IS_FORGET_PRODUCT = "lstm_cell_gradient_is_forget_product"
PROPERTY_BPTT_MATCHES_NUMERICAL = "bptt_matches_numerical"
PROPERTY_LSTM_OUTLASTS_RNN_LONG_RANGE = "lstm_outlasts_rnn_on_long_range"

#: 七条性质（顺序 = 从"定义对不对"到"长程信号谁更强"）.
RECURRENT_PROPERTIES: tuple[str, ...] = (
    PROPERTY_RNN_CELL_MATCHES_MANUAL,
    PROPERTY_LSTM_GATES_CONTROL_STATE,
    PROPERTY_WEIGHT_SHARING_ACROSS_TIME,
    PROPERTY_SEQUENCE_ALIGNMENT,
    PROPERTY_LSTM_CELL_GRADIENT_IS_FORGET_PRODUCT,
    PROPERTY_BPTT_MATCHES_NUMERICAL,
    PROPERTY_LSTM_OUTLASTS_RNN_LONG_RANGE,
)

#: 每条性质在讲什么（一句话）.
PROPERTY_DESCRIPTIONS: dict[str, str] = {
    PROPERTY_RNN_CELL_MATCHES_MANUAL: "rnn_cell_forward 的一步与手写的那一行递推逐位一致",
    PROPERTY_LSTM_GATES_CONTROL_STATE: "i=0、f=1、o=1 时 LSTM 的记忆**原样冻结**（c_t = c_{t−1}）",
    PROPERTY_WEIGHT_SHARING_ACROSS_TIME: "T 步与 1 步用同一组权重：参数量与 T 无关，且第 t 步可单独复算",
    PROPERTY_SEQUENCE_ALIGNMENT: "T 个输入返回 T 个状态：h₀ 是初值，不出现在输出里",
    PROPERTY_LSTM_CELL_GRADIENT_IS_FORGET_PRODUCT: "∂c_T/∂c_0 = ∏f_t（记忆通道的梯度就是遗忘门之积）",
    PROPERTY_BPTT_MATCHES_NUMERICAL: "BPTT 的解析梯度（RNN 与 LSTM 各一次）与 day074 的数值差分一致",
    PROPERTY_LSTM_OUTLASTS_RNN_LONG_RANGE: "同一组设置下，LSTM 能原样把信号带回起点，而 tanh-RNN 指数衰减",
}

#: 每条性质"失败意味着什么"（**不通过时要去看哪里**）.
PROPERTY_FAILURE: dict[str, str] = {
    PROPERTY_RNN_CELL_MATCHES_MANUAL: "矩阵乘的转置被写反了（W·x 写成了 Wᵀ·x，形状仍对）",
    PROPERTY_LSTM_GATES_CONTROL_STATE: "门的**行块顺序**被写错了：i/f/o/g 与权重行块错位",
    PROPERTY_WEIGHT_SHARING_ACROSS_TIME: "每一步各造了一份权重（参数量随 T 增长，而报告里看不出）",
    PROPERTY_SEQUENCE_ALIGNMENT: "把初值 h₀ 也塞进了输出 ⇒ 全部状态整体错位一格",
    PROPERTY_LSTM_CELL_GRADIENT_IS_FORGET_PRODUCT: "c_t 的两个加项只回传了 i⊙g 那条链，漏掉了 f⊙c_{t−1}",
    PROPERTY_BPTT_MATCHES_NUMERICAL: "dW 用了 `=` 而不是 `+=`（只留下最后一步的贡献）",
    PROPERTY_LSTM_OUTLASTS_RNN_LONG_RANGE: "记忆通道与隐状态被写成了同一条（LSTM 退化成 RNN）",
}

if not (
    set(RECURRENT_PROPERTIES) == set(PROPERTY_DESCRIPTIONS) == set(PROPERTY_FAILURE)
):  # pragma: no cover
    raise RecurrentError("七条性质的三张表不一致：名单 / 说明 / 失败意味着什么必须逐键对齐。")

# --------------------------------------------------------------------------- #
# 闭合表 5：十条笔记 / 五条边界 / 一张 PyTorch 对照表
# --------------------------------------------------------------------------- #

NOTES_ORDER: tuple[str, ...] = (
    "time_sharing",
    "hidden_state",
    "initial_state",
    "alignment",
    "bptt_accumulate",
    "vanishing",
    "cell_state_highway",
    "forget_bias",
    "serial_versus_parallel",
    "project_position",
)

#: 十条笔记（键 -> 一句话）.
RECURRENT_NOTES: dict[str, str] = {
    "time_sharing": "循环的全部秘密是一句话：**同一组权重在时间轴的每一个时刻被重复使用**。",
    "hidden_state": "隐状态 h_t 是对「到目前为止的全部历史」的**有损摘要**——它的宽度是唯一的记忆预算。",
    "initial_state": "h₀ 是**初值**（通常是全零），不是输出；把它算进输出会让状态整体错位一格。",
    "alignment": "T 个输入 ⇒ T 个状态：`states[t]` 对应 `inputs[t]`，这条约定是反向能对齐的前提。",
    "bptt_accumulate": "BPTT 与 day090 是同一条纪律：梯度一律 `+=`；写成 `=` 只留下最后一步。",
    "vanishing": "tanh 的导数 ≤ 1，T 步连乘之后梯度指数衰减——这就是「记不住远处」的算术来源。",
    "cell_state_highway": "LSTM 的 c 是一条**加法**通道：c_t = f⊙c_{t−1} + i⊙g，梯度沿它走只乘 f。",
    "forget_bias": "遗忘门偏置初始化成 1（σ(1)≈0.731）：一开始就让记忆通道基本连通。",
    "serial_versus_parallel": "循环必须按时间顺序算（第 t 步依赖第 t−1 步）；注意力可以一次算完全序列。",
    "project_position": "本项目里循环的位置是**小规模序列**（短文本、日志、序列标注），大模型侧一律是 Transformer。",
}

#: 五条边界（**这一课明确不承诺的事**）.
RECURRENT_BOUNDARIES: tuple[str, ...] = (
    "只做单层单向的两个单元（rnn / lstm）：不做双向、不做多层堆叠、不做 GRU。",
    "不做批量（batch）维：一次前向处理一条序列（批量是工程上的循环，不是数学）。",
    "不做 GPU / 向量化：纯 Python 的逐时刻实现，为的是可读与可对账，不是速度。",
    "不做截断 BPTT：本课**展开到全长**（所有读数都在 T ≤ 8 的小序列上）。",
    "不新增第三方依赖：全部纯标准库实现，不 import torch / numpy。",
)

#: 纯 Python ↔ PyTorch 对照表（**只核对语义，本仓库不安装也不调用 torch**）.
TORCH_COUNTERPARTS: dict[str, str] = {
    "rnn_cell": "torch.nn.RNNCell(input_size, hidden_size, bias=True)（一步：h' = tanh(...)）",
    "lstm_cell": "torch.nn.LSTMCell(input_size, hidden_size)（一步：返回 (h', c')）",
    "rnn_sequence": "torch.nn.RNN(input_size, hidden_size, batch_first=False)（整段：返回全部时刻的 h）",
    "lstm_sequence": "torch.nn.LSTM(input_size, hidden_size)（整段：返回全部时刻的 h 与最后一步的 c）",
    "weight_sharing": "nn.RNN 内部只有一组权重：它被同一个循环体在 T 个时刻复用",
    "bptt": "loss.backward() + autograd（把 T 步的计算图一次反向；本课手写同一条链）",
    "grad_clipping": "torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)——爆炸的第一道闸",
    "forget_bias": "forget gate bias 初始化为 1（LSTM 的常见做法，本包 DEFAULT_FORGET_BIAS）",
    "sequence_classifier": "把最后一步的 h 送进 nn.Linear(hidden_size, classes)",
    "optimizer": "torch.optim.Adam(model.parameters(), lr)（day092 的同一套更新规则）",
}

#: 这一课的性质名单（供报告核对：表里的行数必须等于它）.
PROPERTIES = RECURRENT_PROPERTIES

__all__ = [
    "BPTT_FORMULA",
    "CELL_DESCRIPTIONS",
    "CELL_LSTM",
    "CELL_RNN",
    "CELL_STATE_SHAPES",
    "CELL_TYPES",
    "DEFAULT_FORGET_BIAS",
    "GATES",
    "GATE_ACTIVATIONS",
    "GATE_CANDIDATE",
    "GATE_DESCRIPTIONS",
    "GATE_FORGET",
    "GATE_INPUT",
    "GATE_OUTPUT",
    "GATE_ROW_BLOCKS",
    "LSTM_GATE_FORMULA",
    "LSTM_PARAM_FORMULA",
    "NOTES_ORDER",
    "PROPERTIES",
    "PROPERTY_BPTT_MATCHES_NUMERICAL",
    "PROPERTY_DESCRIPTIONS",
    "PROPERTY_FAILURE",
    "PROPERTY_LSTM_CELL_GRADIENT_IS_FORGET_PRODUCT",
    "PROPERTY_LSTM_GATES_CONTROL_STATE",
    "PROPERTY_LSTM_OUTLASTS_RNN_LONG_RANGE",
    "PROPERTY_RNN_CELL_MATCHES_MANUAL",
    "PROPERTY_SEQUENCE_ALIGNMENT",
    "PROPERTY_WEIGHT_SHARING_ACROSS_TIME",
    "RECURRENT_BOUNDARIES",
    "RECURRENT_NOTES",
    "RECURRENT_PROPERTIES",
    "RNN_ACTIVATIONS",
    "RNN_PARAM_FORMULA",
    "RNN_RECURRENCE_FORMULA",
    "TORCH_COUNTERPARTS",
]
