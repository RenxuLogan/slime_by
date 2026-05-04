<div align="center">
  <h1>slime 学习型 README</h1>
  <p>一份适合边读源码边记录的学习笔记</p>
</div>

<hr/>

<div>
  <h2>1. 这份 README 怎么用</h2>
  <ul>
    <li>目标：不是重复官方文档，而是把 <code>slime</code> 的关键入口整理成“代码 + 作用 + 解释 + 踩坑”的学习手册。</li>
    <li>建议：先按本文的顺序阅读，再去对照源码和脚本做实验。</li>
    <li>维护方式：你后面每学到一个新点，就继续往下追加一个新的 <code>details</code> 模块。</li>
  </ul>
</div>

<div>
  <h2>2. 我对 slime 的当前理解</h2>
  <table>
    <tr>
      <th>模块</th>
      <th>职责</th>
      <th>我的理解</th>
    </tr>
    <tr>
      <td>training (Megatron)</td>
      <td>负责训练 Actor / Critic</td>
      <td>真正做参数更新的地方</td>
    </tr>
    <tr>
      <td>rollout (SGLang + router)</td>
      <td>负责生成回复、采样数据、计算奖励相关信息</td>
      <td>更像“数据生产端”</td>
    </tr>
    <tr>
      <td>data buffer</td>
      <td>负责在 rollout 和 training 之间传递样本</td>
      <td>更像“训练数据中转站”</td>
    </tr>
    <tr>
      <td>Ray</td>
      <td>负责资源编排和进程组织</td>
      <td>把训练和推理组件组织成一个分布式系统</td>
    </tr>
  </table>
</div>

<div>
  <h2>3. 推荐学习顺序</h2>
  <ol>
    <li>先看启动脚本：<code>scripts/run-qwen3-4B.sh</code></li>
    <li>再看模型参数：<code>scripts/models/qwen3-4B.sh</code></li>
    <li>再看训练主入口：<code>train.py</code></li>
    <li>最后看参数系统：<code>slime/utils/arguments.py</code></li>
  </ol>
</div>

<hr/>

<details open>
  <summary><strong>知识点 1：模型配置是怎么注入训练脚本的</strong></summary>

### 代码

来源：`scripts/models/qwen3-4B.sh`

```bash
MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-1000000}"
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)
```

### 作用

- 这段代码定义了 Qwen3-4B 在 Megatron 训练后端中需要的结构参数。

### 解释

- `MODEL_ARGS` 是一个 Bash 数组，后续会在启动训练时通过 `${MODEL_ARGS[@]}` 整体拼接到命令行。
- 这些参数不是“训练策略”，而是“模型结构说明书”。
- `--num-layers 36`、`--hidden-size 2560`、`--num-attention-heads 32` 共同描述 Transformer 主干。
- `--group-query-attention` 和 `--num-query-groups 8` 说明这个模型用了 GQA，而不是标准 MHA。
- `--rotary-base` 很关键，它影响 RoPE 的位置编码行为；同系列不同版本模型，这个值可能不同。
- `--normalization "RMSNorm"`、`--qk-layernorm` 表示归一化策略必须和原模型保持一致，否则即使权重能加载，训练行为也可能异常。

### 为什么这么写

- Megatron 训练时不能完全像 Hugging Face 一样只靠 checkpoint 自动恢复全部结构信息，所以启动脚本里通常要显式补齐模型结构超参数。
- 这也是 slime 把不同模型的配置拆到 `scripts/models/*.sh` 的原因：方便复用、切换和手动覆盖。

### 踩坑与思考

- `rotary-base` 配错，往往不是“立刻报错”，而是训练质量异常或推理行为怪异。
- 同一个模型名字，不同发布版本的结构细节也可能不完全一样，不能想当然复用旧脚本。
- 我后续需要养成一个习惯：先核对 Hugging Face `config.json`，再核对 `scripts/models/*.sh`。
</details>

<hr/>

<details open>
  <summary><strong>知识点 2：训练脚本其实是在组装参数分层</strong></summary>

### 代码

来源：`scripts/run-qwen3-4B.sh`

```bash
source "${SCRIPT_DIR}/models/qwen3-4B.sh"

CKPT_ARGS=(
   --hf-checkpoint /home/brx/models/Qwen3-4B-Instruct-2507
   --ref-load /home/brx/models/Qwen3-4B_torch_dist
   --load /home/brx/data/Qwen3-4B_slime/
   --save /home/brx/data/Qwen3-4B_slime/
   --save-interval 2
)

ROLLOUT_ARGS=(
   --prompt-data /home/brx/datasets/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 5
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1
   --global-batch-size 256
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)
```

### 作用

- 这段脚本把一次训练拆成“模型参数 / checkpoint / rollout / 性能优化”等几类参数，便于管理和调试。

### 解释

- `source "${SCRIPT_DIR}/models/qwen3-4B.sh"`：先把模型结构参数载入当前 Shell 环境。
- `CKPT_ARGS`：控制从哪里读取模型、从哪里继续训练、往哪里保存。
- `ROLLOUT_ARGS`：控制训练数据从哪来、如何采样、采样多少条、奖励怎么打。
- `PERF_ARGS`：控制并行策略和吞吐优化。

### 重点理解

- `--hf-checkpoint`：主要给 tokenizer 和模型配置用，不一定是当前最新训练权重。
- `--ref-load`：参考模型 checkpoint。很多 RL 训练逻辑都要拿当前策略和参考策略做比较，所以它非常关键。
- `--load`：Actor 的继续训练入口。如果这个目录不是有效 checkpoint，slime 会回退到 `--ref-load`。
- `--rollout-batch-size 32` 和 `--n-samples-per-prompt 8`：表示每轮先取 32 个 prompt，每个 prompt 采样 8 个回复，总共得到 256 个样本。
- `--global-batch-size 256`：说明这一轮 rollout 产出的 256 个样本，刚好都能被当前训练轮次消耗掉。
- `--use-dynamic-batch-size` + `--max-tokens-per-gpu 9216`：不是按“固定样本数”切 micro batch，而是按“每张卡承受的 token 总量”动态打包。

### 为什么这么写

- slime 的脚本风格非常工程化，不把所有参数揉成一行，而是分组组织。
- 这样做有三个好处：
- 更容易定位问题是出在“数据 / checkpoint / 并行 / 算法”哪一层。
- 更容易复制脚本改模型或改实验。
- 更适合后续沉淀成自己的实验模板。

### 踩坑与思考

- `rollout-batch-size * n-samples-per-prompt` 和 `global-batch-size * num-steps-per-rollout` 最好先手算一遍，不然容易出现“采样和训练消耗不匹配”的理解偏差。
- `--apply-chat-template` 只有在数据格式匹配时才该开，不能无脑打开。
- 启动脚本前半段有不少 `pkill` 和 `ray stop --force`，这是为了清理旧进程；自己复用脚本时要意识到它是偏“实验机脚本”而不是温和脚本。
</details>

<hr/>

<details open>
  <summary><strong>知识点 3：train.py 是 slime 的同步训练主循环</strong></summary>

### 代码

来源：`train.py`

```python
def train(args):
    configure_logger()
    pgs = create_placement_groups(args)
    init_tracking(args)

    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])
    router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
    update_tracking_open_metrics(args, router_addr)

    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    actor_model.update_weights()

    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))

        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

        if args.use_critic:
            value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
            if actor_trains_this_step:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
            else:
                ray.get(value_refs)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

        if args.offload_rollout:
            ray.get(rollout_manager.onload_weights.remote())
            actor_model.update_weights()
```

### 作用

- 这段代码定义了 slime 最核心的训练闭环：先 rollout 采样，再训练 actor/critic，再把新权重同步回 rollout 端。

### 解释

- `create_placement_groups(args)`：先向 Ray 申请资源布局，可以理解为“把训练和推理进程放到哪些 GPU 上”。
- `create_rollout_manager(...)`：创建 rollout 侧管理器，内部会拉起 SGLang 推理服务。
- `create_training_models(...)`：创建训练侧模型，通常是 actor，有时还包含 critic。
- `actor_model.update_weights()`：把训练侧的 actor 权重推给 rollout 端，保证采样使用的是当前策略。
- `rollout_manager.generate.remote(rollout_id)`：请求 rollout 端生成训练样本。
- `critic_model.async_train(...)`：如果当前算法需要 critic，就先训练 critic，得到 value 相关结果。
- `actor_model.async_train(...)`：再训练 actor。

### 核心链路

- 第一步：rollout 生成样本。
- 第二步：训练端消费这些样本。
- 第三步：把更新后的 actor 权重再次同步给 rollout。
- 这就是 RL 训练里非常典型的“采样 - 更新 - 再采样”闭环。

### 为什么这么写

- slime 把 rollout 和 train 解耦成不同组件，这样可以分别优化吞吐、资源占用和扩展方式。
- 训练和推理不是一个简单的 `for batch in dataloader` 关系，而是一个分布式系统协作关系。
- `ray.get(...)` 明确了哪些步骤必须等待完成，说明这里的主流程仍然是“同步轮次驱动”的。

### 踩坑与思考

- `actor_model.update_weights()` 不是可有可无，它直接决定 rollout 用的是不是最新策略。
- `args.use_critic` 为真时，训练链路会复杂很多，所以阅读代码时先区分自己当前实验到底是 GRPO 还是 PPO。
- 如果以后看 `train_async.py`，要重点对比它和 `train.py` 的区别：同步版本是“一轮 rollout 对一轮 train”，异步版本则会提前启动下一轮生成。
</details>

<hr/>

<details open>
  <summary><strong>知识点 4：参数系统是 slime 的扩展核心</strong></summary>

### 代码

来源：`slime/utils/arguments.py`

```python
def parse_args(add_custom_arguments=None):
    configure_logger()
    add_slime_arguments = get_slime_extra_args_provider(add_custom_arguments)

    pre = _pre_parse_mode()
    skip_sglang = pre.debug_train_only or pre.load_debug_rollout_data is not None

    sglang_ns = None
    if not skip_sglang:
        sglang_ns = sglang_parse_args()

    args = megatron_parse_args(
        extra_args_provider=add_slime_arguments,
        skip_hf_validate=pre.debug_rollout_only,
    )

    for key, value in vars(pre).items():
        setattr(args, key, value)

    if sglang_ns is not None:
        for key, value in vars(sglang_ns).items():
            setattr(args, key, value)

    slime_validate_args(args)
    return args
```

### 作用

- 这段代码负责把 Megatron、SGLang、slime 自己的参数系统统一到一个 `args` 对象里。

### 解释

- `_pre_parse_mode()`：先预解析少数几个“会影响后续解析流程”的参数。
- `sglang_parse_args()`：单独解析 SGLang 参数，因为它和 Megatron 的参数体系不是同一套。
- `megatron_parse_args(...)`：解析 Megatron 参数，同时通过 `extra_args_provider` 注入 slime 自己新增的大量参数。
- 最后把多套参数 merge 到一个 `args` 命名空间里，再统一做校验。

### 为什么这么写

- slime 不是从零写一个训练框架，而是把 Megatron 和 SGLang 拼成一个完整训练系统。
- 因为底层组件来自不同项目，所以参数解析天然是“多源输入”，必须有一层统一入口。
- 这也解释了为什么官方 README 里把参数分成三类：
- Megatron 参数
- SGLang 参数
- slime 自身参数

### 进一步看一个非常关键的校验逻辑

```python
if args.num_steps_per_rollout is not None:
    global_batch_size = args.rollout_batch_size * args.n_samples_per_prompt // args.num_steps_per_rollout
    if args.global_batch_size is not None:
        assert args.global_batch_size == global_batch_size
    args.global_batch_size = global_batch_size
```

### 这段校验的意义

- 它在自动维持 rollout 产出样本数和训练消耗样本数之间的平衡。
- 如果设置了 `--num-steps-per-rollout`，那 `global_batch_size` 就不是随便填的，而是受公式约束。
- 这个设计很像在强制你把“RL 数据闭环”想明白，而不是只会堆参数。

### 踩坑与思考

- slime 里很多参数不是独立的，而是“联动关系”，只记字面意思不够。
- 真正高效的学习方式不是背参数，而是先弄清：
- 这个参数属于哪一层？
- 它影响 rollout、training、还是资源编排？
- 它会不会和别的参数发生约束关系？
</details>

<hr/>

<details open>
  <summary><strong>知识点 5：同步版和异步版训练入口的区别</strong></summary>

### 代码

来源：`train_async.py`

```python
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    if rollout_data_next_future is not None:
        rollout_data_curr_ref = ray.get(rollout_data_next_future)

    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    if args.use_critic:
        value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
        if rollout_id >= args.num_critic_only_steps:
            ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
        else:
            ray.get(value_refs)
    else:
        ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
```

### 作用

- 这段代码展示了 slime 的异步训练思想：当前轮在训练时，下一轮 rollout 可以提前开始生成。

### 解释

- `rollout_data_next_future = rollout_manager.generate.remote(...)`：先把下一轮生成任务挂出去。
- 当前轮训练前，通过 `ray.get(...)` 取回上一次提前发出的 rollout 结果。
- 然后立刻再把下一轮 rollout 发出去，实现“生成和训练重叠”。

### 为什么重要

- 在 RL 场景里，rollout 常常比训练更慢，尤其是长回答、多轮交互、工具调用这些任务。
- 异步化的本质就是尽量减少 GPU 空转，提高整体流水线利用率。

### 踩坑与思考

- 异步不只是“更快”，还意味着更复杂的权重时序问题。
- 因此源码里会有 `update_weights_interval` 这类参数，控制多久同步一次权重。
- 学习时先把同步版 `train.py` 吃透，再来看异步版，理解成本会小很多。
</details>

<hr/>

<details open>
  <summary><strong>知识点 6：train.py 与 train_async.py 的核心区别</strong></summary>

### 核心区别一句话总结
- **`train.py` (同步版)**：严格串行。采样和训练像接力赛，SGLang 跑完 Megatron 才能跑。
- **`train_async.py` (异步版)**：流水线并行。Megatron 在训练第 $N$ 轮数据时，SGLang 已经在后台生成第 $N+1$ 轮的数据了。

### 详细对比

| 特性 | `train.py` (同步版) | `train_async.py` (异步版) |
| :--- | :--- | :--- |
| **执行时序** | 等待生成 -> 训练 -> 更新权重 -> 下一轮 | 拿上轮结果 -> **触发下轮生成** -> 训练本轮 -> 视情况更新权重 |
| **GPU 利用率** | 存在空窗期（生成时训练闲置，训练时生成闲置） | 极致压榨（生成和训练在不同卡上同时进行） |
| **硬件部署要求** | **支持 `colocate`**（生成和训练可共用同一批 GPU） | **不支持 `colocate`**（生成和训练必须独占各自的 GPU） |
| **策略新鲜度** | 每次生成用的都是刚更新的、绝对最新的策略权重 | 生成第 $N+1$ 轮数据时，Actor 还在训练第 $N$ 轮，存在策略滞后（Staleness） |

### 代码层面的直观差异

**1. 阻塞等待 vs 提前派发**
在 `train.py` 中，生成是阻塞等待的：
```python
# 必须干等着这轮生成完，才能往下走去训练
rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
```
在 `train_async.py` 中，是提前把下一轮的任务派发出去：
```python
# 先把第 N+1 轮的生成任务派发出去后台执行（不阻塞）
if rollout_id + 1 < args.num_rollout:
    rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

# 然后当前进程踏踏实实去训练第 N 轮的数据...
ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
```

**2. 权重同步频率**
- 同步版：每步必同步 (`actor_model.update_weights()`)。
- 异步版：引入了 `--update-weights-interval` 参数，允许你攒几步再把新权重通过网络发给 SGLang 推理引擎，进一步减少通信开销。

### 踩坑与思考
- **什么时候用同步？** 卡比较少，必须把 Megatron 和 SGLang 挤在同一批卡上交替执行（开启 `--colocate`）时；或者你的算法对“策略滞后”极度敏感，必须保证 on-policy 绝对纯正时。
- **什么时候用异步？** 算力充足，专门划了几张卡/几台机器做 Rollout，另几张卡/几台机器做 Train。此时开启异步，能直接掩盖掉漫长的文本生成时间，训练吞吐量起飞。
- **前置条件**：`train_async.py` 顶部的代码明确写了 `assert not args.colocate`，意味着你想跑异步流水线，就必须准备充足的资源让推理和训练物理隔离。
</details>

<hr/>

<details open>
  <summary><strong>知识点 7：init_tracking(args) 的作用</strong></summary>

### 代码位置
来源：`train.py` 中的 `init_tracking(args)`

### 作用一句话总结
初始化实验日志跟踪系统（Tracking System），主要是 **WandB (Weights & Biases)** 和 **TensorBoard**。

### 详细解释
在 RL 大模型训练中，你要监控非常多的指标（Loss、Reward、KL散度、生成长度、通过率等）。如果不记录下来，跑完根本不知道模型是在变好还是在崩溃。

当你看到 `train.py` 第 13 行的 `init_tracking(args)` 时，它其实是在做这几件事：
1. 检查你启动时有没有带上 `--use-wandb` 或者 `--use-tensorboard` 参数。
2. 如果带了，它会去初始化对应的服务（比如登录 WandB 账号，建一个新 Run，把所有 `--xxx` 的超参数都存下来当实验配置）。
3. 如果你在启动脚本（比如 `run-qwen3-4B.sh`）里没开这些参数，那这行代码就相当于什么也没做。

### 关联代码：后续的 `update_tracking_open_metrics`
紧接着在 `train.py` 的第 21 行，你还会看到一个相关的调用：
```python
router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
update_tracking_open_metrics(args, router_addr)
```
这是因为 slime 用了 SGLang 做推理生成，而 SGLang 自己有一套吐出性能指标（比如 KV Cache 命中率、吞吐量）的端点（metrics endpoint）。
这行代码的意思是：等 SGLang 服务启动后，把它的指标接口地址也喂给 WandB，让 WandB 把 SGLang 的底层推理性能数据一并收集上来。

### 踩坑与思考
- **卡死或者连不上网**：如果你在没有外网的机器上开了 `--use-wandb`，这行代码可能会一直卡住等待连接。解决方案是去掉 `--use-wandb`，或者把环境变量 `WANDB_MODE` 设为 `offline`。
- **为什么不直接用 `print`？** 因为 RL 训练的数据是海量的字典结构，且常常分布在多台机器上，用标准的 tracking 工具能直接画出折线图，这对于判断模型有没有“奖励黑客 (Reward Hacking)”行为至关重要。
</details>

<hr/>

<details open>
  <summary><strong>知识点 8：同步和异步模式下，权重同步时机的差异</strong></summary>

### 权重同步（Weight Sync）是什么？
在 RL 训练里，Megatron 负责更新策略模型（Actor）的权重，而 SGLang 负责用策略模型生成数据（Rollout）。
所以每训练完一段，Megatron 都需要把最新的权重通过网络同步给 SGLang，这就叫“权重同步”。对应代码里的 `actor_model.update_weights()`。

### 1. `train.py` (同步模式) 中的时机
**时机：每轮训练结束后，下一轮生成开始前。**

**代码过程**：
```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # 1. SGLang 生成数据 (阻塞等)
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
    
    # 2. Megatron 用这些数据训练
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
    
    # 3. 训练完立刻把新权重同步给 SGLang
    actor_model.update_weights()
```
**特点**：因为是串行的，所以 SGLang 每次生成前，都能保证拿到的是**绝对最新**的、刚刚更新完的策略。

### 2. `train_async.py` (异步模式) 中的时机
**时机：由 `--update-weights-interval` 参数控制，不一定是每轮都同步。**

**代码过程**：
```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # ...前面是并行训练和并行生成的代码...
    
    # 到了最后，检查是不是该同步权重了
    if (rollout_id + 1) % args.update_weights_interval == 0:
        # 为了防止 SGLang 正在生成一半时权重突然变了，必须先把后台的生成任务阻塞等完
        rollout_data_curr_ref = ray.get(x) if (x := rollout_data_next_future) is not None else None
        rollout_data_next_future = None
        
        # SGLang 停下手中的活后，再同步权重
        actor_model.update_weights()
```
**特点**：
1. **可以攒几轮再发**：为了极致压榨吞吐量，异步模式允许配置 `--update-weights-interval`。比如设为 5，就是训练了 5 轮才把权重发给 SGLang 一次。
2. **必须安全停车**：如果在 SGLang 生成到一半的时候强行推入新权重，会导致当前生成的回复前半段是旧脑子写的，后半段是新脑子写的。所以异步代码在同步权重前，会有一个小小的同步点（`ray.get(x)`），让 SGLang 先把手头的长篇大论写完，然后再接收新脑子。

### 踩坑与思考
- 在异步模式下，如果你把 `--update-weights-interval` 设得很大，训练速度会极快（因为网络通信少了），但 SGLang 用的策略就会非常旧（高 Staleness），可能会导致算法收敛变慢或者崩掉。
- 这个参数就像是在“系统吞吐量”和“算法收敛稳定性”之间找平衡。
</details>

<hr/>

<details open>
  <summary><strong>知识点 9：ray.get() 调用时机决定了同步还是异步</strong></summary>

### 核心结论
**同步和异步训练的区别，本质上在于 `ray.get()` 的调用时机。**

在 Ray 这个分布式框架里：
- `xxx.remote()` 是把任务派发出去，**它立刻返回一个凭证（Future/Ref），不阻塞当前进程**。
- `ray.get(凭证)` 是拿着凭证去拿结果，**如果任务没做完，它就会一直卡在这里死等（阻塞）**。

理解了这两句，再来看代码就很清晰了：

### 1. `train.py` (同步版)：当场派发，当场死等

在同步版里，代码是这样的：
```python
# train.py 
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    
    # 1. 把“生成数据”的任务派发给 SGLang，紧接着外面套了一层 ray.get()
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
    
    # 2. Megatron 必须等上面那行彻底执行完（拿到数据了），才能开始训练
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
```

**发生了什么？**
派发任务（`.remote()`）和拿结果（`ray.get()`）被写在了一行。Megatron 发出指令后，立刻自己把自己卡死，一直等到 SGLang 把所有长篇大论生成完，才继续往下走。
这就造成了接力赛：**SGLang 跑，Megatron 看；Megatron 跑，SGLang 看。**

---

### 2. `train_async.py` (异步版)：提前派发，秋后算账

在异步版里，代码变成了这样：
```python
# train_async.py
# 循环开始前，先把第 0 轮的生成任务派发出去，拿到一个凭证（future），但不套 ray.get()！
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

for rollout_id in range(args.start_rollout_id, args.num_rollout):
    
    # 1. 拿回上一轮（或初始派发）已经生成好的数据。
    # 因为 SGLang 已经在后台跑了很久，这里的 ray.get 几乎不用等，瞬间拿到。
    if rollout_data_next_future is not None:
        rollout_data_curr_ref = ray.get(rollout_data_next_future)

    # 2. 【核心精髓】：不等本轮训练开始，先把下一轮的生成任务派发给 SGLang！
    # 同样只拿凭证，不套 ray.get()，绝不卡死自己。
    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    # 3. 此时，SGLang 已经在后台哼哧哼哧地生成第 N+1 轮的数据了
    # 而 Megatron 可以拿着刚才瞬间取回的第 N 轮数据，安心开始自己的训练
    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
```

**发生了什么？**
派发任务（`.remote()`）和拿结果（`ray.get()`）被**错开**了！
- SGLang 在后台收到第 N+1 轮的 `.remote()` 指令后，立刻开始生成。
- Megatron 则用 `ray.get()` 等待自己第 N 轮的训练结束。
这就实现了流水线并行：**Megatron 在专心训练时，SGLang 同时在另一个车间疯狂生成下一批语料。** 完美掩盖了文本生成那漫长的时间。

</details>

<hr/>

<details open>
  <summary><strong>知识点 10：Ray Placement Group 与核心调度器</strong></summary>

### 代码位置
来源：`slime/ray/placement_group.py`

### 核心作用
这段代码是整个 slime 分布式系统的**资源大管家**。它负责向 Ray 集群申请物理 GPU，并把这些 GPU 严格、稳定地分配给底层的不同角色（Actor 模型、Critic 模型、Rollout 引擎）。

你可以把它理解为**“车间分配图”**。

### 核心概念解释

#### 1. Ray Placement Group (PG)
- **是什么**：在集群中预留一组 bundle（每个 bundle 包含 1 GPU + 1 CPU）。
- **为什么需要它**：如果只用普通的 `ray.remote()` 申请 GPU，节点上的卡可能会被乱序分配（比如 Rank 0 分到了卡 7，Rank 1 分到了卡 0）。使用 PG 和 `PACK` 策略，可以强制后续的分布式任务**严格按照物理卡的顺序**固定绑定。这对需要跑 NCCL/Megatron 这种底层极其依赖物理拓扑的训练框架来说，是保证性能和稳定的刚需。
- **怎么做的**：`_create_placement_group` 函数甚至还巧妙地生成了一批 `InfoActor`，提前探明每张卡的真实物理 ID 并做排序，保证后续分配不会错乱。

#### 2. RayTrainGroup (封装为 `actor_model` / `critic_model`)
- **是什么**：训练侧 Actor/Critic 的管理器。
- **怎么做的**：它通过 `_allocate_gpus_for_actor` 为每个训练 Rank 创建独立的 handler（`self._actor_handlers`）。
- **职责**：之后你在主循环里调用的 `actor_model.async_train()`、`update_weights()`、`offload()`，其实都是这个管理器在向底层几十上百个并行的训练进程同时下发指令。

#### 3. RolloutManager (封装为 `rollout_manager`)
- **是什么**：推理和数据侧的总指挥。
- **职责**：它不是简单的生成函数，而是一个宏大的编排器。它负责在分配到的 PG 资源上，拉起 **Rollout Engine (SGLang)**、**Data Buffer**、**Router** 等一系列组件。
- **联动**：它暴露了 `.generate.remote()` 等接口，让主循环能轻松地向后台索要生成的样本数据。

### 代码中的经典分配逻辑
```python
if args.colocate:
    # 资源复用模式：训练和生成挤在同一批卡上
    num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
    rollout_offset = 0  # rollout 和 actor 的起始位置都是 0
else:
    # 资源隔离模式：训练和生成分家
    num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
    rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node # rollout 往后排
```
这段代码完美解释了前面“同步版”和“异步版”的硬件部署差异：不开 `colocate` 时，它会向集群申请两波独立的卡，一波给 Train，一波给 Rollout。
</details>

<hr/>

<details open>
  <summary><strong>知识点 11：_create_placement_group 是如何保证 GPU 物理拓扑顺序的？</strong></summary>

### 代码位置
来源：`slime/ray/placement_group.py` 中的 `_create_placement_group(num_gpus)` 函数

### 这个函数的核心痛点是什么？
在写常规 Python 脚本时，我们默认 `CUDA_VISIBLE_DEVICES=0,1,2,3` 就是按顺序来的。
但是在 Ray 分布式集群中，**Ray 分配 GPU 的逻辑索引（bundle index）并不一定等于机器上的物理 GPU ID（PCIe 顺序）**。
如果把逻辑索引乱序的卡直接丢给 Megatron（或 NCCL），会导致底层跨卡通信（AllReduce/AllGather）没有走最快的 NVLink/NVSwitch，而是绕远路，导致训练性能暴跌甚至卡死。

### 这个函数是怎么解决的？（“探路兵”机制）

这个函数用了一个非常“黑客”但极其有效的办法来摸清底细：

1. **申请资源（圈地）**：
   ```python
   bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
   pg = placement_group(bundles, strategy="PACK")
   ray.get(pg.ready())
   ```
   先向 Ray 申请一块连续的资源（PACK 策略尽量保证都在同一台或相邻机器上）。此时拿到了一堆 bundle（逻辑坑位）。

2. **派出探路兵（InfoActor）**：
   ```python
   info_actors = []
   for i in range(num_bundles):
       info_actors.append(
           InfoActor.options(...).remote()
       )
   gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
   ```
   **这是最精彩的一步**。它在每个坑位（bundle）上强行启动了一个小小的探路兵进程（`InfoActor`）。
   探路兵进去后只干一件事：跑一段代码看看自己到底在哪台机器（IP）、被分配到了哪张物理卡（GPU ID），然后汇报给主控。

3. **过河拆桥（杀掉探路兵）**：
   ```python
   for actor in info_actors:
       ray.kill(actor)
   ```
   情报拿到手了，探路兵就没用了，立刻杀掉，把宝贵的 GPU 显存腾出来给后面的真神（Megatron / SGLang）。

4. **重新排序（建立物理映射）**：
   ```python
   bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
   sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
   pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
   pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]
   ```
   根据探路兵带回来的 IP 和物理 GPU ID，按照真实的物理顺序（先按 IP 排序，再按物理卡号排序）对那些逻辑坑位（bundle_indices）重新洗牌。

### 最终产出
函数返回了重新排序后的 `pg_reordered_bundle_indices`。
后续 `RayTrainGroup` 拉起真正的训练进程时，会按照这个**物理纠正后**的顺序去拿卡。这样，Rank 0 必定对应物理卡 0，Rank 1 对应物理卡 1，底层 NCCL 通信就能跑出满血带宽。

### 踩坑与思考
- **为什么不直接信任 Ray 的分配？** 因为 Ray 是个通用任务调度框架，不是专门为 NCCL 定制的。它在填坑时有自己的碎片填补逻辑，很容易造成乱序。
- 这种“先拉轻量级探测进程 -> 获取物理信息 -> 杀进程腾资源 -> 基于物理信息重新调度真任务”的做法，在很多对硬件拓扑敏感的大模型工程框架中（比如 vLLM 的多机调度、Megatron 的外围封装）是非常经典的手段。
</details>

<hr/>

<div>
  <h2>8. 当前我先记住的核心公式</h2>

```text
单轮 rollout 产出样本数
= rollout_batch_size * n_samples_per_prompt

单轮训练消耗样本数
= global_batch_size * num_steps_per_rollout
```

  <p><strong>关键理解：</strong>这两个量最好保持一致，否则你对“每轮 rollout 到底喂了几步训练”会很容易失去直觉。</p>
</div>

<div>
  <h2>9. 我当前的学习结论</h2>
  <ul>
    <li>slime 不是“单脚本训练器”，而是一个把 Ray、Megatron、SGLang 组合起来的 RL 后训练系统。</li>
    <li>启动脚本是最好的学习入口，因为它把模型、数据、算法、并行策略全部摊开了。</li>
    <li><code>train.py</code> 负责主循环，<code>arguments.py</code> 负责把复杂系统变成可配置系统。</li>
    <li>以后如果我要魔改 slime，最可能先改的是 rollout、reward、参数配置和启动脚本。</li>
  </ul>
</div>

<hr/>

<div>
  <h2>10. 踩坑记录区</h2>
  <p>后续每次遇到问题，就直接按下面格式追加。</p>
</div>

<table>
  <tr>
    <th>日期</th>
    <th>场景</th>
    <th>问题 / 报错</th>
    <th>原因分析</th>
    <th>解决方案</th>
  </tr>
  <tr>
    <td>2026-05-04</td>
    <td>第一次读源码</td>
    <td>参数很多，看不出主线</td>
    <td>直接从大文件入手，缺少入口感</td>
    <td>改成“启动脚本 - 主循环 - 参数系统”的顺序阅读</td>
  </tr>
</table>

<hr/>

<div>
  <h2>11. 后续可继续补的专题</h2>
  <ul>
    <li>自定义 Reward：<code>--custom-rm-path</code> 怎么接入</li>
    <li>自定义 Rollout：<code>--custom-generate-function-path</code> 怎么接入</li>
    <li>多轮对话 / Tool Calling：Search-R1 例子怎么读</li>
    <li>为什么要同时有 <code>train.py</code> 和 <code>train_async.py</code></li>
    <li>Checkpoint 转换链路：HF 和 Megatron 之间如何互转</li>
  </ul>
</div>

<hr/>

<div align="center">
  <p><strong>备注：</strong>这是一版“学习手册”起稿，后续建议继续按专题追加，而不是一次写成大而全文档。</p>
</div>
