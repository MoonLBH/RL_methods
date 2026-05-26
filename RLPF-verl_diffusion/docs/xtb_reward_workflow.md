# xTB 奖励构建说明（代码仓库无关版）

> 目标读者：将在**其他代码仓库**里实现“以 xTB 为目标的 RL 奖励”任务的人。  
> 本文档不依赖任何特定项目源码路径，不要求读者能访问当前仓库。

---

## 1) 奖励目标定义（先统一口径）

在分子生成/优化任务里，xTB 常见做法是把“力场残差（forces）”作为优化目标：

- 对每个分子计算原子受力向量 `F_i`
- 聚合为标量（常用 RMS）：
  \[
  s_{\text{force}}=\sqrt{\frac{1}{N}\sum_i ||F_i||^2}
  \]
- 强化学习奖励定义为负值：
  \[
  r_{\text{xtb}}=-s_{\text{force}}
  \]

这样“力越小，奖励越高”，优化方向清晰。

---

## 2) 你需要的最小输入/输出接口

无论你的生成模型是扩散、VAE、自回归还是图策略网络，xTB 奖励模块建议满足：

### 输入（每个 batch）
- `positions`: 每个分子的 3D 坐标（可还原为 `[B, N, 3]`）
- `atom_types`: 原子类型（index 或 one-hot）
- `num_atoms`: 每个样本的真实原子数（用于去 padding）

### 输出（每个 batch）
- `rewards`: shape `[B]`，每个分子的标量奖励
- （可选）`stability`: shape `[B]`，结构稳定性标记/分数
- （可选）`validity`: shape `[B]`，是否成功完成 xTB 计算

---

## 3) 单分子奖励计算流程（标准模板）

对 batch 中每个样本执行：

1. **裁剪真实原子**  
   用 `num_atoms[i]` 截断坐标和原子类型，去掉 padding。

2. **原子类型映射**  
   把模型内部原子 index 映射为元素符号（如 C/N/O/F/...）。

3. **构建分子对象**  
   使用 ASE `Atoms(symbols=..., positions=...)` 或等价结构。

4. **调用 xTB 计算器**  
   典型为 `GFN2-xTB`，执行 forces 计算。

5. **计算标量目标**  
   `force_score = RMS(forces)`（或你约定的 norm 聚合方式）。

6. **生成奖励**  
   `reward = -force_score`。

7. **异常处理**  
   若 xTB 失败（SCF 不收敛、输入非法等），返回固定惩罚（例如 `-5.0`）。

---

## 4) 批量并行建议（重点，决定速度）

xTB 单分子计算较慢，建议：

- **并行粒度**：一分子一个任务（进程/worker）
- **调度方式**：`ray` / `multiprocessing` / 作业队列均可
- **批量回收**：提交 futures 后统一回收结果，减少同步开销

实践上通常做两层控制：
- `max_workers`：并发 worker 数
- `timeout/retry`：单任务超时与重试策略

---

## 5) 数值鲁棒性与失败策略

必须明确 3 件事：

1. **失败惩罚值是多少**（例如 `-5.0`）  
2. **失败样本是否计入 replay/update**（建议计入，但带惩罚）  
3. **日志里单独统计失败率**（否则 reward 均值会误导）

建议日志至少包含：
- `reward_mean`, `reward_std`
- `xtb_fail_ratio`
- `stability_mean`（如果有）
- 每 step/epoch 的 wall time

---

## 6) 与多目标奖励融合（常见需求）

如果你还要联合 QED/SA/对接分数，可用线性加权：

\[
r = w_{\text{xtb}}\,r_{\text{xtb}} + w_{\text{qed}}\,r_{\text{qed}} - w_{\text{sa}}\,s_{\text{sa}} + ...
\]

经验建议：
- 先把各子目标做范围归一化（例如 z-score / min-max）
- 再调权重，否则某一项会数值主导训练

---

## 7) 代码无关伪代码（可直接迁移）

```python
def compute_xtb_rewards(batch, atom_decoder, xtb_calc, fail_penalty=-5.0):
    rewards = []
    validity = []
    for i in range(len(batch)):
        pos = batch.positions[i][: batch.num_atoms[i]]
        typ = batch.atom_types[i][: batch.num_atoms[i]]
        try:
            symbols = [atom_decoder[t] for t in typ]
            atoms = Atoms(symbols=symbols, positions=pos)
            atoms.calc = xtb_calc
            forces = atoms.get_forces()
            force_rms = rms(forces)         # sqrt(mean(||F_i||^2))
            r = -force_rms
            v = 1.0
        except Exception:
            r = fail_penalty
            v = 0.0
        rewards.append(r)
        validity.append(v)
    return rewards, validity
```

---

## 8) 交付给其他仓库时的“最小落地清单”

把下面 8 项给到对方，基本就能落地：

1. 原子 index -> 元素符号映射表  
2. batch 中真实原子数字段定义  
3. xTB 方法（通常 GFN2-xTB）  
4. force 聚合方式（RMS / mean norm）  
5. 奖励符号（负号）  
6. 失败惩罚值  
7. 并行策略与 worker 数  
8. 训练日志指标定义

---

## 9) 常见坑（迁移时高频）

- 坐标单位/格式不一致（Å vs 其他）  
- 原子类型映射错位（index 对不上元素）  
- padding 未截断导致伪原子进入 xTB  
- 并发太高导致 CPU/RAM 爆掉  
- 把异常吞掉但不统计失败率，训练看似“稳定”实则无效

---

## 10) 一句话总结

**把每个分子的 xTB 力 RMS 作为代价并取负作为奖励，再配合失败惩罚与并行计算，就是最稳妥、可迁移的 xTB 目标奖励实现。**

