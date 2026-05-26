# RLPF xTB Reward Workflow (可迁移说明文档)

本文档说明 RLPF 中 xTB 奖励是如何计算并接入 RL/DDPO 训练循环的，方便你在其他模型中复用同一思路。

---

## 1. 在 RLPF 里，xTB 奖励由谁触发

训练入口在 `main_edm.py`：当参数 `--reward_type xtb` 时，构建 `ForceReward(dataset_info)` 作为奖励器，并传给 `DDPOTrainer`。  
即：`rollout -> rewarder -> trainer.update_policy` 的标准路径。  

关键代码位置：
- 选择奖励类型：`main_edm.py` 第 98-107 行。
- 传给 trainer：`main_edm.py` 第 115-124 行。

---

## 2. 输入数据结构（奖励器收到什么）

`EDMRollout.generate_minibatch()` 会返回 `DataProto`，其中至少包含：
- `x`: 生成的 3D 坐标
- `categorical`: 原子类型 one-hot
- `nodesxsample`: 每个样本真实原子数
- `group_index`, `latents`, `logps`, `timesteps` 等（RL/DDPO 需要）

关键代码位置：
- `edm_rollout.py` 第 129-140 行（DataProto 输出字段）。

---

## 3. xTB 奖励的逐步计算逻辑

奖励实现文件：`verl_diffusion/worker/reward/force.py`。

### 3.1 初始化

`ForceReward.__init__` 中：
- 建立 xTB 计算器：`XTB(method="GFN2-xTB")`
- 保存 `dataset_info`
- `atom_encoder = dataset_info['atom_decoder']`（把模型内部原子索引映射为元素符号）

关键代码位置：
- `force.py` 第 36-43 行。

---

### 3.2 从 DataProto 还原单分子

`process_data(samples)` 从 batch 中提取每个样本：
1. `categorical.argmax(1)` 得到原子类别索引
2. 用 `nodesxsample[i]` 截断 padding 部分
3. 形成 `(pos, atom_type)`，其中：
   - `pos`: `[n_atoms, 3]`
   - `atom_type`: `[n_atoms]`

关键代码位置：
- `force.py` 第 146-165 行。

---

### 3.3 单分子打分函数（Ray 远程）

`@ray.remote(num_cpus=8) calcuate_xtb_force(...)` 对每个分子做：
1. `check_stability(...)` 计算结构稳定性（返回的第一个值用于稳定分子标记）
2. 原子索引 -> 元素符号
3. 用 ASE `Atoms(symbols, positions)` 组装分子
4. 绑定 xTB 计算器并执行 `atoms.get_forces()`
5. 对力张量做 `rmsd(forces)`，得到 `mean_abs_forces`
6. 奖励定义：`reward = -1 * mean_abs_forces`
7. 异常容错：若 xTB 失败，`mean_abs_forces = 5.0`（等价 reward = -5.0）

关键代码位置：
- `force.py` 第 19-33 行。

> 结论：**RLPF 的 xTB 奖励本质是“最小化力大小”**（因为奖励为负力 RMSD，力越小奖励越高）。

---

### 3.4 批量并行与回收

`calculate_rewards(data)`：
1. 先 `process_data(data)` 得到分子列表
2. 每个分子提交一个 `ray` future 到 `calcuate_xtb_force.remote(...)`
3. `ray.get(futures)` 回收 `(reward, stability)` 列表
4. 组装成 `DataProto(batch=TensorDict(...))`，其中包含：
   - `rewards`: shape `[B]`
   - `stability`: shape `[B]`

关键代码位置：
- `force.py` 第 167-191 行。

---

## 4. 奖励如何进入策略更新

在 `DDPOTrainer` 中：
1. rollout 产生 `sample`（DataProto）
2. reward worker 线程调用 `self.rewarder.calculate_rewards(sample)`
3. 将 sample 与 reward 结果分别 concat 后做 union
4. `filters.filter(samples)` 可选去重/新颖性惩罚
5. `compute_advantage(samples)` 用 `samples.batch["rewards"]` 计算 group 内标准化优势
6. `actor.update_policy(samples)` 执行策略更新

关键代码位置：
- reward worker 调用：`ddpo_trainer.py` 第 82 行。
- sample/reward 合并：第 150-153 行。
- 过滤：第 267 行。
- 优势计算使用 rewards：第 169-191 行。
- 更新策略：第 269 行。

---

## 5. 迁移到“别的模型”时的最小接口要求

如果你要在其他生成模型里复用 xTB 奖励，建议遵循以下接口：

### 5.1 生成器输出（等价于 DataProto batch）
每个 batch 至少提供：
- `positions`: `[B, N, 3]`（或可恢复为此形状）
- `atom_type_onehot` 或 `atom_type_index`
- `n_atoms_per_sample`: `[B]`（去除 padding）

### 5.2 奖励器输入输出
- 输入：一个 batch（可迭代到单分子）
- 输出：至少 `rewards: [B]`
- 可选输出：`stability: [B]`（用于日志或联合目标）

### 5.3 单分子计算步骤复用
1. 截断真实原子
2. 原子类型映射为元素符号
3. 构建 `ase.Atoms`
4. `XTB(method="GFN2-xTB")` + `get_forces()`
5. `reward = -rmsd(forces)`
6. 异常给固定负值惩罚（如 -5）

---

## 6. 可直接复用的伪代码（框架无关）

```python
def xtb_reward_batch(batch):
    rewards, stability = [], []
    for mol in batch:
        pos, atom_idx, n = mol["pos"], mol["atom_idx"], mol["n_atoms"]
        pos = pos[:n]
        atom_idx = atom_idx[:n]
        stable = check_stability(pos, atom_idx, dataset_info)[0]
        try:
            symbols = [atom_decoder[i] for i in atom_idx]
            atoms = Atoms(symbols=symbols, positions=pos)
            atoms.calc = XTB(method="GFN2-xTB")
            forces = atoms.get_forces()
            r = -rmsd(forces)
        except Exception:
            r = -5.0
        rewards.append(r)
        stability.append(float(stable))
    return {"rewards": rewards, "stability": stability}
```

---

## 7. 实践建议（跨任务复用时）

1. **先做小 batch 烟雾测试**：确认每个分子都能正确转为 `Atoms`，避免训练中频繁异常回退。  
2. **并行策略**：RLPF 用 Ray；你也可以换成多进程池，但保持“一分子一任务 + 批量回收”思路。  
3. **异常惩罚值**：`-5.0` 是当前实现里的经验值，可根据任务分布调整。  
4. **可混合目标**：将 xTB 奖励与 QED/SA 等做加权和（如 `r = w_xtb*r_xtb + w_qed*r_qed - w_sa*sa`）。  
5. **日志监控**：至少记录 `reward mean/std` 与 `stability mean`，便于定位是否只学到“规避失败样本”。  

---

## 8. 代码锚点索引（便于快速跳转）

- 奖励入口选择：`RLPF-verl_diffusion/main_edm.py`
- xTB 奖励实现：`RLPF-verl_diffusion/verl_diffusion/worker/reward/force.py`
- rollout 数据打包：`RLPF-verl_diffusion/verl_diffusion/worker/rollout/edm_rollout.py`
- DDPO 主训练循环：`RLPF-verl_diffusion/verl_diffusion/trainer/ddpo_trainer.py`
- 过滤器（可选）：`RLPF-verl_diffusion/verl_diffusion/worker/filter/filter.py`

