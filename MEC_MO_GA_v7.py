"""
=============================================================================
MEC_MO_GA_v3.py — [纯多目标无约束自由进化版] 选址定容演化内核
=============================================================================
优化重点：
  1. 响应用户最高指示：彻底移除一切形式的 ε-约束红线与断层式大数惩罚。
  2. 强制将所有个体的违规度归零 (CV = 0.0)，恢复纯粹的帕累托两维目标梯度寻优。
  3. 保留代际历史 Archive 快照收集，全力支持收敛轨迹曲线输出。
=============================================================================
"""

import os
import json
import numpy as np
import time

try:
    import Fitness_Function_v3 as ff
except ImportError:
    ff = None

# ============================================================================
# 1. 全局变量声明
# ============================================================================
Pop_Size = None
Max_Gen = None
Loop_Times = None
Cross_Rate = None
Mut_Rate = None

MainPop = None
Archive = None
Archive_F = None
Archive_CV = None

MainFit = None
MainCV = None

Fronts = None
Ranks = None
CrowdDist = None

Stag_Counter = 0
Stag_Threshold = 40

Coords = Demands = Grid_Caps = Dist_Matrix = None
Num_Nodes = 0
Fit_Params_MO = {}


# ============================================================================
# 2. 全矩阵非支配排序 (2目标特化, O(N log N))
# ============================================================================

def non_dominated_sort(F, cv):
    """2目标 Kung 算法: 排序+扫描, 每层 O(N log N)"""
    N = F.shape[0]
    fronts, ranks = [], np.full(N, -1, dtype=int)
    remaining = np.ones(N, dtype=bool)

    while np.any(remaining):
        idx = np.where(remaining)[0]
        Fi = F[idx]
        si = np.lexsort((Fi[:, 1], Fi[:, 0]))
        sorted_idx = idx[si]; sorted_f = F[sorted_idx]

        # 扫描: 累计最小 F2, 第一个非支配层
        min_f2 = np.minimum.accumulate(sorted_f[:, 1])
        f1 = sorted_f[:, 0]
        f1_shift = np.concatenate([[f1[0] - 1], f1[:-1]])
        first_in_f1 = f1 != f1_shift
        is_front = (sorted_f[:, 1] <= min_f2) & first_in_f1

        fm = sorted_idx[is_front]
        fronts.append(fm)
        ranks[fm] = len(fronts) - 1
        remaining[fm] = False

    return fronts, ranks


def crowding_distance(F, front_indices):
    """全矩阵拥挤距离"""
    if len(front_indices) <= 2:
        return np.full(len(front_indices), np.inf)
    Ff = F[front_indices]; n = len(front_indices)
    cd = np.zeros(n)
    for m in range(Ff.shape[1]):
        si = np.argsort(Ff[:, m]); sf = Ff[si, m]
        rng = sf[-1] - sf[0]
        if rng < 1e-10: continue
        cd[si[0]] = np.inf; cd[si[-1]] = np.inf
        cd[si[1:-1]] += (sf[2:] - sf[:-2]) / rng
    return cd


def binary_tournament_selection():
    """全矩阵锦标赛选择"""
    global MainPop, MainFit, MainCV, Fronts, Ranks, CrowdDist
    N = MainPop.shape[0]

    Fronts, Ranks = non_dominated_sort(MainFit, MainCV)
    CrowdDist = np.zeros(N)
    for fr in Fronts:
        if len(fr) > 0:
            CrowdDist[fr] = crowding_distance(MainFit, fr)

    a = np.random.randint(0, N, size=N)
    b = np.random.randint(0, N, size=N)
    ra, rb = Ranks[a], Ranks[b]
    ca, cb = CrowdDist[a], CrowdDist[b]
    pick = np.where(ra < rb, a, np.where(ra > rb, b, np.where(ca > cb, a, b)))

    MainPop = MainPop[pick].copy()
    MainFit = MainFit[pick].copy()
    MainCV = MainCV[pick].copy()


# ============================================================================
# 3. 矩阵化并行演化算子
# ============================================================================
def crossover(pop):
    """
    【纯净矩阵化】全局均匀交叉算子
    """
    global Pop_Size, Num_Nodes, Cross_Rate
    indices2 = np.random.permutation(Pop_Size)
    p2_pop = pop[indices2]

    trigger_cross = np.random.rand(Pop_Size, 1, 1) < Cross_Rate
    uniform_masks = np.random.rand(Pop_Size, Num_Nodes) < 0.5

    final_full_mask = np.zeros((Pop_Size, 2, Num_Nodes), dtype=bool)
    final_full_mask[:, 0, :] = uniform_masks
    final_full_mask[:, 1, :] = uniform_masks

    actual_exchange_mask = trigger_cross & final_full_mask
    pop[:] = np.where(actual_exchange_mask, p2_pop, pop)


def mutation(pop):
    """
    【MEA 纯矩阵效率优化版】自适应双极有向/随机混合变异算子
    严格执行纯 2D 矩阵平级广播，100% 根除 3D/4D 混级高级索引 Bug。
    """
    global Pop_Size, Num_Nodes, Mut_Rate, Grid_Caps, MainFit

    # 建立 2D 干净缓存切片 (Pop_Size, Num_Nodes)
    status_2d = pop[:, 0, :].copy()
    caps_2d = pop[:, 1, :].copy()

    # 1. 基础变异触发概率掩码 (Pop_Size, Num_Nodes)
    mut_trigger = np.random.rand(Pop_Size, Num_Nodes) < Mut_Rate

    # 2. 区分 50% 随机个体 与 50% 有向个体
    mode_rand = np.random.rand(Pop_Size, 1) < 0.5  # 50% 概率走全局随机

    # 计算当前归档/种群的时间中位数，识别个体所处目标象限
    median_time = np.median(MainFit[:, 1])
    # 识别出不满意度过高（车主时间大于中位数、位于左上角）的退化个体掩码 (Pop_Size, 1)
    high_dissatisfaction_mask = (MainFit[:, 1] > median_time)[:, np.newaxis]

    # --- 随机路径：一枪生成全矩阵纯随机翻转和定容值 ---
    rand_flipped_status = 1 - status_2d
    rand_piles_pool = np.random.randint(1, np.maximum(2, Grid_Caps[np.newaxis, :]))

    # --- 有向路径：根据用户满意度情况选择变异方向 ---
    directional_rand = np.random.rand(Pop_Size, Num_Nodes)

    # 针对不满意度偏高（左上角）的个体：变异方向必须是“有向开启、批量堆桩以挽救满意度”
    dir_high_status = np.where(directional_rand < 0.70, 1, 1 - status_2d)  # 70% 概率定向强开
    dir_high_caps = np.clip(caps_2d + 2, 1, Grid_Caps[np.newaxis, :])  # 追加 2 个桩，挽救时间

    # 针对已经充分满足需求的解：变异方向是“定向关站、降产减桩”
    dir_low_status = np.where(directional_rand < 0.65, 0, 1 - status_2d)  # 65% 概率定向关闭
    dir_low_caps = np.maximum(1, caps_2d - 1)  # 缩减桩数

    # 融合有向阵营分支
    biased_status = np.where(high_dissatisfaction_mask, dir_high_status, dir_low_status)
    biased_caps = np.where(high_dissatisfaction_mask, dir_high_caps, dir_low_caps)

    # 3. 最终融合：50% 纯随机 vs 50% 有向自适应
    final_mut_status = np.where(mode_rand, rand_flipped_status, biased_status)
    final_mut_caps = np.where(mode_rand, rand_piles_pool, biased_caps)

    # 4. 执行掩码改写，确保在纯 2D 矩阵层面对齐
    status_2d[:] = np.where(mut_trigger, final_mut_status, status_2d)
    caps_2d[:] = np.where(mut_trigger & (status_2d == 1), final_mut_caps, caps_2d)

    # 🔒 物理安全底线清洗
    caps_2d[status_2d == 0] = 0
    caps_2d[:] = np.minimum(caps_2d, Grid_Caps[np.newaxis, :])

    # 写回大种群
    pop[:, 0, :] = status_2d
    pop[:, 1, :] = caps_2d





def repair(population):
    """移除邻域引力分流，仅保留基础开站保底，让种群自由跨越目标曲面"""
    population[:, 1, :] = np.where(population[:, 0, :] == 1, np.maximum(population[:, 1, :], 1), 0)


# ============================================================================
# 4. 纯双目标无约束并行评估算子
# ============================================================================

def evaluate_population(population):
    N = population.shape[0]
    F = np.zeros((N, 2))
    CV = np.zeros(N)
    # 缓存全局变量避免每次查global
    dm, dmds, gc, fp = Dist_Matrix, Demands, Grid_Caps, Fit_Params_MO

    for i in range(N):
        ind_copy = population[i].copy()
        ind_copy[0] = (ind_copy[0] >= 0.5).astype(int)
        _, details = ff.fitness_function(ind_copy, dm, dmds, gc, fp)
        F[i, 0] = details['BuildCost']
        F[i, 1] = details.get('TravelTime', 0.0) + details.get('WaitingTime', 0.0)
        CV[i] = 0.0

    return F, CV


def update_archive(MainPop, MainFit, MainCV=None):
    """
    全矩阵并行化 Archive 归档更新函数（无循环、高能去重与非支配筛选版）

    参数:
      MainPop: 当前代种群矩阵，形状为 (Pop_Size, 2, Num_Nodes)
      MainFit: 当前代自适应得分/目标值矩阵，形状为 (Pop_Size, 2)
      MainCV: 违规度矩阵（自由进化版通常为全0或None，此处做兼容处理）
    """
    global Archive, Archive_F, Archive_CV, Pop_Size

    # 1. 如果 Archive 为空，直接用当前代进行初始化筛选
    if Archive is None or len(Archive) == 0:
        # 首先对当前代进行非支配排序筛选第一层
        fronts, _ = non_dominated_sort(MainFit, MainCV)
        best_idx = fronts[0]  # 获取第一帕累托前沿的索引

        # 决策空间去重（防止相同配置写入）
        _, unique_indices = np.unique(MainPop[best_idx], axis=0, return_index=True)
        final_idx = best_idx[unique_indices]

        Archive = MainPop[final_idx].copy()
        Archive_F = MainFit[final_idx].copy()
        if MainCV is not None:
            Archive_CV = MainCV[final_idx].copy()
        else:
            Archive_CV = np.zeros(len(final_idx))
        return

    # 2. 矩阵级合并：将当前的 Archive 与新一代的种群直接拼接
    combined_pop = np.concatenate([Archive, MainPop], axis=0)
    combined_fit = np.concatenate([Archive_F, MainFit], axis=0)
    if MainCV is not None:
        combined_cv = np.concatenate([Archive_CV, MainCV], axis=0)
    else:
        combined_cv = np.zeros(len(combined_pop))

    # 3. 决策空间去重（Decision Space De-duplication）
    # 科学计算中，完全相同的建桩方案在目标空间会重叠，必须在决策空间用 unique 强行剔除
    # 将 (N, 2, Num_Nodes) 展平为 (N, 2 * Num_Nodes) 以适配 np.unique 的 axis=0
    flat_pop = combined_pop.reshape(combined_pop.shape[0], -1)
    _, unique_idx = np.unique(flat_pop, axis=0, return_index=True)

    # 提取去重后的合并集合
    combined_pop = combined_pop[unique_idx]
    combined_fit = combined_fit[unique_idx]
    combined_cv = combined_cv[unique_idx]

    # 4. 全矩阵非支配筛选（Pareto Front Filtering）
    # 调用高效的矩阵化非支配排序，只把真正互相不支配的“绝对前沿解”留下来
    fronts, _ = non_dominated_sort(combined_fit, combined_cv)
    pareto_idx = fronts[0]  # 永远只保留最优的第一层

    # 5. 归档容量防御控制（防止 Archive 规模无限膨胀导致后期排序变慢）
    # 如果筛选出来的帕累托解集超过了种群规模的 2 倍（或自定义上限），利用拥挤度截断
    max_archive_size = int(Pop_Size/2)
    if len(pareto_idx) > max_archive_size:
        fit_to_crop = combined_fit[pareto_idx]

        # 计算拥挤度 (Crowding Distance) 的并行化实现
        obj_min = fit_to_crop.min(axis=0)
        obj_max = fit_to_crop.max(axis=0)
        denom = obj_max - obj_min
        denom[denom == 0] = 1.0  # 防止除以 0

        # 对两个目标分别进行排序并计算空间跨度距离
        sort_idx_obj1 = np.argsort(fit_to_crop[:, 0])
        sort_idx_obj2 = np.argsort(fit_to_crop[:, 1])

        distances = np.zeros(len(pareto_idx))
        # 边界个体赋予无限大拥挤度，确保极端优秀解不丢失
        distances[sort_idx_obj1[0]] = distances[sort_idx_obj1[-1]] = np.inf
        distances[sort_idx_obj2[0]] = distances[sort_idx_obj2[-1]] = np.inf

        # 内部个体计算前后邻居的距离差
        distances[sort_idx_obj1[1:-1]] += (fit_to_crop[sort_idx_obj1[2:], 0] - fit_to_crop[sort_idx_obj1[:-2], 0]) / \
                                          denom[0]
        distances[sort_idx_obj2[1:-1]] += (fit_to_crop[sort_idx_obj2[2:], 1] - fit_to_crop[sort_idx_obj2[:-2], 1]) / \
                                          denom[1]

        # 按拥挤度从大到小降序排列，截取前 max_archive_size 个
        keep_sub_idx = np.argsort(-distances)[:max_archive_size]
        pareto_idx = pareto_idx[keep_sub_idx]

    # 6. 写回全局 Archive 变量
    Archive = combined_pop[pareto_idx].copy()
    Archive_F = combined_fit[pareto_idx].copy()
    Archive_CV = combined_cv[pareto_idx].copy()

# def adaptive_control():
#     global Cross_Rate, Mut_Rate, MainPop, MainFit, Stag_Counter
#     X = MainPop[:, 0, :]
#     centroid = np.mean(X, axis=0)
#     dist_vector = np.linalg.norm(X - centroid, axis=1)
#     max_d, mean_d = np.max(dist_vector), np.mean(dist_vector)
#     I_spatial = mean_d / (max_d + 1e-8) if max_d > 0 else 0.0
#
#     F1 = MainFit[:, 0]
#     f_min, f_max = np.min(F1), np.max(F1)
#     I_temporal = (np.mean(F1) - f_min) / (f_max - f_min + 1e-8) if f_max > f_min else 0.0
#     I_stagnation = min(1.0, Stag_Counter / Stag_Threshold)
#
#     omega = 0.4 * I_spatial + 0.3 * (1 - I_temporal) + 0.3 * (1 - I_stagnation)
#     Cross_Rate = np.clip(0.40 + 0.50 * omega, 0.40, 0.90)
#     Mut_Rate = np.clip(0.20 - 0.15 * omega, 0.05, 0.20)


# ============================================================================
# 4.5 空间-时间自适应参数控制器
# ============================================================================
hv_history_window = []


def adaptive_control():
    """
    【严谨学术版】空间-时间-停滞自适应参数控制器
    修正：严格基于归档精英 Archive_F 计算多目标超容量 (HV) 的滑动窗口提升率。
    """
    global Cross_Rate, Mut_Rate, MainPop, MainFit, Stag_Counter, hv_history_window
    global Archive, Archive_F

    # ------------------------------------------------------------------------
    # 1. 空间多样性指标 I_spat
    # ------------------------------------------------------------------------
    X = MainPop[:, 0, :]
    centroid = np.mean(X, axis=0)
    dist_vector = np.linalg.norm(X - centroid, axis=1)
    max_d = np.max(dist_vector)
    mean_d = np.mean(dist_vector)
    I_spat = mean_d / (max_d + 1e-8) if max_d > 0 else 0.0

    # ------------------------------------------------------------------------
    # 2. 时间收敛性指标 I_temp (严格基于 Archive 积分，消灭劣解污染)
    # ------------------------------------------------------------------------
    w = 10  # 窗口尺寸
    eps_val = 1e-8

    # 防御性保护：如果优化刚开始，Archive 尚未建立，直接给予 0.0 初始值
    if Archive_F is None or len(Archive_F) == 0:
        current_hv = 0.0
    else:
        # 【核心修正】归一化基准必须来源于 Archive 本身，保证绝对前沿的度量尺度稳定
        f_min = Archive_F.min(axis=0)
        f_max = Archive_F.max(axis=0)
        denom = f_max - f_min
        denom[denom == 0] = 1.0

        # 仅对归档中的非支配精英解进行归一化映射
        normalized_archive = (Archive_F - f_min) / denom

        # 2目标阶梯面积积分，以 (1.1, 1.1) 为安全极值边界点
        si = np.argsort(normalized_archive[:, 0])
        sorted_p = normalized_archive[si]

        current_hv = 0.0
        prev_x = 0.0
        for idx in range(len(sorted_p)):
            cur_x = sorted_p[idx, 0]
            cur_y = sorted_p[idx, 1]
            if cur_x <= 1.1 and cur_y <= 1.1:
                current_hv += (cur_x - prev_x) * (1.1 - cur_y)
                prev_x = cur_x

    hv_history_window.append(current_hv)
    if len(hv_history_window) > w + 1:
        hv_history_window.pop(0)

    if len(hv_history_window) >= 2:
        hv_t = hv_history_window[-1]
        hv_t_w = hv_history_window[0]
        I_temp = (hv_t - hv_t_w) / max(hv_t_w, eps_val)
        I_temp = np.clip(I_temp, 0.0, 1.0)
    else:
        I_temp = 0.0

    # ------------------------------------------------------------------------
    # 3. 停滞度指标 I_stag (Stagnation)
    # ------------------------------------------------------------------------
    G_stag = 20  # 停滞容忍阈值
    I_stag = min(1.0, Stag_Counter / G_stag)

    # ------------------------------------------------------------------------
    # 4. 指标融合与解耦反向控制映射 (论文 Equation 19, 20)
    # ------------------------------------------------------------------------
    c1, c2, c3 = 0.4, 0.3, 0.3
    I_composite = c1 * I_spat + c2 * (1 - I_temp) + c3 * (1 - I_stag)
    I_composite = np.clip(I_composite, 0.0, 1.0)

    p_cmin, p_cmax = 0.40, 0.90
    p_mmin, p_mmax = 0.05, 0.20

    Cross_Rate = p_cmin + (p_cmax - p_cmin) * I_composite
    Mut_Rate = p_mmax - (p_mmax - p_mmin) * I_composite

def adaptive_control11111():
    """
    Spatial-temporal adaptive parameter controller
    Consistent with the paper formulation.
    """
    global Cross_Rate, Mut_Rate, MainPop, MainFit, Stag_Counter, hv_history_window
    global Archive, Archive_F

    # ------------------------------------------------------------------------
    # 1. Spatial diversity indicator I_spat
    # ------------------------------------------------------------------------
    X = MainPop[:, 0, :]
    centroid = np.mean(X, axis=0)
    dist_vector = np.linalg.norm(X - centroid, axis=1)
    max_d = np.max(dist_vector)
    mean_d = np.mean(dist_vector)
    I_spat = mean_d / (max_d + 1e-8) if max_d > 0 else 0.0
    I_spat = np.clip(I_spat, 0.0, 1.0)

    # ------------------------------------------------------------------------
    # 2. Temporal convergence indicator I_temp
    # ------------------------------------------------------------------------
    w = 10          # sliding window size
    eps_val = 1e-8

    if Archive_F is None or len(Archive_F) == 0:
        current_hv = 0.0
    else:
        f_min = Archive_F.min(axis=0)
        f_max = Archive_F.max(axis=0)
        denom = f_max - f_min
        denom[denom == 0] = 1.0
        normalized_archive = (Archive_F - f_min) / denom

        # 2-objective hypervolume with reference point (1.1, 1.1)
        si = np.argsort(normalized_archive[:, 0])
        sorted_p = normalized_archive[si]

        current_hv = 0.0
        prev_x = 0.0
        for idx in range(len(sorted_p)):
            cur_x = sorted_p[idx, 0]
            cur_y = sorted_p[idx, 1]
            if cur_x <= 1.1 and cur_y <= 1.1:
                current_hv += (cur_x - prev_x) * (1.1 - cur_y)
                prev_x = cur_x

    hv_history_window.append(current_hv)
    if len(hv_history_window) > w + 1:
        hv_history_window.pop(0)

    if len(hv_history_window) >= 2:
        hv_t = hv_history_window[-1]
        hv_t_w = hv_history_window[0]
        I_temp = (hv_t - hv_t_w) / max(hv_t_w, eps_val)
        I_temp = np.clip(I_temp, 0.0, 1.0)   # negative values clipped to 0
    else:
        I_temp = 0.0

    # ------------------------------------------------------------------------
    # 3. Stagnation indicator I_stag
    # ------------------------------------------------------------------------
    G_stag = 20
    I_stag = min(1.0, Stag_Counter / G_stag)

    # ------------------------------------------------------------------------
    # 4. Composite indicator (consistent with paper)
    # I = [I_spat + (1 - I_temp) + I_stag] / 3
    # ------------------------------------------------------------------------
    I_composite = (I_spat + (1.0 - I_temp) + I_stag) / 3.0
    I_composite = np.clip(I_composite, 0.0, 1.0)

    # Parameter update (consistent with paper)
    p_cmin, p_cmax = 0.40, 0.90
    p_mmin, p_mmax = 0.05, 0.20

    # High I → high mutation, low crossover (exploration)
    Mut_Rate   = p_mmin + (p_mmax - p_mmin) * I_composite
    Cross_Rate = p_cmax - (p_cmax - p_cmin) * I_composite


def post_processing_refine():
    """纯自由帕累托对撞，不执行人为确定的后处理容量重写，将配置决定权完完全全还给演化算子本身"""
    global Archive, Archive_F, Archive_CV
    if Archive is None or len(Archive) == 0: return
    Archive_F, Archive_CV = evaluate_population(Archive)


def load_dataset(filename):
    global Coords, Demands, Grid_Caps, Dist_Matrix, Num_Nodes
    with open(filename) as f:
        data = json.load(f)
    Coords = np.array([[n['x'], n['y']] for n in data['nodes']])
    Demands = np.array([n['demand'] for n in data['nodes']])
    Grid_Caps = np.array([n['grid_capacity'] for n in data['nodes']])
    Num_Nodes = len(Coords)
    Dist_Matrix = np.sum(np.abs(Coords[:, np.newaxis, :] - Coords[np.newaxis, :, :]), axis=2)


# ============================================================================
# 5. 主主演化控制内核
# ============================================================================
def run(problem_file, pop_size=100, max_gen=300, loop_times=1, verbose=True):
    global Pop_Size, Max_Gen, Loop_Times, Cross_Rate, Mut_Rate
    global MainPop, MainFit, MainCV, Archive, Archive_F, Archive_CV, Stag_Counter, Fronts

    Pop_Size, Max_Gen, Loop_Times = pop_size, max_gen, loop_times
    global Fit_Params_MO
    Fit_Params_MO = {
        'w_cost': 0.5, 'w_service': 0.5, 'C_fixed': 50000.0, 'C_pile': 5000.0,
        'Service_Rate': 10.0, 'Max_Dist': 30.0, 'P_Grid': 1e6, 'P_Coverage': 1e7,
        'beta_dist': 0.1, 'gamma_wait': 0.05, 'C_wait': 500.0, 'P_Unstable': 1e8,
        'alpha': 0.5, 'Travel_Speed': 30.0
    }

    load_dataset(problem_file)
    all_pareto_fronts, all_times = [], []
    all_loop_histories = []

    # 预计算全局空间吸引力势能矩阵
    attraction_matrix = Demands[:, np.newaxis] / (Dist_Matrix + 1.0)
    node_potential = np.sum(attraction_matrix, axis=0)

    for loop in range(loop_times):
        t0 = time.time();
        Stag_Counter = 0
        Archive = Archive_F = Archive_CV = None
        run_generation_history = []

        # ... 初始化代码不变 ...
        MainPop = np.zeros((Pop_Size, 2, Num_Nodes))
        init_status = np.zeros((Pop_Size, Num_Nodes))
        init_caps = np.zeros((Pop_Size, Num_Nodes))
        valid_nodes = np.where(Grid_Caps > 0)[0]
        half_pop = Pop_Size // 2
        # 基于供需比算基础站点数
        total_demand = np.sum(Demands)
        avg_serve = np.mean(Grid_Caps[valid_nodes]) * Fit_Params_MO['Service_Rate'] * 0.3
        base_stations = max(5, int(total_demand / max(1, avg_serve)))
        max_open = min(len(valid_nodes), int(base_stations * 2.5))
        # 阵营 A: 重资产 (1.5x~2.5x 基础站)
        for i in range(half_pop):
            ratio = 1.5 + (i / max(1, half_pop - 1))
            n_open = min(len(valid_nodes), max(base_stations, int(base_stations * ratio)))
            chosen = np.random.choice(valid_nodes, n_open, replace=False)
            init_status[i, chosen] = 1
            init_caps[i, chosen] = np.random.randint(1, np.maximum(2, Grid_Caps[chosen] // 2) + 1)
        top_k = max(2, int(len(valid_nodes) * 0.15))
        top_potential_nodes = valid_nodes[np.argsort(-node_potential[valid_nodes])[:top_k]]
        for i in range(half_pop, Pop_Size):
            target_num = np.random.randint(2, min(5, len(top_potential_nodes) + 1))
            chosen_nodes = np.random.choice(top_potential_nodes, target_num, replace=False)
            init_status[i, chosen_nodes] = 1
            #init_caps[i, chosen_nodes] = np.random.randint(1, 3)
            # 解放潜力节点的容量限制，让其在 1 到 p_i 之间自由采样
            init_caps[i, chosen_nodes] = np.random.randint(1, np.maximum(2, Grid_Caps[chosen_nodes]) + 1)
        init_caps[init_status == 0] = 0
        MainPop[:, 0, :] = init_status
        MainPop[:, 1, :] = init_caps
        MainFit, MainCV = evaluate_population(MainPop)
        update_archive(MainPop, MainFit, MainCV)
        run_generation_history.append(Archive_F.copy())
        print(f"  Loop {loop+1}/{loop_times}: Initialized, Best=({MainFit[:,0].min():.0f},{MainFit[:,1].min():.1f}) |PF|={len(Fronts[0]) if Fronts else 0}", flush=True)

        for gen in range(max_gen):
            Fit_Params_MO['Current_Gen'] = gen + 1
            adaptive_control()
            binary_tournament_selection()
            crossover(MainPop)
            mutation(MainPop)
            repair(MainPop)

            MainFit, MainCV = evaluate_population(MainPop)

            # 把容易跑偏前沿的局部搜索剥离，让自然进化的基因流彻底主导
            comb_pop = MainPop;
            comb_F = MainFit;
            comb_CV = MainCV

            # ---- 环境选择 (严谨拥挤度裁剪) ----
            fr, _ = non_dominated_sort(comb_F, comb_CV)

            sel = []
            for f_idx in fr:
                if len(sel) + len(f_idx) <= Pop_Size:
                    sel.extend(f_idx)
                else:
                    need = Pop_Size - len(sel)
                    if need > 0:
                        cd = crowding_distance(comb_F, f_idx)
                        top = np.argsort(-cd)[:need]
                        sel.extend(f_idx[top])
                    break

            MainPop = comb_pop[sel].copy()
            MainFit = comb_F[sel].copy()
            MainCV = comb_CV[sel].copy()

            # ---- 精英注入与停滞控制 ----
            if gen > max_gen // 3 and np.random.rand() < 0.15 and Archive is not None:
                inject_size = min(Pop_Size // 6, len(Archive))
                if inject_size > 0:
                    inject_idx = np.random.choice(len(Archive), inject_size, replace=False)
                    MainPop[-inject_size:] = Archive[inject_idx].copy()
                    MainFit[-inject_size:] = Archive_F[inject_idx].copy()

            prev_front_len = len(Fronts[0]) if Fronts else 0
            update_archive(MainPop, MainFit, MainCV)

            Fronts, _ = non_dominated_sort(MainFit, MainCV)
            if len(Fronts[0]) <= prev_front_len:
                Stag_Counter += 1
            else:
                Stag_Counter = 0

            if Stag_Counter >= Stag_Threshold:
                print(f"    [Stagnation reset at gen {gen+1}, re-evaluating...]", flush=True)
                n_new = Pop_Size - min(5, len(Fronts[0]) if Fronts else 0)
                if n_new > 0 and Archive is not None and len(Archive) > 0:
                    arc_idx = np.random.choice(min(len(Archive), n_new), n_new, replace=True)
                    MainPop[-n_new:] = Archive[arc_idx].copy()
                    perturb = np.random.rand(n_new, Num_Nodes) < 0.25
                    MainPop[-n_new:, 0, :] = np.where(perturb, 1 - MainPop[-n_new:, 0, :], MainPop[-n_new:, 0, :])
                    MainFit, MainCV = evaluate_population(MainPop)
                Stag_Counter = 0

            # 进度输出
            if gen % 5 == 0 or gen == max_gen - 1:
                print(f"    Gen {gen+1:4d}/{max_gen} |PF|={len(Fronts[0]) if Fronts else 0} "
                      f"Cost=[{MainFit[:,0].min():.0f},{MainFit[:,0].max():.0f}] "
                      f"Time=[{MainFit[:,1].min():.1f},{MainFit[:,1].max():.1f}] "
                      f"Stag={Stag_Counter}", flush=True)

            run_generation_history.append(Archive_F.copy())

        post_processing_refine()
        all_pareto_fronts.append((Archive.copy(), Archive_F.copy()))
        all_times.append(time.time() - t0)
        all_loop_histories.append(run_generation_history)

    return all_pareto_fronts, all_times, all_loop_histories


# ============================================================================
# 6. 批处理主函数：依次优化各个案例、保存结果并绘制真实世界地图
# ============================================================================
# ============================================================================
# ============================================================================
# 6. Academic Batch Processing Main Function: Label-Free Vector Street Network
# ============================================================================
if __name__ == "__main__":
    from datetime import datetime
    from plot_map_final import plot_map, load_problem as load_p
    from basemap_utils import get_basemap
    import matplotlib.pyplot as plt
    def _set_font():
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['mathtext.fontset'] = 'stix'
    problem_path = "EV_Problems/shenzhen_275_EV.json"
    output_dir = "Results"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(problem_path):
        print(f"Error: {problem_path} not found")
        sys.exit(1)

    case_name = os.path.splitext(os.path.basename(problem_path))[0]

    # ---- 步骤1: 获取底图 ----
    basemap_path, xlim, ylim = get_basemap(problem_path)

    print(f"📂 Running optimization for: {case_name}")
    print("-" * 70)

    # ---- 步骤2: 优化 (跑10次) ----
    all_pareto_fronts, all_times, _ = run(problem_path, pop_size=100, max_gen=300, loop_times=10)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(output_dir, f"{case_name}_{timestamp}")
    os.makedirs(base_dir, exist_ok=True)

    # 保存每次的结果
    all_fronts = []
    for r, (pop_r, fit_r) in enumerate(all_pareto_fronts):
        np.savez(os.path.join(base_dir, f'run{r+1}_solution.npz'),
                 status=pop_r[:,0,:], caps=pop_r[:,1,:], F=fit_r)
        np.savetxt(os.path.join(base_dir, f'run{r+1}_pareto.csv'), fit_r,
                   delimiter=',', header='F1_Cost,F2_Time', comments='')
        all_fronts.append(fit_r)

    # 合并所有前沿并提取总Pareto
    merged = np.vstack(all_fronts)
    pf_all = merged.copy()
    # 提取非支配前沿
    N = len(pf_all); dc = np.zeros(N, int)
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if np.all(pf_all[j] <= pf_all[i]) and np.any(pf_all[j] < pf_all[i]):
                dc[i] += 1
    pf_all = pf_all[dc == 0]

    # 选折中解
    fn = (pf_all - pf_all.min(axis=0)) / (pf_all.max(axis=0) - pf_all.min(axis=0) + 1e-10)
    best_idx = np.argmin(np.linalg.norm(fn, axis=1))
    best_f = pf_all[best_idx]

    # 保存汇总
    np.savetxt(os.path.join(base_dir, 'pareto_front.csv'), pf_all,
               delimiter=',', header='F1_Cost,F2_Time', comments='')
    with open(os.path.join(base_dir, 'summary.json'), 'w') as f:
        json.dump({'times': all_times, 'mean_time': float(np.mean(all_times)),
                   'std_time': float(np.std(all_times)),
                   'pf_size': len(pf_all),
                   'best': best_f.tolist()}, f, indent=2)

    # 画图: 所有10次前沿 + 总前沿(高亮)
    _set_font()
    fig, ax = plt.subplots(figsize=(8, 6))
    for r, fr in enumerate(all_fronts):
        ax.scatter(fr[:, 0]/1e6, fr[:, 1], s=8, alpha=0.3, label=f'Run {r+1}' if r==0 else '')
    ax.scatter(pf_all[:, 0]/1e6, pf_all[:, 1], s=40, c='red', marker='*', zorder=5, label='Combined PF')
    ax.scatter(best_f[0]/1e6, best_f[1], s=100, c='gold', marker='D', zorder=6, label='Compromise')
    ax.set_xlabel('Cost (M¥)', fontsize=12)
    ax.set_ylabel('Time (h)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'pareto_combined.png'), dpi=300)
    plt.close()
    print(f"  Total PF: {len(pf_all)} solutions across {len(all_fronts)} runs")