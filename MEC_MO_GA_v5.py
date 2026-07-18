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
# 2. 纯手写标量尺度归一化非支配排序
# ============================================================================

def non_dominated_sort(F, cv):
    N = F.shape[0]
    # 无约束状态下，所有人默认都是可行解 (feasible 恒为 True)
    feasible = np.ones(N, dtype=bool)

    F_scaled = F.copy()
    f_max = np.max(F_scaled, axis=0)
    f_min = np.min(F_scaled, axis=0)
    denom = f_max - f_min
    denom[denom < 1e-8] = 1.0
    F_scaled = (F_scaled - f_min) / denom

    F_i = F_scaled[:, np.newaxis, :]
    F_j = F_scaled[np.newaxis, :, :]

    better_or_equal = np.all(F_j <= F_i, axis=2)
    strictly_better = np.any(F_j < F_i, axis=2)
    pareto_dominates = better_or_equal & strictly_better

    # 约束控制矩阵全部置零失活
    constraint_dominates = np.zeros((N, N), dtype=bool)
    dominated_by = pareto_dominates

    dominated_count = np.sum(dominated_by, axis=1)
    runs_dominates_list = [np.where(dominated_by[:, i])[0] for i in range(N)]

    fronts = []
    ranks = np.full(N, -1, dtype=int)
    current_front = np.where(dominated_count == 0)[0].tolist()

    while current_front:
        current_front = np.array(current_front)
        fronts.append(current_front)
        ranks[current_front] = len(fronts) - 1

        next_front = []
        for p in current_front:
            for q in runs_dominates_list[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    next_front.append(q)
        current_front = next_front

    return fronts, ranks


def crowding_distance(F, front_indices):
    if len(front_indices) <= 2:
        return np.full(len(front_indices), np.inf)

    F_front = F[front_indices]
    n_points, n_obj = F_front.shape
    distances = np.zeros(n_points)

    for m in range(n_obj):
        sorted_idx = np.argsort(F_front[:, m])
        sorted_F = F_front[sorted_idx, m]
        f_min, f_max = sorted_F[0], sorted_F[-1]
        if f_max - f_min < 1e-8: continue
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        distances[sorted_idx[1:-1]] += (sorted_F[2:] - sorted_F[:-2]) / (f_max - f_min)

    return distances


def binary_tournament_selection():
    global MainPop, MainFit, MainCV, Fronts, Ranks, CrowdDist
    N = MainPop.shape[0]

    Fronts, Ranks = non_dominated_sort(MainFit, MainCV)
    CrowdDist = np.zeros(N)
    for front in Fronts:
        if len(front) > 0:
            CrowdDist[front] = crowding_distance(MainFit, front)

    idx_a = np.random.randint(0, N, size=N)
    idx_b = np.random.randint(0, N, size=N)

    rank_a, rank_b = Ranks[idx_a], Ranks[idx_b]
    crowd_a, crowd_b = CrowdDist[idx_a], CrowdDist[idx_b]

    choose_a = (rank_a < rank_b) | ((rank_a == rank_b) & (crowd_a > crowd_b))
    selected_idx = np.where(choose_a, idx_a, idx_b)

    MainPop = MainPop[selected_idx].copy()
    MainFit = MainFit[selected_idx].copy()
    MainCV = MainCV[selected_idx].copy()


# ============================================================================
# 3. 矩阵化并行演化算子
# ============================================================================

def crossover(population):
    N, D = population.shape[0], Num_Nodes
    if N < 2: return
    pair_idx = np.random.permutation(N)
    mates = population[pair_idx]

    do_cross = np.random.rand(N, 1, 1) < Cross_Rate
    col_mask = np.random.randint(0, 2, size=(N, 1, D)).astype(bool)

    final_mask = do_cross & np.tile(col_mask, (1, 2, 1))
    population[:, :, :] = np.where(final_mask, mates, population)

    cap = population[:, 1, :].astype(float)
    cap_m = mates[:, 1, :].astype(float)

    rand = np.random.rand(N, D)
    beta = np.where(rand <= 0.5, (2.0 * rand) ** (1.0 / 16.0), (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / 16.0))
    c1 = 0.5 * ((1 + beta) * cap + (1 - beta) * cap_m)

    # 恢复完整的物理变压器容量限制，不进行 50% 减半刚性约束
    population[:, 1, :] = np.clip(np.round(c1), 0, Grid_Caps).astype(int)
    population[:, 1, :] = np.where((population[:, 0, :] == 1) & (population[:, 1, :] == 0), 1, population[:, 1, :])


def mutation(population):
    N, D = population.shape[0], Num_Nodes
    if N == 0: return
    mask = np.random.rand(N, D) < Mut_Rate
    population[:, 0, :] = np.where(mask, 1 - population[:, 0, :], population[:, 0, :])

    cap = population[:, 1, :].astype(float)
    rand = np.random.rand(N, D)
    delta = np.where(rand <= 0.5, (2.0 * rand) ** (1.0 / 21.0) - 1.0, 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / 21.0))

    x_new = cap + delta * Grid_Caps[np.newaxis, :]
    population[:, 1, :] = np.clip(np.round(x_new), 0, Grid_Caps).astype(int)
    population[:, 1, :] = np.where(population[:, 0, :] == 1, np.maximum(population[:, 1, :], 1), 0)


def repair(population):
    """移除邻域引力分流，仅保留基础开站保底，让种群自由跨越目标曲面"""
    population[:, 1, :] = np.where(population[:, 0, :] == 1, np.maximum(population[:, 1, :], 1), 0)


# ============================================================================
# 4. 纯双目标无约束并行评估算子
# ============================================================================

def evaluate_population(population):
    N = population.shape[0]
    F = np.zeros((N, 2))
    CV = np.zeros(N) # 约束惩罚完全归零

    for i in range(N):
        ind_copy = population[i].copy()
        ind_copy[0] = (ind_copy[0] >= 0.5).astype(int)

        _, details = ff.fitness_function(ind_copy, Dist_Matrix, Demands, Grid_Caps, Fit_Params_MO)

        # 目标一：纯初始化基础设施固定+变动投入
        F[i, 0] = details['BuildCost']

        # 目标二：纯物理时间。彻底冲刷掉任何刚性 50000 斩落上限，让出行时间与排队等待无污染累加
        t_travel = details.get('TravelTime', 0.0)
        t_wait = details.get('WaitingTime', 0.0)
        F[i, 1] = t_travel + t_wait

        # 强制所有人完全可行合规
        CV[i] = 0.0

    return F, CV


def update_archive(population, F, cv, max_size=100):
    global Archive, Archive_F, Archive_CV
    # 移除严格可行解过滤器，所有人无条件入场自由淘沙
    current_pop, current_F, current_cv = population, F, cv

    if Archive is None:
        Archive = current_pop.copy(); Archive_F = current_F.copy(); Archive_CV = current_cv.copy()
        return

    combined_pop = np.vstack([Archive, current_pop])
    combined_F = np.vstack([Archive_F, current_F])
    combined_CV = np.concatenate([Archive_CV, current_cv])

    fronts, _ = non_dominated_sort(combined_F, combined_CV)
    first_front = np.array([int(i) for i in fronts[0] if i < len(combined_pop)], dtype=int)

    new_archive = combined_pop[first_front]; new_F = combined_F[first_front]; new_CV = combined_CV[first_front]
    if len(new_archive) > 1:
        flat = new_archive.reshape(len(new_archive), -1)
        _, unique_idx = np.unique(flat, axis=0, return_index=True)
        new_archive, new_F, new_CV = new_archive[unique_idx], new_F[unique_idx], new_CV[unique_idx]

    if len(new_archive) > max_size:
        crowd = crowding_distance(new_F, np.arange(len(new_F)))
        keep = np.argsort(-crowd)[:max_size]
        new_archive, new_F, new_CV = new_archive[keep], new_F[keep], new_CV[keep]

    Archive, Archive_F, Archive_CV = new_archive.copy(), new_F.copy(), new_CV.copy()


def adaptive_control():
    global Cross_Rate, Mut_Rate, MainPop, MainFit, Stag_Counter
    X = MainPop[:, 0, :]
    centroid = np.mean(X, axis=0)
    dist_vector = np.linalg.norm(X - centroid, axis=1)
    max_d, mean_d = np.max(dist_vector), np.mean(dist_vector)
    I_spatial = mean_d / (max_d + 1e-8) if max_d > 0 else 0.0

    F1 = MainFit[:, 0]
    f_min, f_max = np.min(F1), np.max(F1)
    I_temporal = (np.mean(F1) - f_min) / (f_max - f_min + 1e-8) if f_max > f_min else 0.0
    I_stagnation = min(1.0, Stag_Counter / Stag_Threshold)

    omega = 0.4 * I_spatial + 0.3 * (1 - I_temporal) + 0.3 * (1 - I_stagnation)
    Cross_Rate = np.clip(0.40 + 0.50 * omega, 0.40, 0.90)
    Mut_Rate = np.clip(0.20 - 0.15 * omega, 0.05, 0.20)


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

def run(problem_file, pop_size=100, max_gen=300, loop_times=10, verbose=True):
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

    for loop in range(loop_times):
        t0 = time.time(); Stag_Counter = 0
        Archive = Archive_F = Archive_CV = None
        run_generation_history = []

        MainPop = np.zeros((Pop_Size, 2, Num_Nodes))
        init_open_ratio = min(0.3, 20.0 / Num_Nodes)

        for i in range(Pop_Size):
            MainPop[i, 0, :] = np.random.choice([0, 1], size=Num_Nodes, p=[1.0 - init_open_ratio, init_open_ratio])
            for j in range(Num_Nodes):
                if MainPop[i, 0, j] == 1 and Grid_Caps[j] > 0:
                    MainPop[i, 1, j] = np.random.randint(1, Grid_Caps[j] + 1)
                else:
                    MainPop[i, 0, j] = MainPop[i, 1, j] = 0

        MainFit, MainCV = evaluate_population(MainPop)
        update_archive(MainPop, MainFit, MainCV)
        run_generation_history.append(Archive_F.copy())

        for gen in range(max_gen):
            Fit_Params_MO['Current_Gen'] = gen + 1
            adaptive_control()
            binary_tournament_selection()

            crossover(MainPop); mutation(MainPop); repair(MainPop)
            MainFit, MainCV = evaluate_population(MainPop)

            # ---- 改进1: Archive 引导局部搜索 ----
            if gen > max_gen * 0.3 and Archive is not None and len(Archive) > 0:
                n_local = max(1, Pop_Size // 10)
                local_idx = np.random.choice(len(Archive), n_local, replace=False)
                local_pop = Archive[local_idx].copy()
                # 小步长邻域扰动
                for k in range(n_local):
                    # 随机选 10% 的节点翻转
                    flip = np.random.rand(Num_Nodes) < 0.1
                    local_pop[k, 0, flip] = 1 - local_pop[k, 0, flip]
                    # 容量微调
                    active = local_pop[k, 0] > 0
                    if np.any(active):
                        delta = np.random.randint(-2, 3, size=Num_Nodes)
                        local_pop[k, 1] = np.clip(local_pop[k, 1] + delta, 1, Grid_Caps)
                        local_pop[k, 1, ~active] = 0
                repair(local_pop)
                local_F, local_CV = evaluate_population(local_pop)
                # 合并选优
                comb_pop = np.vstack([MainPop, local_pop])
                comb_F = np.vstack([MainFit, local_F])
                comb_CV = np.concatenate([MainCV, local_CV])
                fr, _ = non_dominated_sort(comb_F, comb_CV)
                sel = []
                for f_idx in fr:
                    if len(sel) + len(f_idx) <= Pop_Size:
                        sel.extend(f_idx)
                    else:
                        need = Pop_Size - len(sel)
                        cd = crowding_distance(comb_F, f_idx)
                        top = np.argsort(-cd)[:need]
                        sel.extend(f_idx[top])
                        break
                MainPop = comb_pop[sel].copy()
                MainFit = comb_F[sel].copy()
                MainCV = comb_CV[sel].copy()

            # ---- 改进2: Archive 注入 (保留) ----
            if gen > max_gen // 3 and np.random.rand() < 0.2 and Archive is not None:
                inject_size = min(Pop_Size // 5, len(Archive))
                inject_idx = np.random.choice(len(Archive), inject_size, replace=False)
                MainPop[-inject_size:] = Archive[inject_idx].copy()
                MainFit[-inject_size:] = Archive_F[inject_idx].copy()
                MainCV[-inject_size:] = Archive_CV[inject_idx].copy()

            prev_front_len = len(Fronts[0]) if Fronts else 0

            # 全部放行进入 Archive 归档更新
            update_archive(MainPop, MainFit, MainCV)

            Fronts, _ = non_dominated_sort(MainFit, MainCV)
            if len(Fronts[0]) <= prev_front_len: Stag_Counter += 1
            else: Stag_Counter = 0

            if Stag_Counter >= Stag_Threshold:
                n_new = Pop_Size - min(5, len(Fronts[0]))
                if n_new > 0 and Archive is not None and len(Archive) > 0:
                    # 从 Archive 注入多样性 + 小扰动，替代纯随机
                    arc_idx = np.random.choice(min(len(Archive), n_new), n_new, replace=True)
                    MainPop[-n_new:] = Archive[arc_idx].copy()
                    # 扰动 30% 节点
                    perturb = np.random.rand(n_new, Num_Nodes) < 0.3
                    MainPop[-n_new:, 0, :] = np.where(perturb, 1 - MainPop[-n_new:, 0, :], MainPop[-n_new:, 0, :])
                    for i_new in range(Pop_Size - n_new, Pop_Size):
                        MainPop[i_new, 1, :] = np.where(MainPop[i_new, 0, :] == 1,
                            np.random.randint(1, max(2, Grid_Caps), size=Num_Nodes), 0)
                    MainFit, MainCV = evaluate_population(MainPop)
                Stag_Counter = 0

            run_generation_history.append(Archive_F.copy())

        post_processing_refine()
        all_pareto_fronts.append((Archive.copy(), Archive_F.copy()))
        all_times.append(time.time() - t0)
        all_loop_histories.append(run_generation_history)

    return all_pareto_fronts, all_times, all_loop_histories