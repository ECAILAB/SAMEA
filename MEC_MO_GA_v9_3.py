"""
MEC_MO_GA_v8_1.py — SAMEA / MEA 选址定容演化内核
主流程只做函数调用，不含零碎逻辑。
"""
import os
import sys
import json
import time
import numpy as np
LS_ENABLED = True
try:
    import Fitness_Function_v3 as ff
except ImportError:
    ff = None

# =============================================================================
# 全局变量
# =============================================================================
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
hv_history_window = []

Coords = Demands = Grid_Caps = Dist_Matrix = None
Num_Nodes = 0
Fit_Params_MO = {}


# =============================================================================
# 非支配排序 / 拥挤度 / 选择
# =============================================================================
def non_dominated_sort(F, cv):
    N = F.shape[0]
    fronts, ranks = [], np.full(N, -1, dtype=int)
    remaining = np.ones(N, dtype=bool)
    while np.any(remaining):
        idx = np.where(remaining)[0]
        Fi = F[idx]
        si = np.lexsort((Fi[:, 1], Fi[:, 0]))
        sorted_idx = idx[si]
        sorted_f = F[sorted_idx]
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
    if len(front_indices) <= 2:
        return np.full(len(front_indices), np.inf)
    Ff = F[front_indices]
    n = len(front_indices)
    cd = np.zeros(n)
    for m in range(Ff.shape[1]):
        si = np.argsort(Ff[:, m])
        sf = Ff[si, m]
        rng = sf[-1] - sf[0]
        if rng < 1e-10:
            continue
        cd[si[0]] = np.inf
        cd[si[-1]] = np.inf
        cd[si[1:-1]] += (sf[2:] - sf[:-2]) / rng
    return cd





# =============================================================================
# 交叉 / 变异 / 修复
# =============================================================================
def crossover(pop):
    global Pop_Size, Num_Nodes, Cross_Rate
    p2_pop = pop[np.random.permutation(Pop_Size)]
    pc = np.asarray(Cross_Rate, dtype=float).reshape(-1)
    if pc.size == 1:
        pc = np.full(Pop_Size, float(pc[0]))
    trigger = np.random.rand(Pop_Size, 1, 1) < pc.reshape(Pop_Size, 1, 1)
    uni = np.random.rand(Pop_Size, Num_Nodes) < 0.5
    mask = np.zeros((Pop_Size, 2, Num_Nodes), dtype=bool)
    mask[:, 0, :] = uni
    mask[:, 1, :] = uni
    pop[:] = np.where(trigger & mask, p2_pop, pop)



def mutation(pop):
    """
    半随机 + 半有向：
      F2 高于中位数：以 0.60 倾向开站/加桩
      F2 低于中位数：以 0.40 倾向开站（即 0.60 倾向关站/减桩）
    """
    global Pop_Size, Num_Nodes, Mut_Rate, Grid_Caps, MainFit
    status_2d = pop[:, 0, :].copy()
    caps_2d = pop[:, 1, :].copy()

    pm = np.asarray(Mut_Rate, dtype=float).reshape(-1)
    if pm.size == 1:
        pm = np.full(Pop_Size, float(pm[0]))
    mut_trigger = np.random.rand(Pop_Size, Num_Nodes) < pm.reshape(Pop_Size, 1)

    mode_rand = np.random.rand(Pop_Size, 1) < 0.5

    median_time = np.median(MainFit[:, 1])
    high_mask = (MainFit[:, 1] > median_time)[:, np.newaxis]

    rand_status = 1 - status_2d
    rand_caps = np.random.randint(1, np.maximum(2, Grid_Caps[np.newaxis, :]))

    p_open_high = 0.60
    p_open_low = 0.40
    dr = np.random.rand(Pop_Size, Num_Nodes)

    dir_high_s = np.where(dr < p_open_high, 1, 0)
    dir_high_c = np.clip(caps_2d + 2, 1, Grid_Caps[np.newaxis, :])
    dir_low_s = np.where(dr < p_open_low, 1, 0)
    dir_low_c = np.maximum(1, caps_2d - 1)

    biased_s = np.where(high_mask, dir_high_s, dir_low_s)
    biased_c = np.where(high_mask, dir_high_c, dir_low_c)

    final_s = np.where(mode_rand, rand_status, biased_s)
    final_c = np.where(mode_rand, rand_caps, biased_c)

    status_2d[:] = np.where(mut_trigger, final_s, status_2d)
    caps_2d[:] = np.where(mut_trigger & (status_2d == 1), final_c, caps_2d)
    caps_2d[status_2d == 0] = 0
    caps_2d[:] = np.minimum(caps_2d, Grid_Caps[np.newaxis, :])

    pop[:, 0, :] = status_2d
    pop[:, 1, :] = caps_2d



def repair(population):
    """耦合约束 + 至少开 1 站，禁止 F1=0"""
    global Grid_Caps
    population[:, 1, :] = np.where(
        population[:, 0, :] == 1,
        np.maximum(population[:, 1, :], 1),
        0,
    )
    population[:, 1, :] = np.minimum(population[:, 1, :], Grid_Caps[np.newaxis, :])

    valid = np.where(Grid_Caps > 0)[0]
    if len(valid) == 0:
        return
    for i in range(population.shape[0]):
        if np.sum(population[i, 0, :]) == 0:
            j = valid[np.random.randint(len(valid))]
            population[i, 0, j] = 1
            population[i, 1, j] = max(1, int(Grid_Caps[j] * 0.1))


# =============================================================================
# 评估 / 归档
# =============================================================================
def evaluate_population(population):
    N = population.shape[0]
    F = np.zeros((N, 2))
    CV = np.zeros(N)
    dm, dmds, gc, fp = Dist_Matrix, Demands, Grid_Caps, Fit_Params_MO
    for i in range(N):
        ind = population[i].copy()
        ind[0] = (ind[0] >= 0.5).astype(int)
        _, details = ff.fitness_function(ind, dm, dmds, gc, fp)
        F[i, 0] = details["BuildCost"]
        F[i, 1] = details.get("TravelTime", 0.0) + details.get("WaitingTime", 0.0)
        CV[i] = 0.0
    return F, CV






def _archive_hv_2d(F):
    """归档近似 HV（归一到 [0,1] 后相对 (1.1,1.1)），仅用于 LS 接受判定。"""
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return 0.0
    fmin, fmax = F.min(0), F.max(0)
    span = np.maximum(fmax - fmin, 1e-8)
    Fn = (F - fmin) / span
    # 非支配
    n = len(Fn)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i != j and keep[j]:
                if np.all(Fn[j] <= Fn[i]) and np.any(Fn[j] < Fn[i]):
                    keep[i] = False
                    break
    P = Fn[keep]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    ref = np.array([1.1, 1.1])
    hv, prev_x = 0.0, ref[0]
    for i in range(len(P)):
        if np.any(P[i] > ref):
            continue
        w = prev_x - P[i, 0]
        h = ref[1] - P[i, 1]
        if w > 0 and h > 0:
            hv += w * h
        prev_x = P[i, 0]
    return float(hv)


def local_search(attraction_matrix=None, intensity=0.5, gen=0, max_gen=100):
    """
    多目标 LS（只精修归档，不改主种群轨迹）：
      - 按前沿位置选邻域方向
      - 期望改动次数 K 与 D 解耦（p=K/D）
      - 不被原解支配，且替换后归档 HV 不降 → 写回 Archive
      - 不 vstack 进 MainPop（避免和 SAA 分叉、拖累 IGD）
    """
    global Archive, Archive_F, Archive_CV
    global Pop_Size, Grid_Caps, Num_Nodes

    if Archive is None or len(Archive) == 0:
        return

    n_arc = len(Archive)
    n_sample = max(2, min(n_arc, max(2, Pop_Size // 10)))
    sel = np.random.choice(n_arc, n_sample, replace=False)

    F = Archive_F.astype(float)
    fmin, fmax = F.min(0), F.max(0)
    span = np.maximum(fmax - fmin, 1e-8)
    u1 = (F[:, 0] - fmin[0]) / span[0]
    u2 = (F[:, 1] - fmin[1]) / span[1]

    K = 2.0 + 2.0 * float(intensity)
    p_flip = float(K / max(Num_Nodes, 1))
    hv_base = _archive_hv_2d(Archive_F)

    # 可选：不动当前两端，减少端点被挪走
    i_f1 = int(np.argmin(Archive_F[:, 0]))
    i_f2 = int(np.argmin(Archive_F[:, 1]))

    for k in sel:
        k = int(k)
        if k == i_f1 or k == i_f2:
            continue

        s = Archive[k, 0].copy()
        c = Archive[k, 1].astype(float).copy()
        fo = Archive_F[k].copy()

        prefer_cost = u1[k] > u2[k] + 0.05
        prefer_svc = u2[k] > u1[k] + 0.05

        active = np.where(s > 0)[0]
        closed = np.where((s == 0) & (Grid_Caps > 0))[0]
        if len(active) == 0:
            continue

        phi = (np.sum(attraction_matrix, axis=0) + 1e-8
               if attraction_matrix is not None else np.ones(Num_Nodes))

        flip_mask = np.random.rand(Num_Nodes) < p_flip
        for j in np.where(flip_mask)[0]:
            if prefer_cost:
                if s[j] == 1 and len(active) > 2 and phi[j] <= np.median(phi[active]):
                    s[j], c[j] = 0, 0
                    active = np.where(s > 0)[0]
                elif s[j] == 0 and Grid_Caps[j] > 0 and np.random.rand() < 0.2:
                    s[j] = 1
                    c[j] = max(1, int(Grid_Caps[j] * 0.1))
            elif prefer_svc:
                if s[j] == 0 and Grid_Caps[j] > 0:
                    if len(closed) == 0 or phi[j] >= np.median(phi[closed]):
                        s[j] = 1
                        c[j] = max(1, int(Grid_Caps[j] * 0.15))
                elif s[j] == 1:
                    c[j] = min(Grid_Caps[j], c[j] + 1)
            else:
                s[j] = 1 - s[j]
                c[j] = max(1, int(c[j])) if s[j] == 1 else 0

        active = np.where(s > 0)[0]
        if len(active) == 0:
            continue

        for j in active:
            if prefer_cost and c[j] > 1 and np.random.rand() < 0.5:
                c[j] = max(1, c[j] - 1)
            elif prefer_svc and c[j] < Grid_Caps[j] and np.random.rand() < 0.5:
                c[j] = min(Grid_Caps[j], c[j] + 1)
            else:
                c[j] = int(np.clip(c[j] + np.random.randint(-1, 2), 1, Grid_Caps[j]))

        trial = np.zeros_like(Archive[k])
        trial[0, :] = s
        trial[1, active] = c[active]
        repair(trial[np.newaxis, ...])
        fn, cv = evaluate_population(trial[np.newaxis, ...])
        fn, cv = fn[0], float(cv[0])

        # 被原解严格支配 → 丢
        if np.all(fo <= fn) and np.any(fo < fn):
            continue

        # 替换后归档 HV 不降 → 才写回
        F_try = Archive_F.copy()
        F_try[k] = fn
        if _archive_hv_2d(F_try) + 1e-12 < hv_base:
            continue

        Archive[k] = trial
        Archive_F[k] = fn
        if Archive_CV is not None:
            Archive_CV[k] = cv
        hv_base = _archive_hv_2d(Archive_F)


def binary_tournament_indices(n):
    """只产生交配下标，不覆盖 MainPop。"""
    global MainFit, MainCV, Fronts, Ranks, CrowdDist
    Fronts, Ranks = non_dominated_sort(MainFit, MainCV)
    CrowdDist = np.zeros(n)
    for fr in Fronts:
        if len(fr) > 0:
            CrowdDist[fr] = crowding_distance(MainFit, fr)
    a = np.random.randint(0, n, size=n)
    b = np.random.randint(0, n, size=n)
    ra, rb = Ranks[a], Ranks[b]
    ca, cb = CrowdDist[a], CrowdDist[b]
    return np.where(ra < rb, a, np.where(ra > rb, b, np.where(ca > cb, a, b)))

def _archive_improved(prev_F, curr_F):
    """多目标：归档是否相对上一代有改进。"""
    if prev_F is None or len(prev_F) == 0:
        return True
    if curr_F is None or len(curr_F) == 0:
        return False
    p1, p2 = float(np.min(prev_F[:, 0])), float(np.min(prev_F[:, 1]))
    c1, c2 = float(np.min(curr_F[:, 0])), float(np.min(curr_F[:, 1]))
    if c1 < p1 - 1e-12 or c2 < p2 - 1e-12:
        return True
    if len(curr_F) > len(prev_F):
        return True
    return False








def post_processing_refine():
    global Archive, Archive_F, Archive_CV
    if Archive is None or len(Archive) == 0:
        return
    Archive_F, Archive_CV = evaluate_population(Archive)


# =============================================================================
# 数据 / 初始化
# =============================================================================
def load_dataset(filename):
    global Coords, Demands, Grid_Caps, Dist_Matrix, Num_Nodes
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    Coords = np.array([[n["x"], n["y"]] for n in data["nodes"]], dtype=float)
    Demands = np.array([n["demand"] for n in data["nodes"]], dtype=float)
    Grid_Caps = np.array([n["grid_capacity"] for n in data["nodes"]], dtype=float)
    Num_Nodes = len(Coords)
    Dist_Matrix = np.sum(np.abs(Coords[:, None, :] - Coords[None, :, :]), axis=2)


def build_attraction():
    return Demands[:, np.newaxis] / (Dist_Matrix + 1.0)





def environmental_selection(parents_pop, parents_F, parents_CV, offspring, off_F, off_CV):
    """(μ+λ) 环境选择：2N → N"""
    global Pop_Size
    comb_pop = np.concatenate([parents_pop, offspring], axis=0)
    comb_F = np.concatenate([parents_F, off_F], axis=0)
    comb_CV = np.concatenate([parents_CV, off_CV], axis=0)
    fr, _ = non_dominated_sort(comb_F, comb_CV)
    sel = []
    for f_idx in fr:
        if len(sel) + len(f_idx) <= Pop_Size:
            sel.extend(list(f_idx))
        else:
            need = Pop_Size - len(sel)
            if need > 0:
                cd = crowding_distance(comb_F, f_idx)
                sel.extend(list(f_idx[np.argsort(-cd)[:need]]))
            break
    return (
        comb_pop[sel].copy(),
        comb_F[sel].copy(),
        comb_CV[sel].copy(),
    )


def inject_elites():
    """
    从归档前沿均匀取点（含两端与中段），替换主群秩最差个体，
    并同步这些位置的交叉/变异率。
    """
    global MainPop, MainFit, MainCV, Archive, Archive_F, Archive_CV, Pop_Size
    global Cross_Rate, Mut_Rate, _elite_pc_base, _elite_pm_base

    if Archive is None or len(Archive) == 0 or MainFit is None:
        return

    n_arc = len(Archive)
    n_inj = min(max(1, Pop_Size // 8), n_arc)

    order_f1 = np.argsort(Archive_F[:, 0])
    if n_inj == 1:
        src = np.array([order_f1[n_arc // 2]], dtype=int)
    elif n_inj == 2:
        src = np.array([order_f1[0], order_f1[-1]], dtype=int)
    else:
        pos = np.linspace(0, n_arc - 1, n_inj)
        pos = np.unique(np.round(pos).astype(int))
        if len(pos) < n_inj:
            extra = [int(i) for i in order_f1 if i not in set(pos.tolist())]
            for e in extra:
                if len(pos) >= n_inj:
                    break
                pos = np.append(pos, e)
            pos = pos[:n_inj]
        src = order_f1[pos]

    i_f1 = int(np.argmin(Archive_F[:, 0]))
    i_f2 = int(np.argmin(Archive_F[:, 1]))
    src = list(np.asarray(src, dtype=int))
    for e in (i_f1, i_f2):
        if e not in src and len(src) < n_arc:
            for k in range(len(src)):
                if src[k] != i_f1 and src[k] != i_f2:
                    src[k] = e
                    break
            else:
                src.append(e)
    src = np.asarray(src[:n_inj], dtype=int)
    n_inj = len(src)

    cv = MainCV if MainCV is not None else np.zeros(len(MainFit))
    _, ranks = non_dominated_sort(MainFit, cv)
    worst = np.lexsort((MainFit[:, 0] + MainFit[:, 1], -ranks))[:n_inj]

    MainPop[worst] = Archive[src].copy()
    MainFit[worst] = Archive_F[src].copy()
    if Archive_CV is not None and MainCV is not None:
        MainCV[worst] = Archive_CV[src].copy()

    cr = np.asarray(Cross_Rate, float).reshape(-1)
    mr = np.asarray(Mut_Rate, float).reshape(-1)
    if cr.size != Pop_Size:
        cr = np.full(Pop_Size, 0.80, dtype=float)
        mr = np.full(Pop_Size, 0.10, dtype=float)
    else:
        cr, mr = cr.copy(), mr.copy()
    elite_pc = float(np.clip(globals().get("_elite_pc_base", 0.80), 0.60, 0.90))
    elite_pm = float(np.clip(globals().get("_elite_pm_base", 0.10), 0.05, 0.25))
    cr[worst] = elite_pc
    mr[worst] = elite_pm
    Cross_Rate = cr
    Mut_Rate = mr



def handle_stagnation():
    global MainPop, MainFit, MainCV, Fronts, Archive, Stag_Counter, Pop_Size, Num_Nodes
    if Stag_Counter < Stag_Threshold:
        return
    print(f"    [Stagnation reset, Stag={Stag_Counter}]", flush=True)
    n_keep = min(5, len(Fronts[0]) if Fronts else 0)
    n_new = Pop_Size - n_keep
    if n_new > 0 and Archive is not None and len(Archive) > 0:
        arc_idx = np.random.choice(min(len(Archive), n_new), n_new, replace=True)
        MainPop[-n_new:] = Archive[arc_idx].copy()
        perturb = np.random.rand(n_new, Num_Nodes) < 0.25
        MainPop[-n_new:, 0, :] = np.where(
            perturb, 1 - MainPop[-n_new:, 0, :], MainPop[-n_new:, 0, :])
        repair(MainPop)
        MainFit, MainCV = evaluate_population(MainPop)
    Stag_Counter = 0




def binary_tournament_selection():
    global MainPop, MainFit, MainCV, Fronts, Ranks, CrowdDist, Cross_Rate, Mut_Rate
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
    Cross_Rate = np.asarray(Cross_Rate, float).reshape(-1)[pick].copy()
    Mut_Rate = np.asarray(Mut_Rate, float).reshape(-1)[pick].copy()


def update_archive(pop, fit, cv=None):
    global Archive, Archive_F, Archive_CV, Pop_Size
    if cv is None:
        cv = np.zeros(len(fit))

    if Archive is None or len(Archive) == 0:
        fronts, _ = non_dominated_sort(fit, cv)
        best_idx = fronts[0]
        _, uniq = np.unique(pop[best_idx], axis=0, return_index=True)
        final_idx = best_idx[uniq]
        Archive = pop[final_idx].copy()
        Archive_F = fit[final_idx].copy()
        Archive_CV = cv[final_idx].copy()
        return

    combined_pop = np.concatenate([Archive, pop], axis=0)
    combined_fit = np.concatenate([Archive_F, fit], axis=0)
    combined_cv = np.concatenate([Archive_CV, cv], axis=0)

    flat = combined_pop.reshape(combined_pop.shape[0], -1)
    _, unique_idx = np.unique(flat, axis=0, return_index=True)
    combined_pop = combined_pop[unique_idx]
    combined_fit = combined_fit[unique_idx]
    combined_cv = combined_cv[unique_idx]

    fronts, _ = non_dominated_sort(combined_fit, combined_cv)
    pareto_idx = fronts[0]

    max_archive_size = max(1, int(Pop_Size * 0.6))
    if len(pareto_idx) > max_archive_size:
        fit_c = combined_fit[pareto_idx]
        must = {int(np.argmin(fit_c[:, 0])), int(np.argmin(fit_c[:, 1]))}

        obj_min, obj_max = fit_c.min(0), fit_c.max(0)
        denom = obj_max - obj_min
        denom[denom == 0] = 1.0
        s1, s2 = np.argsort(fit_c[:, 0]), np.argsort(fit_c[:, 1])
        dist = np.zeros(len(pareto_idx))
        dist[s1[0]] = dist[s1[-1]] = np.inf
        dist[s2[0]] = dist[s2[-1]] = np.inf
        dist[s1[1:-1]] += (fit_c[s1[2:], 0] - fit_c[s1[:-2], 0]) / denom[0]
        dist[s2[1:-1]] += (fit_c[s2[2:], 1] - fit_c[s2[:-2], 1]) / denom[1]

        order = np.argsort(-dist)
        keep_pos = list(must)
        for p in order:
            p = int(p)
            if p not in must:
                keep_pos.append(p)
            if len(keep_pos) >= max_archive_size:
                break
        pareto_idx = pareto_idx[np.asarray(keep_pos[:max_archive_size], dtype=int)]

    Archive = combined_pop[pareto_idx].copy()
    Archive_F = combined_fit[pareto_idx].copy()
    Archive_CV = combined_cv[pareto_idx].copy()


def initialize_population(attraction_matrix=None):
    """10% 超轻资产 + 原 8.3 半侧重资产 / 半侧高势能。"""
    global Pop_Size, Num_Nodes, Grid_Caps, Demands, Fit_Params_MO
    global _elite_pc_base, _elite_pm_base
    _elite_pc_base, _elite_pm_base = 0.80, 0.30

    if attraction_matrix is None:
        attraction_matrix = build_attraction()
    node_potential = np.sum(attraction_matrix, axis=0)

    pop = np.zeros((Pop_Size, 2, Num_Nodes))
    valid = np.where(Grid_Caps > 0)[0]
    if len(valid) == 0:
        return pop

    n_light = max(1, Pop_Size // 10)
    half = max(n_light + 1, Pop_Size // 2)

    # 10% 超轻：1～3 站，每站 1 桩
    for i in range(n_light):
        k = np.random.randint(1, min(4, len(valid) + 1))
        ch = np.random.choice(valid, k, replace=False)
        pop[i, 0, ch] = 1
        pop[i, 1, ch] = 1

    total_demand = float(np.sum(Demands))
    avg_serve = float(np.mean(Grid_Caps[valid])) * Fit_Params_MO["Service_Rate"] * 0.3
    base = max(5, int(total_demand / max(1.0, avg_serve)))

    for i in range(n_light, half):
        ratio = 1.5 + ((i - n_light) / max(1, half - n_light - 1))
        n_open = max(1, min(len(valid), max(base, int(base * ratio))))
        ch = np.random.choice(valid, n_open, replace=False)
        pop[i, 0, ch] = 1
        pop[i, 1, ch] = np.random.randint(1, np.maximum(2, Grid_Caps[ch] // 2) + 1)

    top_k = max(2, int(len(valid) * 0.15))
    top = valid[np.argsort(-node_potential[valid])[:top_k]]
    for i in range(half, Pop_Size):
        hi = min(5, len(top))
        tn = max(1, min(hi, 2 if hi < 2 else np.random.randint(2, hi + 1)))
        ch = np.random.choice(top, tn, replace=False)
        pop[i, 0, ch] = 1
        pop[i, 1, ch] = np.random.randint(1, np.maximum(2, Grid_Caps[ch]) + 1)

    pop[:, 1, :][pop[:, 0, :] == 0] = 0
    repair(pop)
    return pop


def adaptive_control(success_pc=None, success_pm=None):
    """
    I_spat / I_temp / I_stag → 精英率基准 + 按秩微调。
    success_pc/pm 仅兼容消融调用，不使用。
    """
    global Cross_Rate, Mut_Rate, MainPop, MainFit, MainCV
    global Stag_Counter, hv_history_window, Ranks, Pop_Size
    global _elite_pc_base, _elite_pm_base

    p_cmin, p_cmax = 0.40, 0.90
    p_mmin, p_mmax = 0.05, 0.2
    R_max = 3.0
    w, eps, tau, G_stag = 10, 1e-8, 1e-4, 20.0

    N = Pop_Size
    if MainFit is None or len(MainFit) == 0:
        Cross_Rate = np.full(N, 0.8)
        Mut_Rate = np.full(N, 0.1)
        return

    X = MainPop[:, 0, :]
    centroid = np.mean(X, axis=0)
    dist = np.linalg.norm(X - centroid, axis=1)
    max_d = float(np.max(dist))
    I_spat = float(np.mean(dist) / (max_d + 1e-8)) if max_d > 0 else 0.0

    cv = MainCV if MainCV is not None else np.zeros(len(MainFit))
    fronts, ranks = non_dominated_sort(MainFit, cv)
    Ranks = ranks
    F_front = MainFit[fronts[0]]
    f1_best = float(np.min(F_front[:, 0]))
    f2_best = float(np.min(F_front[:, 1]))

    hv_history_window.append((f1_best, f2_best))
    if len(hv_history_window) > w + 1:
        hv_history_window.pop(0)

    if len(hv_history_window) >= 2:
        f1_old, f2_old = hv_history_window[0]
        imp1 = (f1_old - f1_best) / max(abs(f1_old), eps)
        imp2 = (f2_old - f2_best) / max(abs(f2_old), eps)
        I_temp = float(np.clip(max(0.0, max(imp1, imp2)), 0.0, 1.0))
    else:
        I_temp = 0.0

    I_stag = min(1.0, float(Stag_Counter) / G_stag)

    I = float(np.clip(
        0.4 * I_spat + 0.3 * (1.0 - I_temp) + 0.3 * (1.0 - I_stag),
        0.0, 1.0))

    pc_target = p_cmin + (p_cmax - p_cmin) * I
    pm_target = p_mmin + (p_mmax - p_mmin) * (1.0 - I) * 0.5

    if "_elite_pc_base" not in globals() or _elite_pc_base is None:
        _elite_pc_base = pc_target
        _elite_pm_base = pm_target

    elite_mask = (ranks == 0)
    if (Cross_Rate is not None
            and len(np.asarray(Cross_Rate).reshape(-1)) == N
            and np.any(elite_mask)):
        cr = np.asarray(Cross_Rate, float).reshape(-1)
        mr = np.asarray(Mut_Rate, float).reshape(-1)
        mean_pc = float(np.mean(cr[elite_mask]))
        mean_pm = float(np.mean(mr[elite_mask]))
        _elite_pc_base = 0.7 * mean_pc + 0.3 * pc_target
        _elite_pm_base = 0.7 * mean_pm + 0.3 * pm_target
    else:
        _elite_pc_base = 0.7 * float(_elite_pc_base) + 0.3 * pc_target
        _elite_pm_base = 0.7 * float(_elite_pm_base) + 0.3 * pm_target

    base_pc = float(np.clip(_elite_pc_base, p_cmin, p_cmax))
    base_pm = float(np.clip(_elite_pm_base, p_mmin, p_mmax))

    r_tilde = np.minimum(1.0, ranks.astype(float) / R_max)
    delta_pc = 0.15 * (p_cmax - p_cmin)
    delta_pm = 0.15 * (p_mmax - p_mmin)

    Cross_Rate = base_pc - delta_pc * r_tilde
    Mut_Rate = base_pm + delta_pm * r_tilde
    Cross_Rate = np.clip(Cross_Rate, p_cmin, p_cmax).astype(float)
    Mut_Rate = np.clip(Mut_Rate, p_mmin, p_mmax).astype(float)



def environmental_selection_1v1(parents_pop, parents_F, parents_CV,
                                offspring, off_F, off_CV):
    """
    1v1 环境选择（下标与繁殖父代对齐）：
      - 子代严格支配父代 → 换子代
      - 父代严格支配子代 → 留父代
      - 互不支配 → 在 2N 并集上算拥挤度，拥挤度更大者胜；相等则随机
    """
    N = parents_pop.shape[0]
    next_pop = parents_pop.copy()
    next_F = parents_F.copy()
    next_CV = parents_CV.copy()

    # 严格支配：两维 <= 且至少一维 <
    off_dom_par = (np.all(off_F <= parents_F, axis=1)
                   & np.any(off_F < parents_F, axis=1))
    par_dom_off = (np.all(parents_F <= off_F, axis=1)
                   & np.any(parents_F < off_F, axis=1))

    # 互不支配
    nondom = ~(off_dom_par | par_dom_off)

    # 2N 并集拥挤度（把全体先当成同一层来估稀疏程度）
    comb_F = np.vstack([parents_F, off_F])
    cd = crowding_distance(comb_F, np.arange(2 * N))
    cd_p = cd[:N]
    cd_o = cd[N:]

    # 互不支配：拥挤度更大者；相等随机
    pick_off_nd = cd_o > cd_p
    tie = cd_o == cd_p
    pick_off_nd = pick_off_nd | (tie & (np.random.rand(N) < 0.5))

    replace = off_dom_par | (nondom & pick_off_nd)

    next_pop[replace] = offspring[replace]
    next_F[replace] = off_F[replace]
    next_CV[replace] = off_CV[replace]
    return next_pop, next_F, next_CV








def reset_run_state(pop_size):
    global Pop_Size, Cross_Rate, Mut_Rate, Stag_Counter
    global Archive, Archive_F, Archive_CV, hv_history_window, Fronts
    global _elite_pc_base, _elite_pm_base
    Pop_Size = pop_size
    Cross_Rate = np.full(Pop_Size, 0.80, dtype=float)
    Mut_Rate = np.full(Pop_Size, 0.15, dtype=float)
    Stag_Counter = 0
    Archive = Archive_F = Archive_CV = None
    hv_history_window = []
    Fronts = None
    _elite_pc_base, _elite_pm_base = 0.80, 0.15




def default_fit_params():
    return {
        "w_cost": 0.5, "w_service": 0.5,
        "C_fixed": 50000.0, "C_pile": 5000.0,
        "Service_Rate": 10.0, "Max_Dist": 30.0,
        "P_Grid": 1e6, "P_Coverage": 1e7,
        "beta_dist": 0.1, "gamma_wait": 0.05,
        "C_wait": 500.0, "P_Unstable": 1e8,
        "alpha": 0.5, "Travel_Speed": 30.0,
    }


def evolve_one_generation(attraction_matrix, gen, max_gen):
    global MainPop, MainFit, MainCV, Fronts, Stag_Counter
    global Fit_Params_MO, Cross_Rate, Mut_Rate, Archive, Archive_F

    Fit_Params_MO["Current_Gen"] = gen + 1
    N = MainPop.shape[0]

    parents_pop = MainPop.copy()
    parents_F = MainFit.copy()
    parents_CV = MainCV.copy()
    parents_pc = np.asarray(Cross_Rate, float).reshape(-1).copy()
    parents_pm = np.asarray(Mut_Rate, float).reshape(-1).copy()
    if parents_pc.size == 1:
        parents_pc = np.full(N, float(parents_pc[0]))
    if parents_pm.size == 1:
        parents_pm = np.full(N, float(parents_pm[0]))

    idx = binary_tournament_indices(N)
    mate = parents_pop[idx].copy()
    Cross_Rate = parents_pc[idx].copy()
    Mut_Rate = parents_pm[idx].copy()

    offspring = mate.copy()
    crossover(offspring)
    mutation(offspring)
    repair(offspring)
    off_F, off_CV = evaluate_population(offspring)

    MainPop, MainFit, MainCV = environmental_selection_1v1(
        parents_pop, parents_F, parents_CV, offspring, off_F, off_CV)

    # 占位；真正个体率由末尾 adaptive_control 按新秩写入
    Cross_Rate = np.full(N, float(np.mean(parents_pc)))
    Mut_Rate = np.full(N, float(np.mean(parents_pm)))

    if (globals().get("LS_ENABLED", True)
            and Archive is not None and len(Archive) > 0
            and gen > int(0.15 * max_gen) and gen % 5 == 0):
        attr = (attraction_matrix if attraction_matrix is not None
                else build_attraction())
        local_search(attr, intensity=0.5, gen=gen, max_gen=max_gen)

    prev_len = len(Fronts[0]) if Fronts else 0
    prev_AF = None if Archive_F is None else Archive_F.copy()
    update_archive(MainPop, MainFit, MainCV)

    improved = _archive_improved(prev_AF, Archive_F)
    if (globals().get("LS_ENABLED", True)
            and (not improved)
            and Archive is not None and len(Archive) > 0
            and gen > int(0.15 * max_gen)):
        inject_elites()

    Fronts, _ = non_dominated_sort(MainFit, MainCV)
    if len(Fronts[0]) <= prev_len:
        Stag_Counter += 1
    else:
        Stag_Counter = 0

    adaptive_control()

# =============================================================================
# 主演化入口
# =============================================================================
def run(problem_file, pop_size=100, max_gen=300, loop_times=1, verbose=True):
    global MainPop, MainFit, MainCV, Archive, Archive_F, Fit_Params_MO, Max_Gen

    Max_Gen = max_gen
    Fit_Params_MO = default_fit_params()
    load_dataset(problem_file)
    attraction = build_attraction()

    all_pareto_fronts, all_times, all_loop_histories = [], [], []

    for loop in range(loop_times):
        t0 = time.time()
        reset_run_state(pop_size)
        history = []

        MainPop = initialize_population(attraction)
        MainFit, MainCV = evaluate_population(MainPop)
        update_archive(MainPop, MainFit, MainCV)
        history.append(Archive_F.copy() if Archive_F is not None else np.zeros((0, 2)))

        if verbose:
            print(
                f"  Loop {loop+1}/{loop_times}: init "
                f"F1[{MainFit[:,0].min():.0f},{MainFit[:,0].max():.0f}] "
                f"F2[{MainFit[:,1].min():.1f},{MainFit[:,1].max():.1f}]",
                flush=True,
            )

        for gen in range(max_gen):
            evolve_one_generation(attraction, gen, max_gen)
            history.append(Archive_F.copy() if Archive_F is not None else np.zeros((0, 2)))
            if verbose and (gen % 5 == 0 or gen == max_gen - 1):
                print(
                    f"    Gen {gen+1:4d}/{max_gen} |A|={0 if Archive is None else len(Archive)} "
                    f"F1[{MainFit[:,0].min():.0f},{MainFit[:,0].max():.0f}] "
                    f"F2[{MainFit[:,1].min():.1f},{MainFit[:,1].max():.1f}] "
                    f"Stag={Stag_Counter}",
                    flush=True,
                )

        post_processing_refine()
        all_pareto_fronts.append((Archive.copy(), Archive_F.copy()))
        all_times.append(time.time() - t0)
        all_loop_histories.append(history)

    return all_pareto_fronts, all_times, all_loop_histories


if __name__ == "__main__":
    problem_path = "EV_Problems/chn31_EV.json"
    if not os.path.exists(problem_path):
        print("missing", problem_path)
        sys.exit(1)
    fronts, times, _ = run(problem_path, pop_size=50, max_gen=50, loop_times=1)
    print("done, time=", times, "|PF|=", len(fronts[0][1]))