"""
=============================================================================
Fitness_Function_v3.py — 综合社会成本多目标清洁重构版 (Journal Edition)
=============================================================================
优化重点：
  1. 彻底剔除 Obj1 和 Obj2 内部的亿级硬编码惩罚项，转为纯物理量纲输出。
  2. Obj1 = 纯建设费用 (元)
  3. Obj2 = 纯车主时间成本 (行驶时间 + 稳定状态排队时间，单位：小时)
  4. 系统级超载、未覆盖、排队发散等硬约束，通过 details 反馈，统一由算法层的 CV 矩阵托管。
=============================================================================
"""

import numpy as np


# ============================================================================
# 1. 安全排队论模块
# ============================================================================

def erlang_c_prob(arrival_rate, service_rate, num_servers):
    """
    数值稳定的 Erlang-C 概率计算
    """
    arrival_rate = np.atleast_1d(np.asarray(arrival_rate, dtype=float))
    num_servers  = np.atleast_1d(np.asarray(num_servers, dtype=int))

    zero_mask = (num_servers <= 0)
    num_servers_safe = np.maximum(num_servers, 1)

    rho = arrival_rate / (num_servers_safe * service_rate)
    unstable = rho >= 1.0
    rho = np.clip(rho, 0, 1 - 1e-8)

    results = np.zeros_like(arrival_rate)

    for i in range(len(arrival_rate)):
        if zero_mask[i] or unstable[i]:
            results[i] = 1.0
            continue

        ci = int(num_servers[i])
        rho_i = float(rho[i])
        c_rho_i = ci * rho_i

        sum_terms = 1.0
        term = 1.0
        for n in range(1, ci):
            term = term * c_rho_i / n
            sum_terms += term

        term_c = 1.0
        for n in range(1, ci + 1):
            term_c = term_c * c_rho_i / n

        numerator = term_c / (1.0 - rho_i)
        denominator = sum_terms + numerator

        results[i] = numerator / denominator if denominator > 0 else 1.0

    results[zero_mask] = 1.0
    results[unstable] = 1.0
    return np.clip(results, 0.0, 1.0)


def expected_waiting_time(arrival_rate, service_rate, num_servers):
    """
    计算 M/M/c 队列的预期排队等待时间 W_q
    当系统不稳定（ρ ≥ 1）时，返回一个封顶的物理高阻抗大数（1000.0小时），绝不使用 1e10 炸毁前沿
    """
    arrival_rate = np.atleast_1d(np.asarray(arrival_rate, dtype=float))
    num_servers  = np.atleast_1d(np.asarray(num_servers, dtype=int))

    zero_mask = (num_servers <= 0)
    num_servers_safe = np.maximum(num_servers, 1)

    rho = arrival_rate / (num_servers_safe * service_rate)
    unstable = rho >= 1.0

    P_wait = erlang_c_prob(arrival_rate, service_rate, num_servers_safe)

    denominator = num_servers_safe * service_rate - arrival_rate
    denominator = np.maximum(denominator, 1e-8)

    W_q = P_wait / denominator

    # 【合理修正】系统不稳定或未配桩时，给予合理的、具有弹性的物理等待时间封顶大数
    W_q[unstable] = 1000.0
    W_q[zero_mask] = 1000.0
    W_q[arrival_rate <= 1e-10] = 0.0

    return W_q


# ============================================================================
# 2. MNL 离散选择与反馈均衡
# ============================================================================

def mnl_allocation_with_congestion_feedback(
    demands, valid_dists, station_capacities_piles,
    service_rate, max_dist, beta_dist, gamma_wait,
    max_iterations=5
):
    num_nodes    = demands.shape[0]
    num_stations = valid_dists.shape[1]

    if num_stations == 0:
        return np.zeros((num_nodes, 0)), np.array([]), np.array([]), np.array([]), 0

    distance_impedance = np.where(valid_dists <= max_dist, valid_dists, 1e4)
    perceived_wait = np.zeros(num_stations)
    damping = 0.5  # 引入中度阻尼，防止高维迭代强烈震荡

    for iteration in range(max_iterations):
        utility = -beta_dist * distance_impedance - gamma_wait * perceived_wait[np.newaxis, :]

        utility_max = np.max(utility, axis=1, keepdims=True)
        exp_utility = np.exp(utility - utility_max)
        sum_exp = np.sum(exp_utility, axis=1, keepdims=True)

        alloc_prob = np.zeros_like(exp_utility)
        np.divide(exp_utility, sum_exp, out=alloc_prob, where=(sum_exp > 0))

        station_loads = np.dot(demands, alloc_prob)

        new_wait_times = expected_waiting_time(
            station_loads, service_rate, station_capacities_piles
        )

        if iteration > 0:
            perceived_wait = damping * new_wait_times + (1 - damping) * perceived_wait
        else:
            perceived_wait = new_wait_times

        if iteration > 0 and np.max(np.abs(new_wait_times - perceived_wait)) < 1e-3:
            perceived_wait = new_wait_times
            break

    # 最终收敛输出
    utility = -beta_dist * distance_impedance - gamma_wait * perceived_wait[np.newaxis, :]
    utility_max = np.max(utility, axis=1, keepdims=True)
    exp_utility = np.exp(utility - utility_max)
    sum_exp = np.sum(exp_utility, axis=1, keepdims=True)
    alloc_prob = np.zeros_like(exp_utility)
    np.divide(exp_utility, sum_exp, out=alloc_prob, where=(sum_exp > 0))

    station_loads = np.dot(demands, alloc_prob)
    waiting_times = expected_waiting_time(station_loads, service_rate, station_capacities_piles)
    station_utils = station_loads / (np.maximum(station_capacities_piles, 1) * service_rate)

    return alloc_prob, station_loads, waiting_times, station_utils, iteration + 1


# ============================================================================
# 3. [合理化重构主算子] 多目标适应度函数
# ============================================================================

def fitness_function(X, Citys_Distance, demands, grid_caps, params):
    """
    [期刊完备平衡版] 多目标清洁型适应度函数
    """
    alpha      = params.get('alpha', 0.5)
    c_fixed    = params.get('C_fixed', 50000.0)
    c_pile     = params.get('C_pile', 5000.0)
    s_rate     = params.get('Service_Rate', 10.0)
    max_dist   = params.get('Max_Dist', 30.0)
    beta_dist  = params.get('beta_dist', 0.1)
    gamma_wait = params.get('gamma_wait', 0.05)
    travel_speed = params.get('Travel_Speed', 30.0)

    location = X[0].astype(bool)
    capacity = X[1].astype(int)
    num_nodes = len(location)

    real_capacity = np.where(location, capacity, 0)

    # ================================================================
    # 目标 1: 纯物理建设开支费用 (元)
    # ================================================================
    num_stations = int(np.sum(location))
    total_piles  = int(np.sum(real_capacity))
    cost_build   = np.sum(location.astype(float) * c_fixed + real_capacity.astype(float) * c_pile)

    # 提取纯粹的电网容量超载 kW 量，用于反馈给 CV，决不在这里乘 1e6
    grid_violation = np.sum(np.maximum(0, real_capacity - grid_caps))

    valid_indices = np.where(real_capacity > 0)[0]
    num_valid = len(valid_indices)

    # 极度退化个体兜底
    if num_valid == 0:
        return 1e12, {
            "Fitness": 1e12, "Obj1_Build": cost_build, "Obj2_User": 5000.0,
            "Stations": 0, "Piles": 0, "BuildCost": cost_build,
            "TravelTime": 500.0, "WaitingTime": 4500.0, "AvgDist": 0.0, "AvgWait": 1000.0,
            "Uncovered": float(np.sum(demands)), "Pen_Grid": 1000.0, "Is_Unstable": 1.0
        }

    valid_dists = Citys_Distance[:, valid_indices]
    valid_caps  = real_capacity[valid_indices]

    # MNL 流分分摊迭代
    alloc_prob, station_loads, waiting_times, station_utils, iters = \
        mnl_allocation_with_congestion_feedback(
            demands, valid_dists, valid_caps,
            s_rate, max_dist, beta_dist, gamma_wait, max_iterations=5
        )

    # 精准隔离覆盖度
    global_min_dists = np.min(valid_dists, axis=1)
    uncovered_mask = global_min_dists > max_dist
    uncovered_amt = np.sum(demands[uncovered_mask])
    covered_mask = ~uncovered_mask

    # ================================================================
    # 目标 2: 纯物理车主时间开销 (单位：小时) — 【核心优化：剔除一切级联放大乘法因子】
    # ================================================================
    # 2a. 计算纯粹的赶路行驶小时数
    travel_cost_matrix = alloc_prob * valid_dists
    total_travel_dist = np.sum(demands[:, np.newaxis] * travel_cost_matrix)
    total_travel_time = total_travel_dist / travel_speed
    avg_dist = (total_travel_dist / np.sum(demands[covered_mask])) if np.any(covered_mask) else 0.0

    # 2b. 计算纯粹的稳定排队等待小时数
    # 判断该方案是否包含排队不稳定的站点（即流量冲破变压器防线导致排队理论发散）
    unstable_mask = waiting_times >= 990.0
    is_unstable = 1.0 if np.any(unstable_mask) else 0.0

    # 对于正常的稳定位点累加等待小时；对于发散位点，给予合理的恒定物理时间保护值
    clean_waiting_times = np.where(unstable_mask, 10.0, waiting_times)
    total_waiting_time = np.sum(station_loads * clean_waiting_times)
    avg_wait = np.mean(waiting_times[~unstable_mask]) if np.any(~unstable_mask) else 0.0

    # 【核心重构】目标二由纯物理行驶时间与纯排队时间相加构成，量纲极其干净（小时）
    f_obj2 = total_travel_time + total_waiting_time
    f_obj1 = cost_build

    # 这里的单目标综合得分仅供兼容显示，多目标实际前沿划分会通过 Ranks 自动对齐
    fitness = 0.5 * f_obj1 + 0.5 * f_obj2

    details = {
        "Fitness":      fitness,
        "Obj1_Build":   f_obj1,      # 纯建设费用 (¥)
        "Obj2_User":    f_obj2,      # 纯车主时间成本 (Hours)
        "Stations":     num_stations,
        "Piles":        total_piles,
        "BuildCost":    cost_build,
        "TravelTime":   total_travel_time,
        "WaitingTime":  total_waiting_time,
        "AvgDist":      avg_dist,
        "AvgWait":      avg_wait,
        "Uncovered":    uncovered_amt,    # 反馈给算法层的 CV (辆)
        "Pen_Grid":     grid_violation,   # 反馈给算法层的 CV (kW)
        "Is_Unstable":  is_unstable       # 反馈给算法层的 CV 状态位
    }

    return fitness, details