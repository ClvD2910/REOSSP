"""
Module giải bài toán REOSSP (Reconfigurable Earth Observation Satellite Scheduling Problem) - Exact.
Bài toán tối ưu toàn cục với khả năng thay đổi quỹ đạo (orbital maneuver).
"""

import time
from gurobipy import Model, GRB, quicksum


def solve_reossp_exact(data, time_limit=3600, verbose=False):
    """
    Giải bài toán REOSSP-Exact (Tối ưu toàn cục với orbital maneuver)
    
    Args:
        data: Dictionary chứa tất cả các tham số từ generate_challenging_instance()
        time_limit: Giới hạn thời gian giải (giây)
        verbose: In thông tin chi tiết từ Gurobi
    
    Returns:
        Dictionary chứa kết quả:
        - status: Trạng thái của solver
        - runtime: Thời gian giải
        - Z_R: Giá trị hàm mục tiêu
        - Z_R_GB: Lượng dữ liệu downlink (GB)
        - Total_Propellant_kms: Tổng nhiên liệu sử dụng
    """
    model = Model("REOSSP_Exact")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = time_limit

    # Lấy các tham số từ data
    S_list = list(range(1, data["S"] + 1))
    K_list = data["K"]
    P_list = data["P"]
    G_list = data["G"]
    T_s = data["T_s_list"]
    C = data["C"]
    
    Dobs, Dcomm = data["Dobs"], data["Dcomm"]
    Bobs, Bcomm, Bcharge, Btime, Brecon = (
        data["Bobs"], data["Bcomm"], data["Bcharge"], data["Btime"], data["Brecon"]
    )
    Dmin, Dmax = data["Dmin"], data["Dmax"]
    Bmin, Bmax = data["Bmin"], data["Bmax"]

    # Khai báo biến quyết định
    x, y, q, h, d, b = {}, {}, {}, {}, {}, {}
    
    # Khai báo biến (7a-7f)
    for s in S_list:      
        for k in K_list:
            # Biến chuyển quỹ đạo
            prev_list = data["J_0"][k] if s == 1 else data["J"][(s-1, k)] 
            for i in prev_list:
                for j in data["J"][(s, k)]:
                    x[(s, k, i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{s}_{k}_{i}_{j}")
            
            for t in T_s:
                # Biến trạng thái
                d[(s, k, t)] = model.addVar(lb=Dmin, ub=Dmax, vtype=GRB.CONTINUOUS, name=f"d_{s}_{k}_{t}")
                b[(s, k, t)] = model.addVar(lb=Bmin, ub=Bmax, vtype=GRB.CONTINUOUS, name=f"b_{s}_{k}_{t}")
                h[(s, k, t)] = model.addVar(vtype=GRB.BINARY, name=f"h_{s}_{k}_{t}")
                
                # Biến quyết định quan sát
                for p in P_list:
                    y[(s, k, t, p)] = model.addVar(vtype=GRB.BINARY, name=f"y_{s}_{k}_{t}_{p}")
                
                # Biến quyết định downlink
                for gidx in G_list:
                    q[(s, k, t, gidx)] = model.addVar(vtype=GRB.BINARY, name=f"q_{s}_{k}_{t}_{gidx}")
    
    model.update()

    # Hàm Mục tiêu (Objective 8)
    model.setObjective(
        quicksum(
            quicksum(C * q.get((s, k, t, gidx), 0) for gidx in G_list) +
            quicksum(y.get((s, k, t, p), 0) for p in P_list) 
            for s in S_list for k in K_list for t in T_s
        ),
        GRB.MAXIMIZE
    )

    # Ràng buộc (10-14)
    for k in K_list:
        # 10a: First stage maneuver - phải chọn đúng 1 slot ở stage 1
        i_init = data["J_0"][k][0]
        model.addConstr(
            quicksum(x.get((1, k, i_init, j), 0) for j in data["J"][(1, k)]) == 1
        )
        
        # 10c: Total Propellant Budget - giới hạn nhiên liệu
        total_cost = quicksum(
            data["c"].get((s, k, i, j), 0) * x.get((s, k, i, j), 0)
            for s in S_list 
            for i in (data["J_0"][k] if s == 1 else data["J"][(s-1, k)]) 
            for j in data["J"][(s, k)]
        )
        model.addConstr(total_cost <= data["c_k_max"][k], name=f"transfer_budget_{k}")

        # Điều kiện ban đầu (d_1^1k và b_1^1k)
        model.addConstr(d[(1, k, 1)] == Dmin)
        recon_cost_init = quicksum(Brecon * x.get((1, k, i_init, j), 0) for j in data["J"][(1, k)])
        model.addConstr(b[(1, k, 1)] == Bmax - recon_cost_init)
        model.addConstr(Bmax - recon_cost_init >= Bmin)  # (14d)

        for s in S_list:
            prev_list = data["J_0"][k] if s == 1 else data["J"][(s-1, k)]
            next_list = data["J"][(s+1, k)] if s < max(S_list) else []
            
            # 10b: Continuity chain - liên tục giữa các stage
            if s < max(S_list):
                for i in data["J"][(s, k)]:
                    lhs = quicksum(x.get((s+1, k, i, j), 0) for j in data["J"][(s+1, k)])
                    rhs = quicksum(x.get((s, k, i_prev, i), 0) for i_prev in prev_list)
                    model.addConstr(lhs - rhs == 0)

            for t in T_s:
                # 11d: At most one task - chỉ làm 1 việc tại mỗi thời điểm
                model.addConstr(
                    quicksum(y.get((s, k, t, p), 0) for p in P_list) +
                    quicksum(q.get((s, k, t, gidx), 0) for gidx in G_list) +
                    h.get((s, k, t), 0) <= 1
                )
                
                # 11a-11c: Visibility constraints - ràng buộc tầm nhìn
                # Quan sát mục tiêu
                for p in P_list: 
                    sum_vis = quicksum(
                        data["V_R"].get((s, k, t, j, p), 0) * x.get((s, k, i, j), 0) 
                        for i in prev_list for j in data["J"][(s, k)]
                    )
                    model.addConstr(sum_vis >= y.get((s, k, t, p), 0))
                
                # Downlink tới trạm mặt đất
                for gidx in G_list: 
                    sum_vis = quicksum(
                        data["W_R"].get((s, k, t, j, gidx), 0) * x.get((s, k, i, j), 0) 
                        for i in prev_list for j in data["J"][(s, k)]
                    )
                    model.addConstr(sum_vis >= q.get((s, k, t, gidx), 0))
                
                # Sạc pin (nhìn thấy mặt trời)
                sum_h = quicksum(
                    data["H_R"].get((s, k, t, j), 0) * x.get((s, k, i, j), 0) 
                    for i in prev_list for j in data["J"][(s, k)]
                )
                model.addConstr(sum_h >= h.get((s, k, t), 0))

                # Data Constraints (12c-12d)
                data_gain = quicksum(Dobs * y.get((s, k, t, p), 0) for p in P_list)
                data_loss = quicksum(Dcomm * q.get((s, k, t, gidx), 0) for gidx in G_list)
                model.addConstr(d[(s, k, t)] + data_gain <= Dmax)  # 12c
                model.addConstr(d[(s, k, t)] - data_loss >= Dmin)   # 12d

                # Battery Constraints (13a-13b & 14a-14c)
                batt_gain = Bcharge * h.get((s, k, t), 0)
                batt_net_loss = (
                    quicksum(Bobs * y.get((s, k, t, p), 0) for p in P_list) +
                    quicksum(Bcomm * q.get((s, k, t, gidx), 0) for gidx in G_list) +
                    Btime
                )
                model.addConstr(b[(s, k, t)] + batt_gain <= Bmax)  # 14a

                if t < max(T_s):
                    # Within stage (13a, part of 14b)
                    model.addConstr(b[(s, k, t+1)] == b[(s, k, t)] + batt_gain - batt_net_loss) 
                    model.addConstr(d[(s, k, t+1)] == d[(s, k, t)] + data_gain - data_loss)  # 12a
                    model.addConstr(b[(s, k, t)] - batt_net_loss >= Bmin)  # 14b
                    
                elif t == max(T_s):
                    recon_cost_next = quicksum(
                        Brecon * x.get((s+1, k, i, j), 0) 
                        for i in data["J"][(s, k)] for j in next_list
                    )
                    if s < max(S_list):
                        # Across stages (13b, 12b, 14c)
                        model.addConstr(b[(s+1, k, 1)] == b[(s, k, t)] + batt_gain - batt_net_loss - recon_cost_next)
                        model.addConstr(d[(s+1, k, 1)] == d[(s, k, t)] + data_gain - data_loss)
                        model.addConstr(b[(s, k, t)] - batt_net_loss - recon_cost_next >= Bmin)
                    else:
                        model.addConstr(b[(s, k, t)] - batt_net_loss >= Bmin)  # 14b (t=T)

    # Tối ưu hóa
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    runtime = end_time - start_time
    
    # Thu thập kết quả
    results = {
        "status": model.Status,
        "runtime": runtime,
        "Z_R": 0,
        "Z_R_GB": 0,
        "Total_Propellant_kms": 0
    }
    
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        results["Z_R"] = model.ObjVal
        
        # Tính tổng dữ liệu downlink
        results["Z_R_GB"] = quicksum(
            Dcomm * q[(s, k, t, gidx)].X 
            for s in S_list for k in K_list for t in T_s for gidx in G_list
        ).getValue() / 1024.0
        
        # Tính tổng nhiên liệu sử dụng
        results["Total_Propellant_kms"] = quicksum(
            data["c"].get((s, k, i, j), 0) * x.get((s, k, i, j), 0).X
            for s in S_list for k in K_list 
            for i in (data["J_0"][k] if s == 1 else data["J"][(s-1, k)]) 
            for j in data["J"][(s, k)]
        ).getValue()
        
    return results


if __name__ == "__main__":
    # Test module
    from parameters import generate_challenging_instance
    
    print("Testing REOSSP-Exact Solver...")
    inst = generate_challenging_instance(S=2, K=2, T_per_stage=10, use_physics_visibility=False)
    
    print("\nSolving REOSSP-Exact...")
    results = solve_reossp_exact(inst, time_limit=60, verbose=False)
    
    print(f"\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Runtime: {results['runtime']:.2f}s")
    print(f"  Objective (Z_R): {results['Z_R']:.2f}")
    print(f"  Downlink (GB): {results['Z_R_GB']:.2f}")
    print(f"  Propellant Used (km/s): {results['Total_Propellant_kms']:.4f}")
