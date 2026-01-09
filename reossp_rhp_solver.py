"""
Module giải bài toán REOSSP (Reconfigurable Earth Observation Satellite Scheduling Problem) 
sử dụng phương pháp RHP (Rolling Horizon Procedure).
Phương pháp chia nhỏ bài toán thành các bài toán con với lookahead L stages.
"""

import time
from gurobipy import Model, GRB, quicksum


def solve_subproblem(data, s, L, J_tilde_prev, d_init_in, b_init_in, c_max_in, 
                     time_limit=None, verbose=False):
    """
    Giải bài toán con RHP(s, L)
    
    Args:
        data: Dictionary chứa tất cả các tham số
        s: Stage hiện tại
        L: Số stages lookahead
        J_tilde_prev: Dict {k: slot_đã_chọn} - vị trí quỹ đạo trước đó của mỗi vệ tinh
        d_init_in: Dict {k: dữ_liệu_ban_đầu} - dữ liệu ban đầu của mỗi vệ tinh
        b_init_in: Dict {k: pin_ban_đầu} - pin ban đầu của mỗi vệ tinh
        c_max_in: Dict {k: ngân_sách_nhiên_liệu} - ngân sách nhiên liệu còn lại
        time_limit: Giới hạn thời gian giải
        verbose: In thông tin chi tiết
    
    Returns:
        model, x, y, q, h, d, b, runtime
    """
    model = Model(f"RHP_s{s}_L{L}")
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    # Lấy các tham số
    S_full = data["S"]
    K_list = data["K"]
    P_list = data["P"]
    G_list = data["G"]
    C = data["C"]
    T_s = data["T_s_list"]
    
    Dobs, Dcomm = data["Dobs"], data["Dcomm"]
    Bobs, Bcomm, Bcharge, Btime, Brecon = (
        data["Bobs"], data["Bcomm"], data["Bcharge"], data["Btime"], data["Brecon"]
    )
    Dmin, Dmax = data["Dmin"], data["Dmax"]
    Bmin, Bmax = data["Bmin"], data["Bmax"]

    # Tính toán horizon
    L_end = min(s + L - 1, S_full)
    Lset = list(range(s, L_end + 1))
    
    # Khai báo biến
    x, y, q, h, d, b = {}, {}, {}, {}, {}, {}

    for ell in Lset:      
        for k in K_list:
            # Biến chuyển quỹ đạo
            prev_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
            for i in prev_list:
                for j in data["J"][(ell, k)]:
                    x[(ell, k, i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{ell}_{k}_{i}_{j}")
            
            for t in T_s:
                # Biến trạng thái
                d[(ell, k, t)] = model.addVar(lb=Dmin, ub=Dmax, vtype=GRB.CONTINUOUS, name=f"d_{ell}_{k}_{t}")
                b[(ell, k, t)] = model.addVar(lb=Bmin, ub=Bmax, vtype=GRB.CONTINUOUS, name=f"b_{ell}_{k}_{t}")
                h[(ell, k, t)] = model.addVar(vtype=GRB.BINARY, name=f"h_{ell}_{k}_{t}")
                
                # Biến quyết định quan sát
                for p in P_list:
                    y[(ell, k, t, p)] = model.addVar(vtype=GRB.BINARY, name=f"y_{ell}_{k}_{t}_{p}")
                
                # Biến quyết định downlink
                for gidx in G_list:
                    q[(ell, k, t, gidx)] = model.addVar(vtype=GRB.BINARY, name=f"q_{ell}_{k}_{t}_{gidx}")
    
    model.update()

    # Hàm Mục tiêu (Objective 17)
    model.setObjective(
        quicksum(
            quicksum(C * q.get((ell, k, t, gidx), 0) for gidx in G_list) +
            quicksum(y.get((ell, k, t, p), 0) for p in P_list) 
            for ell in Lset for k in K_list for t in T_s
        ),
        GRB.MAXIMIZE
    )

    # Ràng buộc (18-22)
    for k in K_list:
        i_init = J_tilde_prev[k]
        
        # 18a: First stage maneuver (s) - chọn đúng 1 slot
        model.addConstr(
            quicksum(x.get((s, k, i_init, j), 0) for j in data["J"][(s, k)]) == 1
        )
        
        # 18c: Total Propellant Budget - giới hạn nhiên liệu còn lại
        total_cost = quicksum(
            data["c"].get((ell, k, i, j), 0) * x.get((ell, k, i, j), 0)
            for ell in Lset 
            for i in ([J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]) 
            for j in data["J"][(ell, k)]
        )
        model.addConstr(total_cost <= c_max_in[k], name=f"transfer_budget_{k}")

        # Điều kiện ban đầu
        model.addConstr(d[(s, k, 1)] == d_init_in[k])
        recon_cost = quicksum(Brecon * x.get((s, k, i_init, j), 0) for j in data["J"][(s, k)])
        model.addConstr(b[(s, k, 1)] == b_init_in[k] - recon_cost)
        model.addConstr(b_init_in[k] - recon_cost >= Bmin)
        
        # Ràng buộc khác (18b, 19, 20, 21, 22)
        for ell in Lset:
            prev_list_x = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
            next_list_x = data["J"][(ell+1, k)] if ell < max(Lset) else []

            # 18b: Continuity chain
            if ell < max(Lset):
                for i in data["J"][(ell, k)]:
                    lhs = quicksum(x.get((ell+1, k, i, j), 0) for j in data["J"][(ell+1, k)])
                    rhs = quicksum(x.get((ell, k, i_prev, i), 0) for i_prev in prev_list_x)
                    model.addConstr(lhs - rhs == 0)

            for t in T_s:
                # 19d: At most one task
                model.addConstr(
                    quicksum(y.get((ell, k, t, p), 0) for p in P_list) +
                    quicksum(q.get((ell, k, t, gidx), 0) for gidx in G_list) +
                    h.get((ell, k, t), 0) <= 1
                )
                
                # 19a-19c: Visibility constraints
                for p in P_list: 
                    sum_vis = quicksum(
                        data["V_R"].get((ell, k, t, j, p), 0) * x.get((ell, k, i, j), 0) 
                        for i in prev_list_x for j in data["J"][(ell, k)]
                    )
                    model.addConstr(sum_vis >= y.get((ell, k, t, p), 0))
                
                for gidx in G_list: 
                    sum_vis = quicksum(
                        data["W_R"].get((ell, k, t, j, gidx), 0) * x.get((ell, k, i, j), 0) 
                        for i in prev_list_x for j in data["J"][(ell, k)]
                    )
                    model.addConstr(sum_vis >= q.get((ell, k, t, gidx), 0))
                
                sum_h = quicksum(
                    data["H_R"].get((ell, k, t, j), 0) * x.get((ell, k, i, j), 0) 
                    for i in prev_list_x for j in data["J"][(ell, k)]
                )
                model.addConstr(sum_h >= h.get((ell, k, t), 0))

                # Data Constraints (20e-20f)
                data_gain = quicksum(Dobs * y.get((ell, k, t, p), 0) for p in P_list)
                data_loss = quicksum(Dcomm * q.get((ell, k, t, gidx), 0) for gidx in G_list)
                model.addConstr(d[(ell, k, t)] + data_gain <= Dmax)
                model.addConstr(d[(ell, k, t)] - data_loss >= Dmin)

                # Battery Constraints (22a)
                batt_gain = Bcharge * h.get((ell, k, t), 0)
                batt_net_loss = (
                    quicksum(Bobs * y.get((ell, k, t, p), 0) for p in P_list) +
                    quicksum(Bcomm * q.get((ell, k, t, gidx), 0) for gidx in G_list) +
                    Btime
                )
                model.addConstr(b[(ell, k, t)] + batt_gain <= Bmax)

                # Tracking/Boundary
                if t < max(T_s):
                    # Within stage (20b/20c, 21b/21c, 22b)
                    model.addConstr(d[(ell, k, t+1)] == d[(ell, k, t)] + data_gain - data_loss)
                    model.addConstr(b[(ell, k, t+1)] == b[(ell, k, t)] + batt_gain - batt_net_loss)
                    model.addConstr(b[(ell, k, t)] - batt_net_loss >= Bmin)
                    
                elif t == max(T_s) and ell < max(Lset):
                    recon_cost_next = quicksum(
                        Brecon * x.get((ell+1, k, i, j), 0) 
                        for i in data["J"][(ell, k)] for j in next_list_x
                    )
                    # Across stages (20d, 21d, 22c)
                    model.addConstr(d[(ell+1, k, 1)] == d[(ell, k, t)] + data_gain - data_loss)
                    model.addConstr(b[(ell+1, k, 1)] == b[(ell, k, t)] + batt_gain - batt_net_loss - recon_cost_next)
                    model.addConstr(b[(ell, k, t)] - batt_net_loss - recon_cost_next >= Bmin)
                    
                elif t == max(T_s) and ell == max(Lset):
                    model.addConstr(b[(ell, k, t)] - batt_net_loss >= Bmin)

    # Tối ưu hóa
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    runtime = end_time - start_time
    
    return model, x, y, q, h, d, b, runtime


def solve_reossp_rhp(data, L=1, time_limit_per_sub=3600, verbose=False):
    """
    Giải bài toán REOSSP sử dụng phương pháp RHP (Rolling Horizon Procedure)
    
    Args:
        data: Dictionary chứa tất cả các tham số
        L: Số stages lookahead
        time_limit_per_sub: Giới hạn thời gian cho mỗi bài toán con
        verbose: In thông tin chi tiết
    
    Returns:
        Dictionary chứa kết quả:
        - z_history: Lịch sử giá trị mục tiêu cho từng stage
        - runtime_history: Lịch sử thời gian giải cho từng stage
        - total_runtime: Tổng thời gian giải
        - Z_RHP_GB: Lượng dữ liệu downlink (GB)
        - Total_Propellant_kms: Tổng nhiên liệu sử dụng
        - status_history: Lịch sử trạng thái solver
    """
    S_full = data["S"]
    K_list = data["K"]
    T_s = data["T_s_list"]
    
    # 1. Khởi tạo trạng thái ban đầu
    J_tilde_prev = {k: data["J_0"][k][0] for k in K_list}
    c_max_in = data["c_k_max"].copy()
    d_init_in = {k: data["Dmin"] for k in K_list}
    b_init_in = {k: data["Bmax"] for k in K_list}

    # Tổng hợp kết quả
    z_history = []
    runtime_history = []
    total_runtime = 0.0
    total_downlink_data_MB = 0.0
    total_propellant_used_kms = 0.0
    status_history = []

    # Vòng lặp RHP (s=1 đến S-L+1)
    for s in range(1, S_full - L + 2): 
        if verbose:
            print(f"\n--- Solving RHP(s={s}, L={L}) ---")

        # Giải bài toán con RHP(s, L)
        model, xvars, yvars, qvars, hvars, dvars, bvars, sub_runtime = solve_subproblem(
            data, s, L, J_tilde_prev, d_init_in, b_init_in, c_max_in,
            time_limit=time_limit_per_sub, verbose=verbose
        )
        
        runtime_history.append(sub_runtime)
        total_runtime += sub_runtime

        status = model.Status
        status_history.append(status)

        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            z_history.append(None)
            if verbose:
                print(f"  Subproblem infeasible or error: status = {status}")
            break 
            
        # Tính z chỉ cho stage s (để tránh overcount khi L > 1)
        z_stage = (
            quicksum(
                data["C"] * qvars.get((s, k, t, gidx), 0).X 
                for k in K_list for t in T_s for gidx in data["G"]
            ) +
            quicksum(
                yvars.get((s, k, t, p), 0).X 
                for k in K_list for t in T_s for p in data["P"]
            )
        )
        z_history.append(z_stage.getValue())

        # --- Trích xuất quyết định giai đoạn s và Cập nhật trạng thái ---
        Tend = max(T_s)
        
        for k in K_list:
            # 1. Cập nhật Vị trí Quỹ đạo và Nhiên liệu đã dùng
            current_choice = None
            for (ell, kk, i, j), var in list(xvars.items()):
                if ell == s and kk == k:
                    val = int(round(var.X)) if hasattr(var, "X") and var.X is not None else 0
                    if val == 1:
                        current_choice = (i, j)
                        break
            
            if current_choice:
                i, j = current_choice
                J_tilde_prev[k] = j

                cost = data["c"].get((s, k, i, j), 0.0)
                total_propellant_used_kms += cost 
                c_max_in[k] = max(0.0, c_max_in[k] - cost)
        
            # 2. Cập nhật Data Downlink
            for t in T_s:
                for g in data["G"]:
                    var = qvars.get((s, k, t, g))
                    if var and hasattr(var, "X") and var.X is not None and int(round(var.X)) == 1:
                        total_downlink_data_MB += data["Dcomm"] 

            # 3. Cập nhật Initial Conditions cho giai đoạn s+1
            if s < S_full - L + 1:
                ell_next = s + 1
                if (ell_next, k, 1) in dvars and hasattr(dvars[(ell_next, k, 1)], "X"):
                    # Cho L > 1
                    d_init_in[k] = dvars[(ell_next, k, 1)].X
                    b_init_in[k] = bvars[(ell_next, k, 1)].X
                else:
                    # Cho L == 1, tính thủ công từ end of s
                    data_gain_Tend = sum(
                        data["Dobs"] * yvars.get((s, k, Tend, p), 0).X 
                        for p in data["P"]
                    )
                    data_loss_Tend = sum(
                        data["Dcomm"] * qvars.get((s, k, Tend, g), 0).X 
                        for g in data["G"]
                    )
                    batt_gain_Tend = data["Bcharge"] * hvars.get((s, k, Tend), 0).X
                    batt_net_loss_Tend = (
                        sum(data["Bobs"] * yvars.get((s, k, Tend, p), 0).X for p in data["P"]) +
                        sum(data["Bcomm"] * qvars.get((s, k, Tend, g), 0).X for g in data["G"]) +
                        data["Btime"]
                    )
                    
                    d_end = dvars[(s, k, Tend)].X + data_gain_Tend - data_loss_Tend
                    b_end = bvars[(s, k, Tend)].X + batt_gain_Tend - batt_net_loss_Tend
                    
                    d_init_in[k] = max(data["Dmin"], min(data["Dmax"], d_end)) 
                    b_init_in[k] = max(data["Bmin"], min(data["Bmax"], b_end))

    total_downlink_data_GB = total_downlink_data_MB / 1024.0

    return {
        "z_history": z_history,
        "runtime_history": runtime_history,
        "total_runtime": total_runtime,
        "Z_RHP_GB": total_downlink_data_GB,
        "Total_Propellant_kms": total_propellant_used_kms,
        "status_history": status_history
    }


if __name__ == "__main__":
    # Test module
    from parameters import generate_challenging_instance
    
    print("Testing REOSSP-RHP Solver...")
    inst = generate_challenging_instance(S=4, K=2, T_per_stage=10, use_physics_visibility=False)
    
    print("\nSolving REOSSP-RHP with L=1...")
    results = solve_reossp_rhp(inst, L=1, time_limit_per_sub=60, verbose=False)
    
    print(f"\nResults:")
    print(f"  Total Runtime: {results['total_runtime']:.2f}s")
    print(f"  Total Objective: {sum(results['z_history']):.2f}")
    print(f"  Downlink (GB): {results['Z_RHP_GB']:.2f}")
    print(f"  Propellant Used (km/s): {results['Total_Propellant_kms']:.4f}")
    print(f"  Z History: {results['z_history']}")
