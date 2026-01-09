"""
Module giải bài toán EOSSP (Earth Observation Satellite Scheduling Problem) - Exact.
Đây là baseline với quỹ đạo cố định (không có maneuver).
"""

import time
from gurobipy import Model, GRB, quicksum


def solve_eossp_exact(data, time_limit=3600, verbose=False):
    """
    Giải bài toán EOSSP-Exact (Baseline với quỹ đạo cố định)
    
    Args:
        data: Dictionary chứa tất cả các tham số từ generate_challenging_instance()
        time_limit: Giới hạn thời gian giải (giây)
        verbose: In thông tin chi tiết từ Gurobi
    
    Returns:
        Dictionary chứa kết quả:
        - status: Trạng thái của solver
        - runtime: Thời gian giải
        - Z_E: Giá trị hàm mục tiêu
        - Z_E_GB: Lượng dữ liệu downlink (GB)
    """
    model = Model("EOSSP_Exact")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = time_limit

    # Lấy các tham số từ data
    K_list = data["K"]
    P_list = data["P"]
    G_list = data["G"]
    T_full = data["T_full"]
    C = data["C"]
    
    Dobs, Dcomm = data["Dobs"], data["Dcomm"]
    Bobs, Bcomm, Bcharge, Btime = data["Bobs"], data["Bcomm"], data["Bcharge"], data["Btime"]
    Dmin, Dmax = data["Dmin"], data["Dmax"]
    Bmin, Bmax = data["Bmin"], data["Bmax"]

    # Khai báo biến quyết định
    y, q, h, d, b = {}, {}, {}, {}, {}
    
    for k in K_list:
        for t in T_full:
            # Biến trạng thái
            d[(k, t)] = model.addVar(lb=Dmin, ub=Dmax, vtype=GRB.CONTINUOUS, name=f"d_{k}_{t}")
            b[(k, t)] = model.addVar(lb=Bmin, ub=Bmax, vtype=GRB.CONTINUOUS, name=f"b_{k}_{t}")
            h[(k, t)] = model.addVar(vtype=GRB.BINARY, name=f"h_{k}_{t}")
            
            # Biến quyết định quan sát
            for p in P_list:
                y[(k, t, p)] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{t}_{p}")
            
            # Biến quyết định downlink
            for gidx in G_list:
                q[(k, t, gidx)] = model.addVar(vtype=GRB.BINARY, name=f"q_{k}_{t}_{gidx}")
    
    model.update()

    # Hàm Mục tiêu (Objective 2)
    # Maximize: C * (số lượng downlink) + (số lượng quan sát)
    model.setObjective(
        quicksum(
            quicksum(C * q.get((k, t, gidx), 0) for gidx in G_list) +
            quicksum(y.get((k, t, p), 0) for p in P_list)
            for k in K_list for t in T_full
        ),
        GRB.MAXIMIZE
    )

    # Ràng buộc
    for k in K_list:
        # Điều kiện ban đầu
        model.addConstr(d[(k, 1)] == Dmin, name=f"data_init_{k}")
        model.addConstr(b[(k, 1)] == Bmax, name=f"batt_init_{k}")
            
        for t in T_full:
            # Time Window & At Most One (4a-4d)
            # Chỉ có thể quan sát nếu có visibility
            for p in P_list:
                model.addConstr(data["V_E"].get((k, t, p), 0) >= y.get((k, t, p), 0))
            
            # Chỉ có thể downlink nếu có visibility với trạm mặt đất
            for gidx in G_list:
                model.addConstr(data["W_E"].get((k, t, gidx), 0) >= q.get((k, t, gidx), 0))
            
            # Chỉ có thể sạc pin nếu nhìn thấy mặt trời
            model.addConstr(data["H_E"].get((k, t), 0) >= h.get((k, t), 0))
            
            # Tại mỗi thời điểm, chỉ làm được 1 việc (quan sát, downlink, hoặc sạc)
            model.addConstr(
                quicksum(y.get((k, t, p), 0) for p in P_list) +
                quicksum(q.get((k, t, gidx), 0) for gidx in G_list) +
                h.get((k, t), 0) <= 1
            )
            
            # Data Constraints (5a-5c)
            data_gain = quicksum(Dobs * y.get((k, t, p), 0) for p in P_list)
            data_loss = quicksum(Dcomm * q.get((k, t, gidx), 0) for gidx in G_list)
            
            # Cập nhật dữ liệu
            if t < max(T_full):
                model.addConstr(d[(k, t+1)] == d[(k, t)] + data_gain - data_loss)
            
            # Ràng buộc bộ nhớ
            model.addConstr(d[(k, t)] + data_gain <= Dmax)
            model.addConstr(d[(k, t)] - data_loss >= Dmin)

            # Battery Constraints (6a-6c)
            batt_gain = Bcharge * h.get((k, t), 0)
            batt_net_loss = (
                quicksum(Bobs * y.get((k, t, p), 0) for p in P_list) +
                quicksum(Bcomm * q.get((k, t, gidx), 0) for gidx in G_list) +
                Btime
            )
            
            # Cập nhật pin
            if t < max(T_full):
                model.addConstr(b[(k, t+1)] == b[(k, t)] + batt_gain - batt_net_loss)
            
            # Ràng buộc pin
            model.addConstr(b[(k, t)] + batt_gain <= Bmax)
            model.addConstr(b[(k, t)] - batt_net_loss >= Bmin)

    # Tối ưu hóa
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    runtime = end_time - start_time
    
    # Thu thập kết quả
    results = {
        "status": model.Status,
        "runtime": runtime,
        "Z_E": 0,
        "Z_E_GB": 0
    }
    
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        results["Z_E"] = model.ObjVal
        
        # Tính tổng dữ liệu downlink
        downlink_MB = quicksum(
            Dcomm * q[(k, t, gidx)].X 
            for k in K_list for t in T_full 
            for gidx in G_list if (k, t, gidx) in q
        ).getValue()
        results["Z_E_GB"] = downlink_MB / 1024.0
        
    return results


if __name__ == "__main__":
    # Test module
    from parameters import generate_challenging_instance
    
    print("Testing EOSSP Solver...")
    inst = generate_challenging_instance(S=2, K=2, T_per_stage=10, use_physics_visibility=False)
    
    print("\nSolving EOSSP...")
    results = solve_eossp_exact(inst, time_limit=60, verbose=False)
    
    print(f"\nResults:")
    print(f"  Status: {results['status']}")
    print(f"  Runtime: {results['runtime']:.2f}s")
    print(f"  Objective (Z_E): {results['Z_E']:.2f}")
    print(f"  Downlink (GB): {results['Z_E_GB']:.2f}")
