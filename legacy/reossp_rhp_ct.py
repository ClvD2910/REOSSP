import random
import time
from gurobipy import Model, GRB, quicksum


def generate_toy_instance(S=8, K=5, P=3, G=2, J_options=20, T_per_stage=500, seed=32):
    random.seed(seed)
    data = {}
    data["S"] = S
    data["K"] = list(range(1, K+1))
    data["P"] = list(range(1, P+1))
    data["G"] = list(range(1, G+1))
    data["T_s"] = {s: list(range(1, T_per_stage+1)) for s in range(1, S+1)}
    data["J"] = { (s,k): list(range(1, J_options+1)) for s in range(1, S+1) for k in data["K"] }

    V = {}
    W = {}
    H = {}
    for s in range(1, S+1):
        for k in data["K"]:
            for j in data["J"][(s,k)]:
                for t in data["T_s"][s]:
                    # Random visibility for targets (y)
                    if random.random() < 0.5:
                        p = random.choice(data["P"])
                        V[(s,k,t,j,p)] = 1
                    # Random visibility for ground stations (q)
                    if random.random() < 0.25:
                        g = random.choice(data["G"])
                        W[(s,k,t,j,g)] = 1
                    # Random visibility for Sun (h) 
                    H[(s,k,t,j)] = 1 if random.random() < 0.5 else 0
    data["V"] = V
    data["W"] = W
    data["H"] = H

    data["Dobs"] = 102.50
    data["Dcomm"] = 100.0
    data["Bobs"] = 16.26
    data["Bcomm"] = 1.20
    data["Bcharge"] = 41.48
    data["Btime"] = 2.0
    data["Brecon"] = 0.50 
    data["Dmin"] = 0.0
    data["Dmax"] = 128.0
    data["Bmin"] = 0.0
    data["Bmax"] = 1647.0

    c = {}
    for s in range(1, S+1):
        for k in data["K"]:
            # J[s-1, k] 
            prev_s = s - 1 if s > 1 else 1 # Simplified, as J_options is fixed
            # Assume initial slot at s=1 is always the first option for simplicity
            prev_list = data["J"][(prev_s, k)] if s>1 else [data["J"][(1, k)][0]] 
            for i in prev_list:
                for j in data["J"][(s,k)]:
                    # Cost c_ij^sk in km/s (as used in the paper for c_max^k)
                    c[(s,k,i,j)] = 0.0 if i==j else random.uniform(0.01, 0.3)
    data["c"] = c
    data["c_k_max"] = {k: 0.75 for k in data["K"]} # Initial total budget in km/s

    data["d_init"] = {k: data["Dmin"] for k in data["K"]}
    data["b_init"] = {k: data["Bmax"] for k in data["K"]}

    data["C"] = 2.0  # weight for downlink
    return data

def solve_subproblem(data, s, L, J_tilde_prev, time_limit=None, verbose=False):
    model = Model(f"RHP_s{s}_L{L}")
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    S = data["S"]
    K_list = data["K"]
    P_list = data["P"]
    G_list = data["G"]
    C = data["C"]
    Dobs = data["Dobs"]
    Dcomm = data["Dcomm"]
    Brecon = data["Brecon"] 

    # lookahead set: s .. min(s+L-1, S)
    L_end = min(s + L - 1, S)
    Lset = list(range(s, L_end + 1))
    T_s = {ell: data["T_s"][ell] for ell in Lset}

    x = {}  
    y = {}  
    q = {}  
    h = {}  
    d = {}  
    b = {}  

    # Variables definition 
    for ell in Lset:
        for k in K_list:
            prev_i_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
            for i in prev_i_list:
                for j in data["J"][(ell, k)]:
                    x[(ell, k, i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{ell}_{k}_{i}_{j}")

    for ell in Lset:
        for k in K_list:
            for t in T_s[ell]:
                d[(ell, k, t)] = model.addVar(lb=data["Dmin"], ub=data["Dmax"], vtype=GRB.CONTINUOUS, name=f"d_{ell}_{k}_{t}")
                b[(ell, k, t)] = model.addVar(lb=data["Bmin"], ub=data["Bmax"], vtype=GRB.CONTINUOUS, name=f"b_{ell}_{k}_{t}")
                h[(ell, k, t)] = model.addVar(vtype=GRB.BINARY, name=f"h_{ell}_{k}_{t}")
                for p in P_list:
                    y[(ell, k, t, p)] = model.addVar(vtype=GRB.BINARY, name=f"y_{ell}_{k}_{t}_{p}")
                for gidx in G_list:
                    q[(ell, k, t, gidx)] = model.addVar(vtype=GRB.BINARY, name=f"q_{ell}_{k}_{t}_{gidx}")

    model.update()

    # OBJECTIVE
    obj_terms = []
    for key, val in q.items():
        obj_terms.append(C * val)
    for key, val in y.items():
        obj_terms.append(val)
    model.setObjective(quicksum(obj_terms), GRB.MAXIMIZE)

    # CONSTRAINTS
    
    # 1) continuity (Constraint 18a for first stage, adapted for toy example)
    for k in K_list:
        i = J_tilde_prev[k]
        expr = quicksum(x[(s, k, i, j)] for j in data["J"][(s, k)] if (s, k, i, j) in x)
        model.addConstr(expr == 1, name=f"cont_first_{s}_{k}")

    # 2) continuity chain (Constraint 18b)
    for ell in Lset:
        if ell == max(Lset):
            continue
        for k in K_list:
            for i in data["J"][(ell, k)]:
                # Left side: transfer from i to j in next stage (ell+1)
                left_terms = [ x[(ell+1, k, i, j)] for j in data["J"][(ell+1, k)] if (ell+1, k, i, j) in x ]
                # Right side: transfer from j' to i in current stage (ell)
                prev_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
                right_terms = [ x[(ell, k, jprev, i)] for jprev in prev_list if (ell, k, jprev, i) in x ]
                model.addConstr(quicksum(left_terms) - quicksum(right_terms) == 0, name=f"cont_chain_{ell}_{k}_{i}")

    # 3) transfer budget (Constraint 18c, assuming c_k_max is the remaining budget)
    for k in K_list:
        expr_terms = []
        for (ell_, kk, i, j) in x.keys():
            if kk != k:
                continue
            cost = data["c"].get((ell_, kk, i, j), 0.0)
            expr_terms.append(cost * x[(ell_, kk, i, j)])
        model.addConstr(quicksum(expr_terms) <= data["c_k_max"][k], name=f"transfer_budget_{k}")

    # 4) time-window constraints (Constraints 19a, 19b, 19c)
    for (ell, k, t, p), vary in y.items():
        lhs = []
        prev_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
        for i in prev_list:
            for j in data["J"][(ell, k)]:
                if data["V"].get((ell, k, t, j, p), 0) == 1 and (ell, k, i, j) in x:
                    lhs.append(x[(ell, k, i, j)])
        if not lhs:
            model.addConstr(vary == 0, name=f"no_vis_y_{ell}_{k}_{t}_{p}")
        else:
            model.addConstr(quicksum(lhs) >= vary, name=f"vtw_y_{ell}_{k}_{t}_{p}")

    for (ell, k, t, gidx), varq in q.items():
        lhs = []
        prev_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
        for i in prev_list:
            for j in data["J"][(ell, k)]:
                if data["W"].get((ell, k, t, j, gidx), 0) == 1 and (ell, k, i, j) in x:
                    lhs.append(x[(ell, k, i, j)])
        if not lhs:
            model.addConstr(varq == 0, name=f"no_vis_q_{ell}_{k}_{t}_{gidx}")
        else:
            model.addConstr(quicksum(lhs) >= varq, name=f"vtw_q_{ell}_{k}_{t}_{gidx}")

    for (ell, k, t), varh in h.items():
        lhs = []
        prev_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
        for i in prev_list:
            for j in data["J"][(ell, k)]:
                if data["H"].get((ell, k, t, j), 0) >= 0.5 and (ell, k, i, j) in x:
                    lhs.append(x[(ell, k, i, j)])
        if not lhs:
            model.addConstr(varh == 0, name=f"no_vis_h_{ell}_{k}_{t}")
        else:
            model.addConstr(quicksum(lhs) >= varh, name=f"vtw_h_{ell}_{k}_{t}")

    # 5) at most one activity per time step (Constraint 19d)
    for ell in Lset:
        for k in K_list:
            for t in T_s[ell]:
                expr = quicksum(y[(ell, k, t, p)] for p in P_list) + quicksum(q[(ell, k, t, gidx)] for gidx in G_list) + h[(ell, k, t)]
                model.addConstr(expr <= 1, name=f"at_most_one_{ell}_{k}_{t}")

    # 6) data tracking (Constraints 20a, 20b, 20c, 20d)
    for ell in Lset:
        Tlist = T_s[ell]
        for k in K_list:
            for t in Tlist:
                term_data = Dobs * quicksum(y[(ell, k, t, p)] for p in P_list) - Dcomm * quicksum(q[(ell, k, t, gidx)] for gidx in G_list)
                # Time step t=1
                if t == 1:
                    if ell == s: 
                        if max(Tlist) > 1:
                            model.addConstr(d[(ell, k, t+1)] == d[(ell, k, t)] + term_data, name=f"data_update_init_{ell}_{k}_{t}")
                # Time steps t=2 to Tend-1 (Constraint 20b/20c)
                elif t > 1 and t < max(Tlist):
                    model.addConstr(d[(ell, k, t+1)] == d[(ell, k, t)] + term_data, name=f"data_update_mid_{ell}_{k}_{t}")

    # Stage boundary for data (Constraint 20d)
    for ell in Lset:
        if ell == max(Lset):
            continue
        Tend = max(T_s[ell])
        for k in K_list:
            term_data = Dobs * quicksum(y[(ell, k, Tend, p)] for p in P_list) - Dcomm * quicksum(q[(ell, k, Tend, gidx)] for gidx in G_list)
            model.addConstr(d[(ell+1, k, 1)] == d[(ell, k, Tend)] + term_data, name=f"data_stage_bound_{ell}_{k}")

    # 7) data bounds after action (Constraints 20e, 20f)
    for (ell, k, t), dv in d.items():
        model.addConstr(dv + Dobs * quicksum(y[(ell, k, t, p)] for p in P_list) <= data["Dmax"], name=f"data_upper_{ell}_{k}_{t}")
        model.addConstr(dv - Dcomm * quicksum(q[(ell, k, t, gidx)] for gidx in G_list) >= data["Dmin"], name=f"data_lower_{ell}_{k}_{t}")

    # 8) battery tracking (Constraints 21a, 21b, 21c, 21d)
    for ell in Lset:
        Tlist = T_s[ell]
        for k in K_list:
            for t in Tlist:
                term_actions = data["Bcharge"] * h[(ell, k, t)] - data["Bobs"] * quicksum(y[(ell, k, t, p)] for p in P_list) - data["Bcomm"] * quicksum(q[(ell, k, t, gidx)] for gidx in G_list) - data["Btime"]
                # Time step t=1
                if t == 1 and max(Tlist) > 1:
                    if ell == s: 
                        model.addConstr(b[(ell, k, t+1)] == b[(ell, k, t)] + term_actions, name=f"batt_update_init_{ell}_{k}_{t}")
                # Time steps t=2 to Tend-1 (Constraint 21b/21c)
                elif t > 1 and t < max(Tlist):
                    model.addConstr(b[(ell, k, t+1)] == b[(ell, k, t)] + term_actions, name=f"batt_update_mid_{ell}_{k}_{t}")

    # battery stage boundary (Constraint 21d)
    for ell in Lset:
        if ell == max(Lset):
            continue
        Tend = max(T_s[ell])
        for k in K_list:
            # Recon cost term: sum(B_recon * x_ij) for the next stage's maneuver (ell+1)
            recon_cost_term = quicksum(Brecon * x[(ell+1, k, i, j)] 
                                       for i in data["J"][(ell, k)] 
                                       for j in data["J"][(ell+1, k)] 
                                       if (ell+1, k, i, j) in x)
            
            term_actions = data["Bcharge"] * h[(ell, k, Tend)] - data["Bobs"] * quicksum(y[(ell, k, Tend, p)] for p in P_list) - data["Bcomm"] * quicksum(q[(ell, k, Tend, gidx)] for gidx in G_list) - data["Btime"]
            
            model.addConstr(b[(ell+1, k, 1)] == b[(ell, k, Tend)] + term_actions - recon_cost_term,
                            name=f"batt_stage_bound_{ell}_{k}")

    # 9) battery storage limits (Constraints 22a, 22b, 22c, 22d)
    for (ell, k, t), bv in b.items():
        # 22a: Max limit after charging
        model.addConstr(bv + data["Bcharge"] * h[(ell, k, t)] <= data["Bmax"], name=f"batt_upper_{ell}_{k}_{t}")
        # 22b: Min limit after drawing power, before maneuver cost 
        model.addConstr(bv - data["Bobs"] * quicksum(y[(ell, k, t, p)] for p in P_list) - data["Bcomm"] * quicksum(q[(ell, k, t, gidx)] for gidx in G_list) - data["Btime"] >= data["Bmin"],
                        name=f"batt_lower_{ell}_{k}_{t}")

    # 22c: Min limit check at stage gap (including next stage's maneuver cost)
    for ell in Lset:
        if ell == max(Lset):
            continue
        Tend = max(T_s[ell])
        for k in K_list:
            # Recon cost term: sum(B_recon * x_ij) for the next stage's maneuver (ell+1)
            recon_cost_term = quicksum(Brecon * x[(ell+1, k, i, j)] 
                                       for i in data["J"][(ell, k)] 
                                       for j in data["J"][(ell+1, k)] 
                                       if (ell+1, k, i, j) in x)
            
            draw_cost = data["Bobs"] * quicksum(y[(ell, k, Tend, p)] for p in P_list) + data["Bcomm"] * quicksum(q[(ell, k, Tend, gidx)] for gidx in G_list) + data["Btime"]

            model.addConstr(b[(ell, k, Tend)] - draw_cost - recon_cost_term >= data["Bmin"],
                            name=f"batt_lower_stage_gap_{ell}_{k}")

    # 22d: Min limit check for the first stage's initial maneuver (s=1 only for this toy logic)
    if s == 1:
        for k in K_list:
            recon_cost_term = quicksum(Brecon * x[(s, k, i, j)] 
                                       for i in [J_tilde_prev[k]] # Initial slot
                                       for j in data["J"][(s, k)] 
                                       if (s, k, i, j) in x)
            model.addConstr(data["Bmax"] - recon_cost_term >= data["Bmin"], name=f"batt_lower_init_maneuver_{k}")
    
    # 10) initial conditions:
    for k in K_list:
        model.addConstr(d[(s, k, 1)] == data["d_init"][k], name=f"init_d_{k}")
        model.addConstr(b[(s, k, 1)] == data["b_init"][k], name=f"init_b_{k}")


    # Optimization with time measurement
    start_time = time.time()
    model.update()
    model.optimize()
    end_time = time.time()
    runtime = end_time - start_time
    
    return model, x, y, q, h, d, b, runtime

def solve(data, L=1, time_limit_per_sub=30, verbose=False):
    S = data["S"]
    K_list = data["K"]
    # initialize J_tilde_prev, assuming initial position is the first slot of stage 1's options
    J_tilde_prev = {k: data["J"][(1, k)][0] for k in K_list} 

    x_tilde = {}
    y_tilde = {}
    q_tilde = {}
    h_tilde = {}
    d_tilde = {}
    b_tilde = {}
    z_history = []
    runtime_history = [] 
    total_runtime = 0.0 
    
    # Metrics for the final schedule (only using decisions from the first stage (s) of each subproblem)
    total_downlink_data_MB = 0.0
    total_propellant_used_kms = 0.0

    # Loop s=1 to S-L+1 (the last possible control stage)
    for s in range(1, S - L + 2): 
        if verbose:
            print(f"\n--- Solving RHP(s={s}, L={L}) ---")
        
        model, xvars, yvars, qvars, hvars, dvars, bvars, sub_runtime = solve_subproblem(data, s, L, J_tilde_prev, time_limit=time_limit_per_sub, verbose=verbose)
        
        runtime_history.append(sub_runtime)
        total_runtime += sub_runtime

        if model.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            if verbose:
                print(f"Solver status for s={s}:", model.Status)
            z_history.append(None)
            # If subproblem fails, we cannot extract stage s decisions. Stop the process.
            break 
            
        zval = model.ObjVal if model.ObjVal is not None else None
        z_history.append(zval)
        if verbose:
            print(f"z_s ({s} to {min(s+L-1, S)}):", zval)

        # --- Extract and Update Logic (First Stage Decisions: ell = s) ---
        Tend = max(data["T_s"][s])
        
        # 1. Extract and update orbital choices (x_tilde, J_tilde_prev, total_propellant_used_kms)
        for k in K_list:
            current_choice = None
            for (ell, kk, i, j), var in list(xvars.items()):
                if ell == s and kk == k:
                    val = int(round(var.X)) if hasattr(var, "X") and var.X is not None else 0
                    if val == 1:
                        current_choice = (i, j)
                        break
            
            # Record decision and update J_tilde_prev
            if current_choice:
                i, j = current_choice
                x_tilde[(s, k)] = (i, j)
                J_tilde_prev[k] = j

                # Calculate and update Propellant (c_ij^sk is cost in km/s)
                cost = data["c"].get((s, k, i, j), 0.0)
                total_propellant_used_kms += cost 
                
                # Subtract transfer cost used at stage s from remaining budget for subsequent stages
                data["c_k_max"][k] = max(0.0, data["c_k_max"][k] - cost)
        
        # 2. Extract y, q, h for stage s, and update Downlink Data (total_downlink_data_MB)
        for t in data["T_s"][s]:
            for k in K_list:
                # Downlink Data for Stage s (for all time steps t)
                q_t_k_val = 0
                for g in data["G"]:
                    var = qvars.get((s, k, t, g))
                    if var and hasattr(var, "X") and var.X is not None:
                        if int(round(var.X)) == 1:
                            q_tilde.setdefault((s, k), []).append((t, g))
                            q_t_k_val = 1
                
                total_downlink_data_MB += data["Dcomm"] * q_t_k_val
                
                # Observation and Charging for Stage s
                for p in data["P"]:
                    var = yvars.get((s, k, t, p))
                    if var and hasattr(var, "X") and var.X is not None:
                        if int(round(var.X)) == 1:
                            y_tilde.setdefault((s, k), []).append((t, p))
                
                var = hvars.get((s, k, t))
                if var and hasattr(var, "X") and var.X is not None:
                    if int(round(var.X)) == 1:
                        h_tilde.setdefault((s, k), []).append(t)
        
        # 3. Store d, b for stage s
        for (ell, k, t), var in list(dvars.items()):
            if ell == s and hasattr(var, "X") and var.X is not None:
                d_tilde[(s, k, t)] = var.X
        for (ell, k, t), var in list(bvars.items()):
            if ell == s and hasattr(var, "X") and var.X is not None:
                b_tilde[(s, k, t)] = var.X

        # 4. Update initial conditions (d_init, b_init) for next stage (s+1)
        if s < S - L + 1:
            for k in K_list:
                # Check for optimal value of d[s+1, k, 1] and b[s+1, k, 1]
                ell_next = s + 1
                
                d_next_stage_start = dvars[(ell_next, k, 1)].X if (ell_next, k, 1) in dvars and hasattr(dvars[(ell_next, k, 1)], "X") and dvars[(ell_next, k, 1)].X is not None else data["d_init"][k]
                data["d_init"][k] = d_next_stage_start
                
                b_next_stage_start = bvars[(ell_next, k, 1)].X if (ell_next, k, 1) in bvars and hasattr(bvars[(ell_next, k, 1)], "X") and bvars[(ell_next, k, 1)].X is not None else data["b_init"][k]
                data["b_init"][k] = b_next_stage_start
                

    # Final Z_RHP calculation (GB)
    total_downlink_data_GB = total_downlink_data_MB / 1024.0

    return {
        "x_tilde": x_tilde,
        "y_tilde": y_tilde,
        "q_tilde": q_tilde,
        "h_tilde": h_tilde,
        "d_tilde": d_tilde,
        "b_tilde": b_tilde,
        "z_history": z_history,
        "runtime_history": runtime_history,
        "total_runtime": total_runtime,
        "Z_RHP_GB": total_downlink_data_GB, 
        "Total_Propellant_kms": total_propellant_used_kms 
    }

if __name__ == "__main__":
    S_val = 8
    T_per_stage_val = 200
    inst = generate_toy_instance(S=S_val, K=5, P=3, G=2, J_options=20, T_per_stage=T_per_stage_val, seed=32)
    
    L_val = 1
    time_limit_per_sub_val = 60 

    print(f"--- Solving REOSSP-RHP (S={S_val}, L={L_val}, T_per_stage={T_per_stage_val}) ---")
    start_total_time = time.time()
    res = solve(inst, L=L_val, time_limit_per_sub=time_limit_per_sub_val, verbose=False)
    end_total_time = time.time()
    
    num_subproblems = S_val - L_val + 1 
    
    print("\n-- Summary --")
    print(f"Total time (Wall clock): {end_total_time - start_total_time:.2f} seconds")
    print(f"Total time (Sum of Gurobi solve times for {num_subproblems} subproblems): {res['total_runtime']:.2f} seconds")
    print("--------------------------------------------------------------------------------------")
    print(f"Figure of Merit (Z_RHP): {res['Z_RHP_GB']:.2f} GB")
    print(f"Total Propellant Used: {res['Total_Propellant_kms']:.2f} km/s")
    print("--------------------------------------------------------------------------------------")
    print("Objective history (per subproblem):", res["z_history"])
    print("First-stage orbital choices x_tilde:", res["x_tilde"])
    print("First-stage observations y_tilde:", res["y_tilde"])
    print("First-stage downlinks q_tilde:", res["q_tilde"])
    print("First-stage charges h_tilde:", res["h_tilde"])