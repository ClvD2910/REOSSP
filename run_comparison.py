"""
Module chạy và so sánh các phương pháp giải bài toán lập lịch vệ tinh.
So sánh giữa:
1. EOSSP-Exact (Baseline với quỹ đạo cố định)
2. REOSSP-Exact (Tối ưu toàn cục với orbital maneuver)
3. REOSSP-RHP (Rolling Horizon Procedure)
"""

from parameters import generate_challenging_instance
from eossp_solver import solve_eossp_exact
from reossp_exact_solver import solve_reossp_exact
from reossp_rhp_solver import solve_reossp_rhp
from visibility_generator import PHYSICS_LIBS_AVAILABLE


def print_header(title, width=80):
    """In header với định dạng đẹp"""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def print_separator(width=80):
    """In đường kẻ phân cách"""
    print("-" * width)


def run_comparison(S=8, K=5, T_per_stage=30, L=1, time_limit=320,
                   use_physics=True, dt_seconds=1000, verbose=False):
    """
    Chạy và so sánh 3 phương pháp giải bài toán
    
    Args:
        S: Số stages
        K: Số vệ tinh
        T_per_stage: Số bước thời gian mỗi stage
        L: Lookahead cho RHP
        time_limit: Giới hạn thời gian giải (giây)
        use_physics: Sử dụng physics-based visibility
        dt_seconds: Thời lượng mỗi bước thời gian (giây)
        verbose: In thông tin chi tiết
    
    Returns:
        Dictionary chứa kết quả của 3 phương pháp
    """
    print_header("SATELLITE OBSERVATION SCHEDULING COMPARISON")
    print(f"Configuration: S={S}, K={K}, T_per_stage={T_per_stage}, L={L}")
    
    visibility_method = 'Physics-based (poliastro/astropy)' if use_physics and PHYSICS_LIBS_AVAILABLE else 'Probabilistic'
    print(f"Visibility Method: {visibility_method}")
    print(f"Time Limit: {time_limit}s per solver")
    print("=" * 80)

    # Tạo instance
    print("\n[1/4] Generating instance...")
    inst = generate_challenging_instance(
        S=S, K=K, T_per_stage=T_per_stage, J_options=20,
        use_physics_visibility=use_physics, dt_seconds=dt_seconds
    )
    print(f"      Visibility method used: {inst['visibility_method']}")

    # Giải EOSSP-Exact
    print("\n[2/4] Solving EOSSP-Exact (Baseline)...")
    res_E = solve_eossp_exact(inst, time_limit=time_limit, verbose=verbose)
    print(f"      Completed in {res_E['runtime']:.2f}s, Z_E = {res_E['Z_E']:.2f}")
    
    # Giải REOSSP-Exact
    print("\n[3/4] Solving REOSSP-Exact (Global Optimization)...")
    res_R = solve_reossp_exact(inst, time_limit=time_limit, verbose=verbose)
    print(f"      Completed in {res_R['runtime']:.2f}s, Z_R = {res_R['Z_R']:.2f}")

    # Giải REOSSP-RHP
    print(f"\n[4/4] Solving REOSSP-RHP (Lookahead L={L})...")
    res_RHP = solve_reossp_rhp(inst, L=L, time_limit_per_sub=time_limit, verbose=verbose)
    z_rhp_total = sum(z for z in res_RHP['z_history'] if z is not None)
    print(f"      Completed in {res_RHP['total_runtime']:.2f}s, Z_RHP = {z_rhp_total:.2f}")

    # Bảng kết quả chính
    print_header("RESULTS COMPARISON")
    print(f"{'Method':<25} | {'Z (Obj)':<10} | {'Runtime':<10} | {'Fuel Used':<10} | {'Improvement':<12}")
    print_separator()
    
    z_e = res_E['Z_E']
    methods = [
        ("EOSSP-Exact (Baseline)", z_e, res_E['runtime'], 0.0),
        ("REOSSP-Exact", res_R['Z_R'], res_R['runtime'], res_R['Total_Propellant_kms']),
        (f"REOSSP-RHP (L={L})", z_rhp_total, res_RHP['total_runtime'], res_RHP['Total_Propellant_kms'])
    ]

    for name, z, rt, fuel in methods:
        imp = ((z - z_e) / z_e * 100) if z_e > 0 else 0
        print(f"{name:<25} | {z:<10.2f} | {rt:<10.2f} | {fuel:<10.4f} | {imp:>10.2f}%")
    
    print("=" * 80)
    
    # Thống kê visibility
    print_header("VISIBILITY STATISTICS")
    print(f"V_reossp (Target Visibility): {len(inst['V_R']):,} opportunities")
    print(f"W_reossp (Ground Downlink):   {len(inst['W_R']):,} opportunities")
    print(f"V_eossp (EOSSP Target):       {len(inst['V_E']):,} opportunities")
    print(f"W_eossp (EOSSP Downlink):     {len(inst['W_E']):,} opportunities")
    print("=" * 80)
    
    # Bảng dữ liệu downlink
    print_header("DATA DOWNLINK STATISTICS (GB)")
    print(f"{'Method':<25} | {'Downlink (GB)':<15}")
    print_separator()
    print(f"{'EOSSP-Exact':<25} | {res_E['Z_E_GB']:<15.4f}")
    print(f"{'REOSSP-Exact':<25} | {res_R['Z_R_GB']:<15.4f}")
    print(f"{'REOSSP-RHP':<25} | {res_RHP['Z_RHP_GB']:<15.4f}")
    print("=" * 80)
    
    # Chi tiết RHP theo stage
    if res_RHP['z_history']:
        print_header("REOSSP-RHP STAGE-BY-STAGE BREAKDOWN")
        print(f"{'Stage':<10} | {'Z_stage':<12} | {'Runtime (s)':<12} | {'Status':<10}")
        print_separator()
        for i, (z, rt, status) in enumerate(zip(
            res_RHP['z_history'], 
            res_RHP['runtime_history'],
            res_RHP['status_history']
        ), 1):
            status_str = "Optimal" if status == 2 else ("TimeLimit" if status == 9 else f"Status {status}")
            z_str = f"{z:.2f}" if z is not None else "N/A"
            print(f"{i:<10} | {z_str:<12} | {rt:<12.2f} | {status_str:<10}")
        print("=" * 80)
    
    return {
        "instance": inst,
        "eossp": res_E,
        "reossp_exact": res_R,
        "reossp_rhp": res_RHP
    }


def quick_test():
    """Chạy test nhanh với các tham số nhỏ"""
    return run_comparison(
        S=2, K=2, T_per_stage=10, L=1, time_limit=60,
        use_physics=False, verbose=False
    )


def medium_test():
    """Chạy test trung bình"""
    return run_comparison(
        S=4, K=3, T_per_stage=20, L=1, time_limit=120,
        use_physics=False, verbose=False
    )


def full_test():
    """Chạy test đầy đủ với các tham số như trong paper"""
    return run_comparison(
        S=8, K=5, T_per_stage=30, L=1, time_limit=320,
        use_physics=True, dt_seconds=1000, verbose=False
    )


if __name__ == "__main__":
    # Cấu hình mặc định
    S_val = 8
    T_val = 30
    K_val = 5
    L_val = 1
    limit = 320
    
    # Cấu hình sử dụng physics-based visibility
    USE_PHYSICS = True  # Đặt False để dùng probabilistic visibility
    DT_SECONDS = 1000   # Thời lượng mỗi bước thời gian (giây)

    # Chạy so sánh
    results = run_comparison(
        S=S_val, K=K_val, T_per_stage=T_val, L=L_val, time_limit=limit,
        use_physics=USE_PHYSICS, dt_seconds=DT_SECONDS, verbose=False
    )
    
    # Có thể truy cập kết quả chi tiết:
    # results["eossp"] - kết quả EOSSP
    # results["reossp_exact"] - kết quả REOSSP-Exact  
    # results["reossp_rhp"] - kết quả REOSSP-RHP
    # results["instance"] - dữ liệu instance
