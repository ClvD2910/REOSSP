"""
Module tạo các tham số cho bài toán EOSSP và REOSSP.
Bao gồm tham số vệ tinh, trạm mặt đất, và các ràng buộc năng lượng/dữ liệu.
"""

import random
from visibility_generator import (
    PHYSICS_LIBS_AVAILABLE,
    generate_physics_based_visibility,
    generate_probabilistic_visibility
)


def generate_challenging_instance(S=5, K=2, P=15, G=3, J_options=20, T_per_stage=40, seed=42, 
                                   use_physics_visibility=True, dt_seconds=100):
    """
    Tạo instance với tùy chọn sử dụng ma trận tầm nhìn dựa trên vật lý
    
    Args:
        S: Số stage (giai đoạn)
        K: Số vệ tinh
        P: Số mục tiêu quan sát
        G: Số trạm mặt đất
        J_options: Số slot quỹ đạo
        T_per_stage: Số bước thời gian mỗi stage
        seed: Random seed
        use_physics_visibility: True để sử dụng thư viện vật lý (poliastro/astropy),
                                False để sử dụng xác suất ngẫu nhiên
        dt_seconds: Thời lượng mỗi bước thời gian (giây) - chỉ dùng cho physics mode
    
    Returns:
        Dictionary chứa tất cả các tham số và ma trận tầm nhìn
    """
    random.seed(seed)
    data = {}
    
    # Cấu hình cơ bản
    data["S"], data["K"] = S, list(range(1, K+1))
    data["P"], data["G"] = list(range(1, P+1)), list(range(1, G+1))
    
    # Cấu hình thời gian
    T_total = S * T_per_stage
    data["T_full"] = list(range(1, T_total + 1))
    data["T_per_stage"] = T_per_stage
    data["T_s_list"] = list(range(1, T_per_stage + 1))
    
    # Cấu hình slot quỹ đạo
    data["J"] = {(s, k): list(range(1, J_options + 1)) 
                 for s in range(1, S + 1) for k in data["K"]}
    data["J_0"] = {k: [1] for k in data["K"]} 

    # Tạo Visibility - sử dụng vật lý hoặc xác suất
    if use_physics_visibility and PHYSICS_LIBS_AVAILABLE:
        print("\n" + "="*80)
        print("GENERATING PHYSICS-BASED VISIBILITY MATRICES")
        print("Using poliastro + astropy orbital mechanics libraries")
        print("="*80)
        V_reossp, W_reossp, H_reossp, V_eossp, W_eossp, H_eossp = generate_physics_based_visibility(
            S, K, P, G, J_options, T_per_stage, seed, dt_seconds
        )
    else:
        if use_physics_visibility and not PHYSICS_LIBS_AVAILABLE:
            print("\nWarning: Physics libraries not available, falling back to probabilistic visibility")
        print("\nGenerating probabilistic visibility matrices...")
        V_reossp, W_reossp, H_reossp, V_eossp, W_eossp, H_eossp = generate_probabilistic_visibility(
            S, K, P, G, J_options, T_per_stage, seed
        )

    # Tham số truyền dữ liệu
    data["Dobs"] = 102.50   # MB/s (Tốc độ sinh dữ liệu khi quan sát)
    data["Dcomm"] = 100.0   # MB/s (Tốc độ downlink)
    
    # Tham số năng lượng
    data["Bobs"] = 16.26    # J/s (Tiêu thụ năng lượng khi quan sát)
    data["Bcomm"] = 1.20    # J/s (Tiêu thụ năng lượng khi truyền)
    data["Bcharge"] = 41.48 # J/s (Nạp năng lượng từ mặt trời)
    data["Btime"] = 2.0     # J/s (Tiêu thụ năng lượng khi idle)
    data["Brecon"] = 0.50   # J (Năng lượng cho thay đổi quỹ đạo)
    
    # Giới hạn bộ nhớ và pin
    data["Dmin"], data["Dmax"] = 0.0, 128000.0  # 128 GB
    data["Bmin"], data["Bmax"] = 200.0, 1000.0  # Giữ pin trên 20%
    
    # Chi phí nhiên liệu (Propellant kms)
    c = {}
    for s in range(1, S+1):
        for k in data["K"]:
            prev_slots = data["J_0"][k] if s == 1 else data["J"][(s-1, k)]
            for i in prev_slots:
                for j in data["J"][(s, k)]:
                    # Giả định chi phí nhảy slot (từ 0.01 đến 0.3)
                    c[(s, k, i, j)] = abs(i - j) * 0.02 
    
    data["c"] = c
    data["c_k_max"] = {k: 1.8 for k in data["K"]}  # Ngân sách nhiên liệu tối đa
    
    # Điều kiện ban đầu
    data["d_init"] = {k: 0 for k in data["K"]}     # Dữ liệu ban đầu
    data["b_init"] = {k: 1000 for k in data["K"]}  # Pin ban đầu
    data["C"] = 2.0  # Hệ số reward cho downlink
    
    # Lưu ma trận tầm nhìn
    data["V_R"], data["W_R"], data["H_R"] = V_reossp, W_reossp, H_reossp
    data["V_E"], data["W_E"], data["H_E"] = V_eossp, W_eossp, H_eossp
    
    # Lưu thông tin về phương pháp tạo visibility
    data["visibility_method"] = "physics" if (use_physics_visibility and PHYSICS_LIBS_AVAILABLE) else "probabilistic"
    
    return data


if __name__ == "__main__":
    # Test module
    print("Testing parameters generation...")
    inst = generate_challenging_instance(S=2, K=2, T_per_stage=10, use_physics_visibility=False)
    print(f"Generated instance with {inst['S']} stages, {len(inst['K'])} satellites")
    print(f"Visibility method: {inst['visibility_method']}")
    print(f"V_R entries: {len(inst['V_R'])}")
    print(f"W_R entries: {len(inst['W_R'])}")
