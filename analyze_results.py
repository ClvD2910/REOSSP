"""
Script phân tích chi tiết lý do mức độ cải thiện quá lớn
"""

from parameters import generate_challenging_instance
from eossp_solver import solve_eossp_exact
from reossp_exact_solver import solve_reossp_exact
from reossp_rhp_solver import solve_reossp_rhp

def analyze_visibility_difference():
    """
    Phân tích sự khác biệt giữa EOSSP và REOSSP visibility matrices
    """
    print("="*80)
    print("ANALYZING VISIBILITY DIFFERENCE BETWEEN EOSSP AND REOSSP")
    print("="*80)
    
    # Tạo instance nhỏ để dễ phân tích
    inst = generate_challenging_instance(
        S=2, K=2, P=10, G=2, J_options=20, T_per_stage=10, 
        seed=42, use_physics_visibility=False
    )
    
    # Phân tích V_eossp vs V_reossp
    print("\n[1] ANALYZING V_eossp (BASELINE):")
    print(f"    Total V_eossp entries: {len(inst['V_E'])}")
    
    # Đếm số (k,t,p) có visibility trong EOSSP
    v_eossp_count = sum(1 for v in inst['V_E'].values() if v == 1)
    print(f"    Number of V_eossp=1: {v_eossp_count}")
    print(f"    Potential opportunities: max = K × T × P = 2 × 20 × 10 = 400")
    print(f"    Utilization: {v_eossp_count / 400 * 100:.1f}%")
    
    print("\n[2] ANALYZING V_reossp (WITH ORBITAL MANEUVERS):")
    print(f"    Total V_reossp entries: {len(inst['V_R'])}")
    
    # Đếm số (s,k,t_l,j,p) có visibility trong REOSSP
    v_reossp_count = sum(1 for v in inst['V_R'].values() if v == 1)
    print(f"    Number of V_reossp=1: {v_reossp_count}")
    print(f"    Potential opportunities: max = S × K × T_per_stage × J_options × P")
    print(f"                            = 2 × 2 × 10 × 20 × 10 = 80,000")
    print(f"    Utilization: {v_reossp_count / 80000 * 100:.1f}%")
    
    print("\n[3] KEY INSIGHT:")
    print(f"    V_eossp lấy từ: slot 10 cố định (1 slot duy nhất)")
    print(f"    V_reossp lấy từ: tất cả 20 slots")
    print(f"    Tỉ lệ: {v_reossp_count / v_eossp_count:.1f}x nhiều cơ hội")
    
    print("\n[4] DOWNLINK OPPORTUNITIES:")
    w_eossp_count = sum(1 for v in inst['W_E'].values() if v == 1)
    w_reossp_count = sum(1 for v in inst['W_R'].values() if v == 1)
    print(f"    W_eossp: {w_eossp_count} opportunities")
    print(f"    W_reossp: {w_reossp_count} opportunities")
    print(f"    Tỉ lệ: {w_reossp_count / max(w_eossp_count, 1):.1f}x")
    
    # Giải 3 solver và so sánh
    print("\n" + "="*80)
    print("SOLVING AND COMPARING:")
    print("="*80)
    
    res_E = solve_eossp_exact(inst, time_limit=60, verbose=False)
    res_R = solve_reossp_exact(inst, time_limit=60, verbose=False)
    res_RHP = solve_reossp_rhp(inst, L=1, time_limit_per_sub=60, verbose=False)
    
    z_e = res_E['Z_E']
    z_r = res_R['Z_R']
    z_rhp = sum(z for z in res_RHP['z_history'] if z is not None)
    
    print(f"\nZ_E (EOSSP):          {z_e:.2f}")
    print(f"Z_R (REOSSP-Exact):   {z_r:.2f}")
    print(f"Z_RHP (REOSSP-RHP):   {z_rhp:.2f}")
    
    print(f"\nImprovement REOSSP-Exact over EOSSP:  {(z_r - z_e) / z_e * 100:.1f}%")
    print(f"Improvement REOSSP-RHP over EOSSP:   {(z_rhp - z_e) / z_e * 100:.1f}%")
    
    # Phân tích chi tiết hàm mục tiêu
    print("\n" + "="*80)
    print("OBJECTIVE FUNCTION BREAKDOWN:")
    print("="*80)
    
    print(f"\nEOSSP:")
    print(f"  Downlink data (GB):  {res_E['Z_E_GB']:.4f} GB")
    print(f"  Observation count:   {z_e - res_E['Z_E_GB']*200:.0f}")  # C=2.0, 1GB=200 units
    print(f"  Total Z_E:           {z_e:.2f}")
    
    print(f"\nREOSSP-Exact:")
    print(f"  Downlink data (GB):  {res_R['Z_R_GB']:.4f} GB")
    print(f"  Fuel cost:           {res_R['Total_Propellant_kms']:.4f} kms")
    
    print(f"\nREOSSP-RHP:")
    print(f"  Downlink data (GB):  {res_RHP['Z_RHP_GB']:.4f} GB")
    print(f"  Fuel cost:           {res_RHP['Total_Propellant_kms']:.4f} kms")
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("="*80)
    print("""
Lý do mức độ cải thiện lớn (trên 100%):

1. EOSSP chỉ dùng 1 slot cố định (slot 10)
   → Tầm nhìn bị giới hạn rất nhiều

2. REOSSP có thể thay đổi quỹ đạo (dùng 20 slots)
   → Có nhiều cơ hội quan sát và downlink hơn

3. V_eossp chỉ là TẬP CON của V_reossp
   → REOSSP có tất cả cơ hội của EOSSP CỘNG THÊM những cơ hội từ các slot khác

4. Chi phí nhiên liệu (propellant) thường rất nhỏ
   → Lợi ích từ thay đổi quỹ đạo > Chi phí nhiên liệu

Để so sánh công bằng hơn:
- So sánh REOSSP với EOSSP mở rộng (EOSSP+: EOSSP + khả năng dùng tất cả slots)
- Hoặc giả định EOSSP có quyền chọn bất kỳ slot nào từ đầu
    """)

if __name__ == "__main__":
    analyze_visibility_difference()
