import random
import time
import numpy as np
from gurobipy import Model, GRB, quicksum

# Import thư viện vật lý chuyên dụng cho tính toán quỹ đạo
try:
    from astropy import units as u
    from astropy.time import Time
    from astropy.coordinates import EarthLocation, GCRS, ITRS, AltAz, CartesianRepresentation
    from poliastro.bodies import Earth
    from poliastro.twobody import Orbit
    PHYSICS_LIBS_AVAILABLE = True
except ImportError:
    PHYSICS_LIBS_AVAILABLE = False
    print("Warning: astropy/poliastro not installed. Using probabilistic visibility generation.")

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


# =====================================================================
# VISIBILITY MATRIX GENERATOR USING PHYSICS-BASED ORBITAL MECHANICS
# =====================================================================

class PhysicsBasedVisibilityGenerator:
    """
    Tạo ma trận tầm nhìn dựa trên cơ học quỹ đạo thực tế
    Sử dụng poliastro cho truyền bá quỹ đạo và astropy cho chuyển đổi tọa độ
    """
    
    def __init__(self, n_satellites, n_targets, n_ground_stations,
                 start_time, time_steps, dt_seconds, n_slots=20, seed=42):
        """
        Khởi tạo bộ tạo ma trận tầm nhìn
        
        Args:
            n_satellites: Số lượng vệ tinh trong chòm sao
            n_targets: Số mục tiêu quan sát  
            n_ground_stations: Số trạm mặt đất
            start_time: Thời gian bắt đầu nhiệm vụ (astropy Time)
            time_steps: Số bước thời gian mô phỏng
            dt_seconds: Thời lượng mỗi bước thời gian (giây)
            n_slots: Số slot quỹ đạo (J_options)
            seed: Random seed
        """
        self.K = n_satellites
        self.n_targets = n_targets
        self.n_ground = n_ground_stations
        self.start_time = start_time
        self.T = time_steps
        self.dt = dt_seconds
        self.n_slots = n_slots
        self.seed = seed
        
        # Tạo lưới thời gian
        self.time_grid = start_time + np.arange(time_steps) * dt_seconds * u.s
        
        # Khởi tạo vệ tinh, mục tiêu và trạm mặt đất
        self.satellites_per_slot = self._create_satellite_constellation_per_slot()
        self.targets = self._create_targets()
        self.ground_stations = self._create_ground_stations()
    
    def _create_satellite_constellation_per_slot(self):
        """
        Tạo chòm vệ tinh với nhiều cấu hình slot quỹ đạo
        Mỗi slot đại diện cho một cấu hình quỹ đạo khác nhau (thay đổi RAAN/true anomaly)
        
        Returns:
            Dict[slot_id -> List[Orbit]] - Quỹ đạo vệ tinh cho mỗi slot
        """
        np.random.seed(self.seed)
        satellites_per_slot = {}
        
        # Thông số quỹ đạo đồng bộ mặt trời (điển hình cho quan sát Trái Đất)
        altitude = 600 * u.km
        a = Earth.R + altitude
        ecc = 0.001 * u.one
        inc = 97.8 * u.deg  # Góc nghiêng đồng bộ mặt trời tại 600km
        
        for slot_j in range(1, self.n_slots + 1):
            satellites = []
            
            # Mỗi slot có offset RAAN khác nhau (mô phỏng cơ động quỹ đạo)
            slot_raan_offset = (slot_j - 1) * (360.0 / self.n_slots / 2)  # Phân bố slot
            
            for k in range(self.K):
                # Phân bố vệ tinh trên các mặt phẳng quỹ đạo khác nhau
                base_raan = (k * 360.0 / self.K)
                raan = (base_raan + slot_raan_offset) * u.deg
                argp = 0 * u.deg
                # Offset pha trong quỹ đạo + offset slot
                nu = ((k * 60) + (slot_j - 1) * 5) * u.deg
                
                orbit = Orbit.from_classical(
                    Earth,
                    a=a, ecc=ecc, inc=inc,
                    raan=raan, argp=argp, nu=nu,
                    epoch=self.start_time
                )
                satellites.append(orbit)
            
            satellites_per_slot[slot_j] = satellites
        
        return satellites_per_slot
    
    def _create_targets(self):
        """
        Tạo mục tiêu quan sát (cặp vĩ độ, kinh độ)
        Phân bố toàn cầu với thiên hướng về vĩ độ trung bình
        """
        np.random.seed(self.seed + 100)
        targets = []
        
        for _ in range(self.n_targets):
            # Sử dụng phân phối beta để tập trung vào vĩ độ trung bình
            lat = (np.random.beta(2, 2) - 0.5) * 120  # -60 đến +60 độ
            lon = np.random.uniform(-180, 180)
            targets.append((lat, lon))
        
        return targets
    
    def _create_ground_stations(self):
        """
        Tạo vị trí trạm mặt đất
        Bao gồm các trạm thực tế + ngẫu nhiên
        """
        np.random.seed(self.seed + 200)
        
        # Một số vị trí trạm mặt đất thực tế
        known_stations = [
            EarthLocation(lat=37.4*u.deg, lon=-122.1*u.deg, height=0*u.m),   # California
            EarthLocation(lat=52.2*u.deg, lon=0.1*u.deg, height=0*u.m),      # UK
            EarthLocation(lat=-35.3*u.deg, lon=149.1*u.deg, height=0*u.m),   # Australia
            EarthLocation(lat=35.7*u.deg, lon=139.7*u.deg, height=0*u.m),    # Japan
            EarthLocation(lat=28.5*u.deg, lon=-80.6*u.deg, height=0*u.m),    # Florida
        ]
        
        ground_stations = known_stations[:min(self.n_ground, len(known_stations))]
        
        # Thêm trạm ngẫu nhiên nếu cần
        for _ in range(self.n_ground - len(ground_stations)):
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)
            gs = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=0*u.m)
            ground_stations.append(gs)
        
        return ground_stations
    
    def compute_target_visibility_per_slot(self, min_elevation=10.0, max_off_nadir=30.0):
        """
        Tính toán ma trận tầm nhìn mục tiêu cho từng slot quỹ đạo
        
        Args:
            min_elevation: Góc ngẩng tối thiểu (độ)
            max_off_nadir: Góc off-nadir tối đa cho quan sát (độ)
        
        Returns:
            Dict V_target[slot_j][k, t, target_idx] -> bool
        """
        V_target_per_slot = {}
        
        print(f"Computing target visibility: {self.K} satellites, {self.T} time steps, {self.n_targets} targets, {self.n_slots} slots...")
        
        for slot_j in range(1, self.n_slots + 1):
            V_target = np.zeros((self.K, self.T, self.n_targets), dtype=bool)
            satellites = self.satellites_per_slot[slot_j]
            
            for t_idx, epoch in enumerate(self.time_grid):
                for k, satellite in enumerate(satellites):
                    # Truyền bá vệ tinh đến epoch hiện tại
                    sat_propagated = satellite.propagate(epoch - self.start_time)
                    r_sat = sat_propagated.r.to(u.km).value
                    
                    for tgt_idx, (lat, lon) in enumerate(self.targets):
                        # Vị trí mục tiêu trên bề mặt Trái Đất
                        target_loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=0*u.m)
                        target_itrs = target_loc.get_itrs(obstime=epoch)
                        target_gcrs = target_itrs.transform_to(GCRS(obstime=epoch))
                        r_tgt = target_gcrs.cartesian.xyz.to(u.km).value
                        
                        # Vector line-of-sight
                        los = r_tgt - r_sat
                        los_norm = np.linalg.norm(los)
                        
                        # Vector nadir (từ vệ tinh đến tâm Trái Đất)
                        nadir = -r_sat / np.linalg.norm(r_sat)
                        
                        # Góc off-nadir
                        cos_angle = np.dot(los / los_norm, nadir)
                        off_nadir_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                        
                        # Góc ngẩng từ mục tiêu
                        up_vector = r_tgt / np.linalg.norm(r_tgt)
                        cos_elev = np.dot(-los / los_norm, up_vector)
                        elevation_angle = np.degrees(np.arcsin(np.clip(cos_elev, -1, 1)))
                        
                        # Kiểm tra điều kiện tầm nhìn
                        if elevation_angle >= min_elevation and off_nadir_angle <= max_off_nadir:
                            V_target[k, t_idx, tgt_idx] = True
            
            V_target_per_slot[slot_j] = V_target
            
            if slot_j % 5 == 0:
                print(f"  Slot {slot_j}/{self.n_slots} completed...")
        
        return V_target_per_slot
    
    def compute_ground_visibility_per_slot(self, min_elevation=10.0):
        """
        Tính toán ma trận tầm nhìn trạm mặt đất cho từng slot
        
        Returns:
            Dict V_ground[slot_j][k, t, gs_idx] -> bool
        """
        V_ground_per_slot = {}
        
        print(f"Computing ground station visibility: {self.K} satellites, {self.T} time steps, {self.n_ground} stations, {self.n_slots} slots...")
        
        for slot_j in range(1, self.n_slots + 1):
            V_ground = np.zeros((self.K, self.T, self.n_ground), dtype=bool)
            satellites = self.satellites_per_slot[slot_j]
            
            for t_idx, epoch in enumerate(self.time_grid):
                for k, satellite in enumerate(satellites):
                    sat_propagated = satellite.propagate(epoch - self.start_time)
                    r_sat = sat_propagated.r.to(u.km).value
                    
                    sat_gcrs_coord = GCRS(
                        CartesianRepresentation(r_sat[0]*u.km, r_sat[1]*u.km, r_sat[2]*u.km),
                        obstime=epoch
                    )
                    
                    for gs_idx, gs in enumerate(self.ground_stations):
                        # Chuyển đổi sang hệ tọa độ AltAz để tính góc ngẩng
                        gs_altaz_frame = AltAz(obstime=epoch, location=gs)
                        sat_altaz = sat_gcrs_coord.transform_to(gs_altaz_frame)
                        
                        elevation = sat_altaz.alt.deg
                        
                        if elevation >= min_elevation:
                            V_ground[k, t_idx, gs_idx] = True
            
            V_ground_per_slot[slot_j] = V_ground
        
        return V_ground_per_slot
    
    def compute_sun_visibility(self):
        """
        Tính toán ma trận tầm nhìn mặt trời (eclipse)
        Mô hình đơn giản: ~64% quỹ đạo trong ánh sáng mặt trời (điển hình cho LEO)
        
        Returns:
            Binary matrix V_sun[K, T]
        """
        V_sun = np.zeros((self.K, self.T), dtype=bool)
        
        # Mô hình eclipse đơn giản
        # Quỹ đạo 600km: chu kỳ ~ 96 phút
        # Eclipse ~ 35 phút (36% quỹ đạo)
        # Ánh sáng ~ 61 phút (64% quỹ đạo)
        
        orbit_period_steps = int(96 * 60 / self.dt)
        sunlight_fraction = 0.64
        
        np.random.seed(self.seed + 300)
        
        for k in range(self.K):
            phase = np.random.randint(0, max(1, orbit_period_steps))
            
            for t in range(self.T):
                orbit_position = (t + phase) % max(1, orbit_period_steps)
                V_sun[k, t] = orbit_position < int(sunlight_fraction * orbit_period_steps)
        
        return V_sun


def generate_physics_based_visibility(S, K, P, G, J_options, T_per_stage, seed=42, dt_seconds=100):
    """
    Tạo ma trận tầm nhìn dựa trên vật lý cho bài toán REOSSP/EOSSP
    
    Args:
        S: Số stage
        K: Số vệ tinh
        P: Số mục tiêu quan sát
        G: Số trạm mặt đất
        J_options: Số slot quỹ đạo
        T_per_stage: Số bước thời gian mỗi stage
        seed: Random seed
        dt_seconds: Thời lượng mỗi bước thời gian
    
    Returns:
        V_reossp, W_reossp, H_reossp, V_eossp, W_eossp, H_eossp dictionaries
    """
    T_total = S * T_per_stage
    start_time = Time("2024-01-01 00:00:00", scale="utc")
    
    # Tạo generator
    generator = PhysicsBasedVisibilityGenerator(
        n_satellites=K,
        n_targets=P,
        n_ground_stations=G,
        start_time=start_time,
        time_steps=T_total,
        dt_seconds=dt_seconds,
        n_slots=J_options,
        seed=seed
    )
    
    # Tính toán các ma trận tầm nhìn
    V_target_per_slot = generator.compute_target_visibility_per_slot()
    V_ground_per_slot = generator.compute_ground_visibility_per_slot()
    V_sun = generator.compute_sun_visibility()
    
    # Chuyển đổi sang định dạng REOSSP/EOSSP
    V_reossp, W_reossp, H_reossp = {}, {}, {}
    V_eossp, W_eossp, H_eossp = {}, {}, {}
    K_list = list(range(1, K + 1))
    P_list = list(range(1, P + 1))
    G_list = list(range(1, G + 1))
    T_s_list = list(range(1, T_per_stage + 1))
    
    for s in range(1, S + 1):
        for t_l in T_s_list:
            t_g = (s - 1) * T_per_stage + t_l  # Global time index
            t_idx = t_g - 1  # 0-based index for numpy arrays
            
            for k in K_list:
                k_idx = k - 1  # 0-based index
                
                # Sun visibility
                is_sun = 1 if V_sun[k_idx, t_idx] else 0
                H_eossp[(k, t_g)] = is_sun
                
                for j in range(1, J_options + 1):
                    V_target_slot = V_target_per_slot[j]
                    V_ground_slot = V_ground_per_slot[j]
                    
                    # Target visibility
                    for p in P_list:
                        p_idx = p - 1
                        if p_idx < generator.n_targets and V_target_slot[k_idx, t_idx, p_idx]:
                            V_reossp[(s, k, t_l, j, p)] = 1
                    
                    # Ground station visibility  
                    for g in G_list:
                        g_idx = g - 1
                        if g_idx < generator.n_ground and V_ground_slot[k_idx, t_idx, g_idx]:
                            W_reossp[(s, k, t_l, j, g)] = 1
                    
                    # Sun visibility cho slot
                    H_reossp[(s, k, t_l, j)] = is_sun
                
                # EOSSP: Mặc định ở Slot trung tâm (Slot 10)
                fixed_j = min(10, J_options)
                for p in P_list:
                    if V_reossp.get((s, k, t_l, fixed_j, p), 0) == 1:
                        V_eossp[(k, t_g, p)] = 1
                for g in G_list:
                    if W_reossp.get((s, k, t_l, fixed_j, g), 0) == 1:
                        W_eossp[(k, t_g, g)] = 1
    
    print(f"\nPhysics-based visibility generation completed:")
    print(f"  V_reossp entries: {len(V_reossp)}")
    print(f"  W_reossp entries: {len(W_reossp)}")
    print(f"  V_eossp entries: {len(V_eossp)}")
    print(f"  W_eossp entries: {len(W_eossp)}")
    
    return V_reossp, W_reossp, H_reossp, V_eossp, W_eossp, H_eossp


def generate_probabilistic_visibility(S, K, P, G, J_options, T_per_stage, seed=42):
    """
    Tạo ma trận tầm nhìn dựa trên xác suất (fallback khi không có thư viện vật lý)
    """
    random.seed(seed)
    
    V_reossp, W_reossp, H_reossp = {}, {}, {}
    V_eossp, W_eossp, H_eossp = {}, {}, {}
    
    K_list = list(range(1, K + 1))
    P_list = list(range(1, P + 1))
    G_list = list(range(1, G + 1))
    T_s_list = list(range(1, T_per_stage + 1))
    
    for s in range(1, S + 1):
        for t_l in T_s_list:
            t_g = (s - 1) * T_per_stage + t_l
            for k in K_list:
                # Xác suất sạc pin (Sun visibility)
                is_sun = 1 if random.random() < 0.35 else 0
                H_eossp[(k, t_g)] = is_sun
                
                for j in range(1, J_options + 1):
                    prob_v = 0.05 + (j / J_options) * 0.15
                    if random.random() < prob_v:
                        p = random.choice(P_list)
                        V_reossp[(s, k, t_l, j, p)] = 1
                    
                    if random.random() < 0.08:
                        g = random.choice(G_list)
                        W_reossp[(s, k, t_l, j, g)] = 1
                    
                    H_reossp[(s, k, t_l, j)] = is_sun
                
                # EOSSP: Mặc định ở Slot trung tâm
                fixed_j = min(10, J_options)
                for p in P_list:
                    if V_reossp.get((s, k, t_l, fixed_j, p), 0) == 1:
                        V_eossp[(k, t_g, p)] = 1
                for g in G_list:
                    if W_reossp.get((s, k, t_l, fixed_j, g), 0) == 1:
                        W_eossp[(k, t_g, g)] = 1
    
    return V_reossp, W_reossp, H_reossp, V_eossp, W_eossp, H_eossp


# =====================================================================
# 1. HÀM TẠO DỮ LIỆU
# =====================================================================

def generate_challenging_instance(S=5, K=2, P=15, G=3, J_options=20, T_per_stage=40, seed=42, 
                                   use_physics_visibility=True, dt_seconds=100):
    """
    Tạo instance với tùy chọn sử dụng ma trận tầm nhìn dựa trên vật lý
    
    Args:
        use_physics_visibility: True để sử dụng thư viện vật lý (poliastro/astropy),
                                False để sử dụng xác suất ngẫu nhiên
        dt_seconds: Thời lượng mỗi bước thời gian (giây) - chỉ dùng cho physics mode
    """
    random.seed(seed)
    data = {}
    data["S"], data["K"] = S, list(range(1, K+1))
    data["P"], data["G"] = list(range(1, P+1)), list(range(1, G+1))
    T_total = S * T_per_stage
    data["T_full"], data["T_per_stage"], data["T_s_list"] = list(range(1, T_total + 1)), T_per_stage, list(range(1, T_per_stage + 1))
    
    data["J"] = {(s, k): list(range(1, J_options + 1)) for s in range(1, S + 1) for k in data["K"]}
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


    data["Dobs"] = 102.50  # MB/s (Generation rate)
    data["Dcomm"] = 100.0  # MB/s (Downlink rate)
    
    data["Bobs"] = 16.26   # J/s (Consump. during Obs)
    data["Bcomm"] = 1.20   # J/s (Consump. during Comm)
    data["Bcharge"] = 41.48 # J/s (Generation from Sun)
    data["Btime"] = 2.0    # J/s (Idle consumption)
    data["Brecon"] = 0.50  # J (Energy for reconfiguration - maneuver)
    
    data["Dmin"], data["Dmax"] = 0.0, 128000.0 # 128 GB
    data["Bmin"], data["Bmax"] = 200.0, 1000.0  # Giữ pin trên 20%
    
    # Chi phí nhiên liệu (Propellant kms)
    c = {}
    for s in range(1, S+1):
        for k in data["K"]:
            prev_slots = data["J_0"][k] if s==1 else data["J"][(s-1,k)]
            for i in prev_slots:
                for j in data["J"][(s,k)]:
                    # Giả định chi phí nhảy slot trong bài báo (từ 0.01 đến 0.3)
                    c[(s,k,i,j)] = abs(i - j) * 0.02 
    
    data["c"] = c
    data["c_k_max"] = {k: 1.8 for k in data["K"]}
    data["d_init"], data["b_init"], data["C"] = {k: 0 for k in data["K"]}, {k: 1000 for k in data["K"]}, 2.0
    
    data["V_R"], data["W_R"], data["H_R"] = V_reossp, W_reossp, H_reossp
    data["V_E"], data["W_E"], data["H_E"] = V_eossp, W_eossp, H_eossp
    
    # Lưu thông tin về phương pháp tạo visibility
    data["visibility_method"] = "physics" if (use_physics_visibility and PHYSICS_LIBS_AVAILABLE) else "probabilistic"
    
    return data

# HÀM GIẢI EOSSP-Exact (Baseline)

def solve_eossp_exact(data, time_limit=3600, verbose=False):
    model = Model("EOSSP_Exact")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = time_limit

    K_list, P_list, G_list, T_full, C = data["K"], data["P"], data["G"], data["T_full"], data["C"]
    Dobs, Dcomm = data["Dobs"], data["Dcomm"]
    Bobs, Bcomm, Bcharge, Btime = data["Bobs"], data["Bcomm"], data["Bcharge"], data["Btime"]
    Dmin, Dmax, Bmin, Bmax = data["Dmin"], data["Dmax"], data["Bmin"], data["Bmax"]

    y, q, h, d, b = {}, {}, {}, {}, {}
    
    for k in K_list:
        for t in T_full:
            d[(k, t)] = model.addVar(lb=Dmin, ub=Dmax, vtype=GRB.CONTINUOUS, name=f"d_{k}_{t}")
            b[(k, t)] = model.addVar(lb=Bmin, ub=Bmax, vtype=GRB.CONTINUOUS, name=f"b_{k}_{t}")
            h[(k, t)] = model.addVar(vtype=GRB.BINARY, name=f"h_{k}_{t}")
            for p in P_list:
                y[(k, t, p)] = model.addVar(vtype=GRB.BINARY, name=f"y_{k}_{t}_{p}")
            for gidx in G_list:
                q[(k, t, gidx)] = model.addVar(vtype=GRB.BINARY, name=f"q_{k}_{t}_{gidx}")
    
    model.update()

    # Hàm Mục tiêu (Objective 2)
    model.setObjective(quicksum(quicksum(C * q.get((k, t, gidx), 0) for gidx in G_list) + \
                                quicksum(y.get((k, t, p), 0) for p in P_list) for k in K_list for t in T_full), GRB.MAXIMIZE)

    # Ràng buộc
    for k in K_list:
        model.addConstr(d[(k, 1)] == Dmin, name=f"data_init_{k}")
        model.addConstr(b[(k, 1)] == Bmax, name=f"batt_init_{k}")
            
        for t in T_full:
            # Time Window & At Most One (4a-4d)
            for p in P_list: model.addConstr(data["V_E"].get((k, t, p), 0) >= y.get((k, t, p), 0))
            for gidx in G_list: model.addConstr(data["W_E"].get((k, t, gidx), 0) >= q.get((k, t, gidx), 0))
            model.addConstr(data["H_E"].get((k, t), 0) >= h.get((k, t), 0))
            model.addConstr(quicksum(y.get((k, t, p), 0) for p in P_list) + quicksum(q.get((k, t, gidx), 0) for gidx in G_list) + h.get((k, t), 0) <= 1)
            
            # Data Constraints (5a-5c)
            data_gain = quicksum(Dobs * y.get((k, t, p), 0) for p in P_list)
            data_loss = quicksum(Dcomm * q.get((k, t, gidx), 0) for gidx in G_list)
            if t < max(T_full): model.addConstr(d[(k, t+1)] == d[(k, t)] + data_gain - data_loss)
            model.addConstr(d[(k, t)] + data_gain <= Dmax)
            model.addConstr(d[(k, t)] - data_loss >= Dmin)

            # Battery Constraints (6a-6c)
            batt_gain = Bcharge * h.get((k, t), 0)
            batt_net_loss = quicksum(Bobs * y.get((k, t, p), 0) for p in P_list) + quicksum(Bcomm * q.get((k, t, gidx), 0) for gidx in G_list) + Btime
            if t < max(T_full): model.addConstr(b[(k, t+1)] == b[(k, t)] + batt_gain - batt_net_loss)
            model.addConstr(b[(k, t)] + batt_gain <= Bmax)
            model.addConstr(b[(k, t)] - batt_net_loss >= Bmin)

    # Tối ưu hóa
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    runtime = end_time - start_time
    
    results = {"status": model.Status, "runtime": runtime, "Z_E": 0, "Z_E_GB": 0}
    
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        results["Z_E"] = model.ObjVal
        downlink_MB = quicksum(Dcomm * q[(k, t, gidx)].X 
                               for k in K_list for t in T_full 
                               for gidx in G_list if (k, t, gidx) in q).getValue()
        results["Z_E_GB"] = downlink_MB / 1024.0
        
    return results

# HÀM GIẢI REOSSP-Exact

def solve_reossp_exact(data, time_limit=3600, verbose=False):
    model = Model("REOSSP_Exact")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.TimeLimit = time_limit

    S_list, K_list, P_list, G_list, T_s, C = list(range(1, data["S"] + 1)), data["K"], data["P"], data["G"], data["T_s_list"], data["C"]
    Dobs, Dcomm, Bobs, Bcomm, Bcharge, Btime, Brecon = data["Dobs"], data["Dcomm"], data["Bobs"], data["Bcomm"], data["Bcharge"], data["Btime"], data["Brecon"]
    Dmin, Dmax, Bmin, Bmax = data["Dmin"], data["Dmax"], data["Bmin"], data["Bmax"]

    x, y, q, h, d, b = {}, {}, {}, {}, {}, {}
    
    # Khai báo biến (7a-7f)
    for s in S_list:      
        for k in K_list:
            prev_list = data["J_0"][k] if s==1 else data["J"][(s-1,k)] 
            for i in prev_list:
                for j in data["J"][(s,k)]:
                    x[(s, k, i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{s}_{k}_{i}_{j}")
            for t in T_s:
                d[(s, k, t)] = model.addVar(lb=Dmin, ub=Dmax, vtype=GRB.CONTINUOUS, name=f"d_{s}_{k}_{t}")
                b[(s, k, t)] = model.addVar(lb=Bmin, ub=Bmax, vtype=GRB.CONTINUOUS, name=f"b_{s}_{k}_{t}")
                h[(s, k, t)] = model.addVar(vtype=GRB.BINARY, name=f"h_{s}_{k}_{t}")
                for p in P_list: y[(s, k, t, p)] = model.addVar(vtype=GRB.BINARY, name=f"y_{s}_{k}_{t}_{p}")
                for gidx in G_list: q[(s, k, t, gidx)] = model.addVar(vtype=GRB.BINARY, name=f"q_{s}_{k}_{t}_{gidx}")
    model.update()

    # Hàm Mục tiêu (Objective 8)
    model.setObjective(quicksum(quicksum(C * q.get((s, k, t, gidx), 0) for gidx in G_list) + \
                                quicksum(y.get((s, k, t, p), 0) for p in P_list) 
                                for s in S_list for k in K_list for t in T_s), GRB.MAXIMIZE)

    # Ràng buộc (10-14)
    for k in K_list:
        # 10a: First stage maneuver
        i_init = data["J_0"][k][0]
        model.addConstr(quicksum(x.get((1, k, i_init, j), 0) for j in data["J"][(1, k)]) == 1)
        
        # 10c: Total Propellant Budget
        total_cost = quicksum(data["c"].get((s, k, i, j), 0) * x.get((s, k, i, j), 0)
                              for s in S_list for i in (data["J_0"][k] if s==1 else data["J"][(s-1,k)]) 
                              for j in data["J"][(s,k)])
        model.addConstr(total_cost <= data["c_k_max"][k], name=f"transfer_budget_{k}")

        # Initial Conditions (d_1^1k and b_1^1k)
        model.addConstr(d[(1, k, 1)] == Dmin)
        recon_cost_init = quicksum(Brecon * x.get((1, k, i_init, j), 0) for j in data["J"][(1,k)])
        model.addConstr(b[(1, k, 1)] == Bmax - recon_cost_init)
        model.addConstr(Bmax - recon_cost_init >= Bmin) # (14d)

        for s in S_list:
            prev_list = data["J_0"][k] if s==1 else data["J"][(s-1,k)]
            next_list = data["J"][(s+1,k)] if s < max(S_list) else []
            
            # 10b: Continuity chain
            if s < max(S_list):
                for i in data["J"][(s, k)]:
                    lhs = quicksum(x.get((s+1, k, i, j), 0) for j in data["J"][(s+1, k)])
                    rhs = quicksum(x.get((s, k, i_prev, i), 0) for i_prev in prev_list)
                    model.addConstr(lhs - rhs == 0)

            for t in T_s:
                # 11d: At most one task
                model.addConstr(quicksum(y.get((s, k, t, p), 0) for p in P_list) + quicksum(q.get((s, k, t, gidx), 0) for gidx in G_list) + h.get((s, k, t), 0) <= 1)
                
                # 11a-11c: Visibility (task <= sum V * x)
                for p in P_list: 
                    sum_vis = quicksum(data["V_R"].get((s, k, t, j, p), 0) * x.get((s, k, i, j), 0) 
                                       for i in prev_list for j in data["J"][(s,k)])
                    model.addConstr(sum_vis >= y.get((s, k, t, p), 0))
                for gidx in G_list: 
                    sum_vis = quicksum(data["W_R"].get((s, k, t, j, gidx), 0) * x.get((s, k, i, j), 0) 
                                       for i in prev_list for j in data["J"][(s,k)])
                    model.addConstr(sum_vis >= q.get((s, k, t, gidx), 0))
                sum_h = quicksum(data["H_R"].get((s, k, t, j), 0) * x.get((s, k, i, j), 0) 
                                 for i in prev_list for j in data["J"][(s,k)])
                model.addConstr(sum_h >= h.get((s, k, t), 0))

                # Data (12c-12d)
                data_gain = quicksum(Dobs * y.get((s, k, t, p), 0) for p in P_list)
                data_loss = quicksum(Dcomm * q.get((s, k, t, gidx), 0) for gidx in G_list)
                model.addConstr(d[(s, k, t)] + data_gain <= Dmax) # 12c
                model.addConstr(d[(s, k, t)] - data_loss >= Dmin) # 12d

                # Battery & Tracking (13a-13b & 14a-14c)
                batt_gain = Bcharge * h.get((s, k, t), 0)
                batt_net_loss = quicksum(Bobs * y.get((s, k, t, p), 0) for p in P_list) + quicksum(Bcomm * q.get((s, k, t, gidx), 0) for gidx in G_list) + Btime
                model.addConstr(b[(s, k, t)] + batt_gain <= Bmax) # 14a

                if t < max(T_s):
                    # Within stage (13a, part of 14b)
                    model.addConstr(b[(s, k, t+1)] == b[(s, k, t)] + batt_gain - batt_net_loss) 
                    model.addConstr(d[(s, k, t+1)] == d[(s, k, t)] + data_gain - data_loss) # 12a
                    model.addConstr(b[(s, k, t)] - batt_net_loss >= Bmin) # 14b
                    
                elif t == max(T_s):
                    recon_cost_next = quicksum(Brecon * x.get((s+1, k, i, j), 0) for i in data["J"][(s,k)] for j in next_list)
                    if s < max(S_list):
                        # Across stages (13b, 12b, 14c)
                        model.addConstr(b[(s+1, k, 1)] == b[(s, k, t)] + batt_gain - batt_net_loss - recon_cost_next)
                        model.addConstr(d[(s+1, k, 1)] == d[(s, k, t)] + data_gain - data_loss)
                        model.addConstr(b[(s, k, t)] - batt_net_loss - recon_cost_next >= Bmin)
                    else:
                        model.addConstr(b[(s, k, t)] - batt_net_loss >= Bmin) # 14b (t=T)


    # Tối ưu hóa
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    runtime = end_time - start_time
    
    results = {"status": model.Status, "runtime": runtime, "Z_R": 0, "Z_R_GB": 0, "Total_Propellant_kms": 0}
    
    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        results["Z_R"] = model.ObjVal
        results["Z_R_GB"] = quicksum(Dcomm * q[(s, k, t, gidx)].X for s in S_list for k in K_list for t in T_s for gidx in G_list).getValue() / 1024.0
        results["Total_Propellant_kms"] = quicksum(data["c"].get((s, k, i, j), 0) * x.get((s, k, i, j), 0).X
                                                  for s in S_list for k in K_list for i in (data["J_0"][k] if s==1 else data["J"][(s-1,k)]) 
                                                  for j in data["J"][(s,k)]).getValue()
        
    return results

# HÀM GIẢI REOSSP-RHP (Procedure)

def solve_subproblem(data, s, L, J_tilde_prev, d_init_in, b_init_in, c_max_in, time_limit=None, verbose=False):
    """ Giải bài toán con RHP(s, L) """
    model = Model(f"RHP_s{s}_L{L}")
    model.Params.OutputFlag = 1 if verbose else 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit

    S_full, K_list, P_list, G_list, C, T_s = data["S"], data["K"], data["P"], data["G"], data["C"], data["T_s_list"]
    Dobs, Dcomm, Bobs, Bcomm, Bcharge, Btime, Brecon = data["Dobs"], data["Dcomm"], data["Bobs"], data["Bcomm"], data["Bcharge"], data["Btime"], data["Brecon"]
    Dmin, Dmax, Bmin, Bmax = data["Dmin"], data["Dmax"], data["Bmin"], data["Bmax"]

    L_end = min(s + L - 1, S_full)
    Lset = list(range(s, L_end + 1))
    
    x, y, q, h, d, b = {}, {}, {}, {}, {}, {}

    # Khai báo biến
    for ell in Lset:      
        for k in K_list:
            prev_list = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1, k)]
            for i in prev_list:
                for j in data["J"][(ell, k)]:
                    x[(ell, k, i, j)] = model.addVar(vtype=GRB.BINARY, name=f"x_{ell}_{k}_{i}_{j}")
            for t in T_s:
                d[(ell, k, t)] = model.addVar(lb=Dmin, ub=Dmax, vtype=GRB.CONTINUOUS, name=f"d_{ell}_{k}_{t}")
                b[(ell, k, t)] = model.addVar(lb=Bmin, ub=Bmax, vtype=GRB.CONTINUOUS, name=f"b_{ell}_{k}_{t}")
                h[(ell, k, t)] = model.addVar(vtype=GRB.BINARY, name=f"h_{ell}_{k}_{t}")
                for p in P_list: y[(ell, k, t, p)] = model.addVar(vtype=GRB.BINARY, name=f"y_{ell}_{k}_{t}_{p}")
                for gidx in G_list: q[(ell, k, t, gidx)] = model.addVar(vtype=GRB.BINARY, name=f"q_{ell}_{k}_{t}_{gidx}")
    model.update()

    # Hàm Mục tiêu (Objective 17)
    model.setObjective(quicksum(quicksum(C * q.get((ell, k, t, gidx), 0) for gidx in G_list) + quicksum(y.get((ell, k, t, p), 0) for p in P_list) for ell in Lset for k in K_list for t in T_s), GRB.MAXIMIZE)

    # Ràng buộc (18-22)
    for k in K_list:
        i_init = J_tilde_prev[k]
        # 18a: First stage maneuver (s)
        model.addConstr(quicksum(x.get((s, k, i_init, j), 0) for j in data["J"][(s, k)]) == 1)
        
        # 18c: Total Propellant Budget
        total_cost = quicksum(data["c"].get((ell, k, i, j), 0) * x.get((ell, k, i, j), 0)
                              for ell in Lset for i in ([J_tilde_prev[k]] if ell == s else data["J"][(ell-1,k)]) 
                              for j in data["J"][(ell,k)])
        model.addConstr(total_cost <= c_max_in[k], name=f"transfer_budget_{k}")

        # Initial Conditions (d_1^sk and b_1^sk)
        model.addConstr(d[(s, k, 1)] == d_init_in[k])
        recon_cost = quicksum(Brecon * x.get((s, k, i_init, j), 0) for j in data["J"][(s, k)])
        model.addConstr(b[(s, k, 1)] == b_init_in[k] - recon_cost)
        model.addConstr(b_init_in[k] - recon_cost >= Bmin)  # Tương tự ràng buộc 14d trong exact
        
        # Ràng buộc khác (18b, 19, 20, 21, 22) - Rất giống REOSSP-Exact, chỉ thay s -> ell, S -> L_end
        for ell in Lset:
            prev_list_x = [J_tilde_prev[k]] if ell == s else data["J"][(ell-1,k)]
            next_list_x = data["J"][(ell+1,k)] if ell < max(Lset) else []

            if ell < max(Lset): # 18b: Continuity chain
                for i in data["J"][(ell, k)]:
                    lhs = quicksum(x.get((ell+1, k, i, j), 0) for j in data["J"][(ell+1, k)])
                    rhs = quicksum(x.get((ell, k, i_prev, i), 0) for i_prev in prev_list_x)
                    model.addConstr(lhs - rhs == 0)

            for t in T_s:
                # 19d: At most one task
                model.addConstr(quicksum(y.get((ell, k, t, p), 0) for p in P_list) + quicksum(q.get((ell, k, t, gidx), 0) for gidx in G_list) + h.get((ell, k, t), 0) <= 1)
                
                # 19a-19c: Visibility (task <= sum V * x)
                for p in P_list: 
                    sum_vis = quicksum(data["V_R"].get((ell, k, t, j, p), 0) * x.get((ell, k, i, j), 0) 
                                       for i in prev_list_x for j in data["J"][(ell,k)])
                    model.addConstr(sum_vis >= y.get((ell, k, t, p), 0))
                for gidx in G_list: 
                    sum_vis = quicksum(data["W_R"].get((ell, k, t, j, gidx), 0) * x.get((ell, k, i, j), 0) 
                                       for i in prev_list_x for j in data["J"][(ell,k)])
                    model.addConstr(sum_vis >= q.get((ell, k, t, gidx), 0))
                sum_h = quicksum(data["H_R"].get((ell, k, t, j), 0) * x.get((ell, k, i, j), 0) 
                                 for i in prev_list_x for j in data["J"][(ell,k)])
                model.addConstr(sum_h >= h.get((ell, k, t), 0))

                # Data (20e-20f)
                data_gain = quicksum(Dobs * y.get((ell, k, t, p), 0) for p in P_list)
                data_loss = quicksum(Dcomm * q.get((ell, k, t, gidx), 0) for gidx in G_list)
                model.addConstr(d[(ell, k, t)] + data_gain <= Dmax)
                model.addConstr(d[(ell, k, t)] - data_loss >= Dmin)

                # Battery (22a)
                batt_gain = Bcharge * h.get((ell, k, t), 0)
                batt_net_loss = quicksum(Bobs * y.get((ell, k, t, p), 0) for p in P_list) + quicksum(Bcomm * q.get((ell, k, t, gidx), 0) for gidx in G_list) + Btime
                model.addConstr(b[(ell, k, t)] + batt_gain <= Bmax)

                # Tracking/Boundary
                if t < max(T_s):
                    # Within stage (20b/20c, 21b/21c, 22b)
                    model.addConstr(d[(ell, k, t+1)] == d[(ell, k, t)] + data_gain - data_loss)
                    model.addConstr(b[(ell, k, t+1)] == b[(ell, k, t)] + batt_gain - batt_net_loss)
                    model.addConstr(b[(ell, k, t)] - batt_net_loss >= Bmin) # 22b
                    
                elif t == max(T_s) and ell < max(Lset):
                    recon_cost_next = quicksum(Brecon * x.get((ell+1, k, i, j), 0) for i in data["J"][(ell,k)] for j in next_list_x)
                    # Across stages (20d, 21d, 22c)
                    model.addConstr(d[(ell+1, k, 1)] == d[(ell, k, t)] + data_gain - data_loss)
                    model.addConstr(b[(ell+1, k, 1)] == b[(ell, k, t)] + batt_gain - batt_net_loss - recon_cost_next)
                    model.addConstr(b[(ell, k, t)] - batt_net_loss - recon_cost_next >= Bmin) # 22c
                elif t == max(T_s) and ell == max(Lset):
                    model.addConstr(b[(ell, k, t)] - batt_net_loss >= Bmin) # 22b (last t)

    # Tối ưu hóa
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    runtime = end_time - start_time
    
    return model, x, y, q, h, d, b, runtime


def solve_reossp_rhp(data, L=1, time_limit_per_sub=3600, verbose=False):
    S_full, K_list, T_s = data["S"], data["K"], data["T_s_list"]
    
    # 1. Khởi tạo trạng thái ban đầu (Giống với REOSSP-Exact)
    J_tilde_prev = {k: data["J_0"][k][0] for k in K_list} 
    c_max_in = data["c_k_max"].copy()
    d_init_in = {k: data["Dmin"] for k in K_list}
    b_init_in = {k: data["Bmax"] for k in K_list}

    # Tổng hợp kết quả cuối cùng
    z_history, runtime_history, total_runtime = [], [], 0.0
    total_downlink_data_MB, total_propellant_used_kms = 0.0, 0.0
    status_history = []

    # Vòng lặp RHP (s=1 đến S-L+1)
    for s in range(1, S_full - L + 2): 
        # Cập nhật b_init (áp dụng maneuver cost của stage s) cho bài toán con đầu tiên (s=1)
        if verbose:
            print(f"\n--- Solving RHP(s={s}, L={L}) ---")

        # Giải bài toán con RHP(s, L)
        model, xvars, yvars, qvars, hvars, dvars, bvars, sub_runtime = solve_subproblem(data, s, L, J_tilde_prev, d_init_in, b_init_in, c_max_in, time_limit=time_limit_per_sub, verbose=verbose)
        
        runtime_history.append(sub_runtime)
        total_runtime += sub_runtime

        status = model.Status
        status_history.append(status)

        if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
            z_history.append(None)
            break 
            
        # Tính z chỉ cho stage s (để tránh overcount khi L > 1)
        z_stage = quicksum(data["C"] * qvars.get((s, k, t, gidx), 0).X for k in K_list for t in T_s for gidx in data["G"]) + \
                  quicksum(yvars.get((s, k, t, p), 0).X for k in K_list for t in T_s for p in data["P"])
        z_history.append(z_stage.getValue())

        # --- Trích xuất quyết định giai đoạn s (ell = s) và Cập nhật trạng thái ---
        Tend = max(T_s)
        
        for k in K_list:
            # 1. Cập nhật Vị trí Quỹ đạo và Nhiên liệu đã dùng
            current_choice = None
            for (ell, kk, i, j), var in list(xvars.items()):
                if ell == s and kk == k:
                    val = int(round(var.X)) if hasattr(var, "X") and var.X is not None else 0
                    if val == 1: current_choice = (i, j); break
            
            if current_choice:
                i, j = current_choice
                J_tilde_prev[k] = j

                cost = data["c"].get((s, k, i, j), 0.0)
                total_propellant_used_kms += cost 
                c_max_in[k] = max(0.0, c_max_in[k] - cost)
        
            # 2. Cập nhật Data Downlink (Z_RHP) và Quyết định hành động
            for t in T_s:
                # Downlink Data for Stage s
                for g in data["G"]:
                    var = qvars.get((s, k, t, g))
                    if var and hasattr(var, "X") and var.X is not None and int(round(var.X)) == 1:
                        total_downlink_data_MB += data["Dcomm"] 

            # 3. Cập nhật Initial Conditions cho giai đoạn s+1
            if s < S_full - L + 1:
                ell_next = s + 1
                if (ell_next, k, 1) in dvars and hasattr(dvars[(ell_next, k, 1)], "X"):  # Cho L > 1
                    d_init_in[k] = dvars[(ell_next, k, 1)].X
                    b_init_in[k] = bvars[(ell_next, k, 1)].X
                else:  # Cho L == 1, tính thủ công từ end of s
                    # Tính gain/loss tại Tend
                    data_gain_Tend = sum(data["Dobs"] * yvars.get((s, k, Tend, p), 0).X for p in data["P"])
                    data_loss_Tend = sum(data["Dcomm"] * qvars.get((s, k, Tend, g), 0).X for g in data["G"])
                    batt_gain_Tend = data["Bcharge"] * hvars.get((s, k, Tend), 0).X
                    batt_net_loss_Tend = sum(data["Bobs"] * yvars.get((s, k, Tend, p), 0).X for p in data["P"]) + \
                                         sum(data["Bcomm"] * qvars.get((s, k, Tend, g), 0).X for g in data["G"]) + data["Btime"]
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

# CHẠY CHƯƠNG TRÌNH VÀ SO SÁNH

if __name__ == "__main__":
    S_val = 8
    T_val = 30
    K_val = 5
    L_val = 1
    limit = 320
    
    # Cấu hình sử dụng physics-based visibility
    USE_PHYSICS = True  # Đặt False để dùng probabilistic visibility
    DT_SECONDS = 1000    # Thời lượng mỗi bước thời gian (giây)

    print("\n" + "="*80)
    print("SATELLITE OBSERVATION SCHEDULING COMPARISON")
    print("="*80)
    print(f"Configuration: S={S_val}, K={K_val}, T_per_stage={T_val}, L={L_val}")
    print(f"Visibility Method: {'Physics-based (poliastro/astropy)' if USE_PHYSICS and PHYSICS_LIBS_AVAILABLE else 'Probabilistic'}")
    print("="*80)

    inst = generate_challenging_instance(
        S=S_val, K=K_val, T_per_stage=T_val, J_options=20,
        use_physics_visibility=USE_PHYSICS, dt_seconds=DT_SECONDS
    )
    
    print(f"\nVisibility method used: {inst['visibility_method']}")

    print("\n--- 1. EOSSP (Fixed at Slot 1) ---")
    res_E = solve_eossp_exact(inst, time_limit=limit)
    
    print("\n--- 2. REOSSP-Exact (Global Optimization) ---")
    res_R = solve_reossp_exact(inst, time_limit=limit)

    print(f"\n--- 3. REOSSP-RHP (Lookahead L={L_val}) ---")
    res_RHP = solve_reossp_rhp(inst, L=L_val, time_limit_per_sub=limit)

    # In bảng kết quả
    print("\n" + "="*80)
    print(f"{'Method':<25} | {'Z (Obj)':<10} | {'Runtime':<10} | {'Fuel Used':<10} | {'Improvement':<10}")
    print("-" * 80)
    
    z_e = res_E['Z_E']
    methods = [
        ("EOSSP-Exact (Baseline)", z_e, res_E['runtime'], 0.0),
        ("REOSSP-Exact", res_R['Z_R'], res_R['runtime'], res_R['Total_Propellant_kms']),
        (f"REOSSP-RHP (L={L_val})", sum(res_RHP['z_history']), res_RHP['total_runtime'], res_RHP['Total_Propellant_kms'])
    ]

    for name, z, rt, fuel in methods:
        imp = ((z - z_e) / z_e * 100) if z_e > 0 else 0
        print(f"{name:<25} | {z:<10.2f} | {rt:<10.2f} | {fuel:<10.2f} | {imp:>9.2f}%")
    print("="*80)
    
    # In thống kê visibility
    print("\n" + "="*80)
    print("VISIBILITY STATISTICS")
    print("="*80)
    print(f"V_reossp (Target Visibility): {len(inst['V_R'])} opportunities")
    print(f"W_reossp (Ground Downlink): {len(inst['W_R'])} opportunities")
    print(f"V_eossp (EOSSP Target): {len(inst['V_E'])} opportunities")
    print(f"W_eossp (EOSSP Downlink): {len(inst['W_E'])} opportunities")
    print("="*80)