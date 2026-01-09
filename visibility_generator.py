"""
Module tạo ma trận tầm nhìn (Visibility Matrix) cho bài toán EOSSP và REOSSP.
Hỗ trợ hai phương pháp:
1. Physics-based: Sử dụng poliastro + astropy cho tính toán quỹ đạo thực
2. Probabilistic: Sử dụng xác suất ngẫu nhiên (fallback)
"""

import random
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
    if not PHYSICS_LIBS_AVAILABLE:
        raise RuntimeError("Physics libraries (astropy/poliastro) are not available")
    
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
    
    Args:
        S: Số stage
        K: Số vệ tinh
        P: Số mục tiêu quan sát
        G: Số trạm mặt đất
        J_options: Số slot quỹ đạo
        T_per_stage: Số bước thời gian mỗi stage
        seed: Random seed
    
    Returns:
        V_reossp, W_reossp, H_reossp, V_eossp, W_eossp, H_eossp dictionaries
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
    
    print(f"\nProbabilistic visibility generation completed:")
    print(f"  V_reossp entries: {len(V_reossp)}")
    print(f"  W_reossp entries: {len(W_reossp)}")
    print(f"  V_eossp entries: {len(V_eossp)}")
    print(f"  W_eossp entries: {len(W_eossp)}")
    
    return V_reossp, W_reossp, H_reossp, V_eossp, W_eossp, H_eossp


if __name__ == "__main__":
    # Test module
    print("Testing visibility generation...")
    V_r, W_r, H_r, V_e, W_e, H_e = generate_probabilistic_visibility(
        S=2, K=2, P=10, G=3, J_options=10, T_per_stage=10
    )
    print(f"\nTest completed successfully!")
