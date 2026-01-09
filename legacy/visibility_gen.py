"""
Visibility Matrix Generator for Satellite Scheduling
Generates realistic visibility matrices using orbital mechanics
Replaces MATLAB/YALMIP aerospace toolbox functionality
"""

import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, GCRS, ITRS, AltAz, CartesianRepresentation
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from typing import List, Tuple
import warnings

# Suppress astropy warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


class VisibilityMatrixGenerator:
    """
    Generate visibility matrices for satellite observation and communication scheduling
    
    Uses poliastro for orbit propagation and astropy for coordinate transformations,
    replicating MATLAB Aerospace Toolbox + YALMIP functionality in Python.
    """
    
    def __init__(self, 
                 n_satellites: int,
                 n_targets: int,
                 n_ground_stations: int,
                 start_time: Time,
                 time_steps: int,
                 dt_seconds: int,
                 seed: int = 42):
        """
        Initialize visibility matrix generator
        
        Args:
            n_satellites: Number of satellites in constellation
            n_targets: Number of observation targets
            n_ground_stations: Number of ground stations
            start_time: Mission start time (astropy Time object)
            time_steps: Number of time steps to simulate
            dt_seconds: Time step duration in seconds
            seed: Random seed for reproducibility
        """
        self.K = n_satellites
        self.n_targets = n_targets
        self.n_ground = n_ground_stations
        self.start_time = start_time
        self.T = time_steps
        self.dt = dt_seconds
        self.seed = seed
        
        # Create time grid
        self.time_grid = start_time + np.arange(time_steps) * dt_seconds * u.s
        
        # Initialize satellites, targets, and ground stations
        self.satellites = self._create_satellite_constellation()
        self.targets = self._create_targets()
        self.ground_stations = self._create_ground_stations()
        
    def _create_satellite_constellation(self) -> List[Orbit]:
        """
        Create a constellation of satellites in sun-synchronous orbits
        
        Returns:
            List of poliastro Orbit objects
        """
        np.random.seed(self.seed)
        satellites = []
        
        # Sun-synchronous orbit parameters (typical for Earth observation)
        altitude = 600 * u.km  # 600 km altitude
        a = Earth.R + altitude  # Semi-major axis
        ecc = 0.001 * u.one  # Near-circular orbit
        inc = 97.8 * u.deg  # Sun-synchronous inclination at 600 km
        
        # Distribute satellites in orbital plane with different RAANs and true anomalies
        for k in range(self.K):
            # Spread satellites across different orbital planes
            raan = (k * 360.0 / self.K) * u.deg
            # Offset phase in orbit
            argp = 0 * u.deg
            nu = (k * 60) * u.deg  # Phase offset
            
            orbit = Orbit.from_classical(
                Earth,
                a=a,
                ecc=ecc,
                inc=inc,
                raan=raan,
                argp=argp,
                nu=nu,
                epoch=self.start_time
            )
            satellites.append(orbit)
        
        return satellites
    
    def _create_targets(self) -> List[Tuple[float, float]]:
        """
        Create observation targets (latitude, longitude pairs)
        
        Returns:
            List of (lat, lon) tuples in degrees
        """
        np.random.seed(self.seed + 100)
        targets = []
        
        # Distribute targets globally with bias toward mid-latitudes
        # (where most Earth observation targets are)
        for _ in range(self.n_targets):
            # Use beta distribution for latitude to concentrate in mid-latitudes
            lat = (np.random.beta(2, 2) - 0.5) * 120  # Range: -60 to +60 degrees
            lon = np.random.uniform(-180, 180)
            targets.append((lat, lon))
        
        return targets
    
    def _create_ground_stations(self) -> List[EarthLocation]:
        """
        Create ground station locations
        
        Returns:
            List of astropy EarthLocation objects
        """
        np.random.seed(self.seed + 200)
        
        # Use some realistic ground station locations plus random ones
        known_stations = [
            EarthLocation(lat=37.4*u.deg, lon=-122.1*u.deg, height=0*u.m),  # California
            EarthLocation(lat=52.2*u.deg, lon=0.1*u.deg, height=0*u.m),      # UK
            EarthLocation(lat=-35.3*u.deg, lon=149.1*u.deg, height=0*u.m),   # Australia
        ]
        
        ground_stations = known_stations[:min(self.n_ground, 3)]
        
        # Add random stations if needed
        for _ in range(self.n_ground - len(ground_stations)):
            lat = np.random.uniform(-60, 60)  # Avoid extreme latitudes
            lon = np.random.uniform(-180, 180)
            gs = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=0*u.m)
            ground_stations.append(gs)
        
        return ground_stations
    
    def compute_target_visibility(self, min_elevation: float = 10.0, max_angle: float = 30.0) -> np.ndarray:
        """
        Compute visibility matrix for observation targets
        
        Args:
            min_elevation: Minimum elevation angle in degrees
            max_angle: Maximum off-nadir angle for observation in degrees
        
        Returns:
            Binary matrix V_target[K, T, n_targets]
        """
        V_target = np.zeros((self.K, self.T, self.n_targets), dtype=bool)
        
        print(f"Computing target visibility for {self.K} satellites, {self.T} time steps, {self.n_targets} targets...")
        
        for t_idx, epoch in enumerate(self.time_grid):
            if t_idx % 500 == 0:
                print(f"  Progress: {t_idx}/{self.T} time steps")
            
            for k, satellite in enumerate(self.satellites):
                # Propagate satellite to current epoch
                sat_propagated = satellite.propagate(epoch - self.start_time)
                
                # Get satellite position in GCRS frame
                r_sat = sat_propagated.r.to(u.km).value
                
                for tgt_idx, (lat, lon) in enumerate(self.targets):
                    # Target position on Earth surface
                    target_loc = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=0*u.m)
                    target_itrs = target_loc.get_itrs(obstime=epoch)
                    target_gcrs = target_itrs.transform_to(GCRS(obstime=epoch))
                    r_tgt = target_gcrs.cartesian.xyz.to(u.km).value
                    
                    # Line of sight vector
                    los = r_tgt - r_sat
                    los_norm = np.linalg.norm(los)
                    
                    # Nadir vector (from satellite to Earth center)
                    nadir = -r_sat / np.linalg.norm(r_sat)
                    
                    # Off-nadir angle
                    cos_angle = np.dot(los / los_norm, nadir)
                    off_nadir_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    
                    # Elevation angle from target
                    up_vector = r_tgt / np.linalg.norm(r_tgt)
                    cos_elev = np.dot(-los / los_norm, up_vector)
                    elevation_angle = np.degrees(np.arcsin(np.clip(cos_elev, -1, 1)))
                    
                    # Check visibility conditions
                    if elevation_angle >= min_elevation and off_nadir_angle <= max_angle:
                        V_target[k, t_idx, tgt_idx] = True
        
        # Apply constraint: only one target visible per satellite per time step
        V_target_constrained = np.zeros_like(V_target)
        for k in range(self.K):
            for t in range(self.T):
                visible_targets = np.where(V_target[k, t, :])[0]
                if len(visible_targets) > 0:
                    # Select the target with highest elevation (best observation geometry)
                    selected = np.random.choice(visible_targets)
                    V_target_constrained[k, t, selected] = True
        
        print(f"  Total target visibility opportunities: {V_target_constrained.sum()}")
        return V_target_constrained
    
    def compute_ground_visibility(self, min_elevation: float = 10.0) -> np.ndarray:
        """
        Compute visibility matrix for ground stations (downlinks)
        
        Args:
            min_elevation: Minimum elevation angle in degrees
        
        Returns:
            Binary matrix V_ground[K, T, n_ground]
        """
        V_ground = np.zeros((self.K, self.T, self.n_ground), dtype=bool)
        
        print(f"Computing ground station visibility for {self.K} satellites, {self.T} time steps, {self.n_ground} stations...")
        
        for t_idx, epoch in enumerate(self.time_grid):
            if t_idx % 500 == 0:
                print(f"  Progress: {t_idx}/{self.T} time steps")
            
            for k, satellite in enumerate(self.satellites):
                # Propagate satellite to current epoch
                sat_propagated = satellite.propagate(epoch - self.start_time)
                
                # Get satellite position in GCRS frame
                r_sat = sat_propagated.r.to(u.km).value
                sat_gcrs_coord = GCRS(
                    CartesianRepresentation(r_sat[0]*u.km, r_sat[1]*u.km, r_sat[2]*u.km),
                    obstime=epoch
                )
                
                for gs_idx, gs in enumerate(self.ground_stations):
                    # Transform to AltAz frame for elevation calculation
                    gs_altaz_frame = AltAz(obstime=epoch, location=gs)
                    sat_altaz = sat_gcrs_coord.transform_to(gs_altaz_frame)
                    
                    elevation = sat_altaz.alt.deg
                    
                    # Check visibility
                    if elevation >= min_elevation:
                        V_ground[k, t_idx, gs_idx] = True
        
        print(f"  Total ground station visibility opportunities: {V_ground.sum()}")
        return V_ground
    
    def compute_sun_visibility(self) -> np.ndarray:
        """
        Compute sun visibility (eclipse) matrix
        
        Simplified model: assumes ~60% of orbit in sunlight (typical for LEO)
        
        Returns:
            Binary matrix V_sun[K, T]
        """
        V_sun = np.zeros((self.K, self.T), dtype=bool)
        
        print(f"Computing sun visibility for {self.K} satellites, {self.T} time steps...")
        
        # Simplified eclipse model
        # For a 600km orbit: period ~ 96 minutes
        # Eclipse duration ~ 35 minutes (36% of orbit)
        # Sunlight duration ~ 61 minutes (64% of orbit)
        
        orbit_period_steps = int(96 * 60 / self.dt)  # ~57 steps for dt=100s
        sunlight_fraction = 0.64
        
        np.random.seed(self.seed + 300)
        
        for k in range(self.K):
            # Random phase offset for each satellite
            phase = np.random.randint(0, orbit_period_steps)
            
            for t in range(self.T):
                orbit_position = (t + phase) % orbit_period_steps
                # In sun for first 64% of orbit
                V_sun[k, t] = orbit_position < int(sunlight_fraction * orbit_period_steps)
        
        print(f"  Average sunlight fraction: {V_sun.mean():.2%}")
        return V_sun
    
    def generate_all_visibility_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate all visibility matrices at once
        
        Returns:
            (V_target, V_ground, V_sun) tuple
        """
        print("\n" + "="*80)
        print("GENERATING VISIBILITY MATRICES")
        print("="*80)
        
        V_target = self.compute_target_visibility()
        V_ground = self.compute_ground_visibility()
        V_sun = self.compute_sun_visibility()
        
        print("\n" + "="*80)
        print("VISIBILITY GENERATION COMPLETE")
        print("="*80)
        print(f"Target observations available: {V_target.sum()}")
        print(f"Ground downlinks available: {V_ground.sum()}")
        print(f"Sunlight time steps: {V_sun.sum()} / {V_sun.size} ({V_sun.mean():.1%})")
        print("="*80 + "\n")
        
        return V_target, V_ground, V_sun


def generate_quick_visibility(K: int, T: int, dt: int, seed: int = 42,
                              n_targets: int = 50, n_ground: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick function to generate visibility matrices with default settings
    
    Args:
        K: Number of satellites
        T: Number of time steps
        dt: Time step duration in seconds
        seed: Random seed
        n_targets: Number of observation targets
        n_ground: Number of ground stations
    
    Returns:
        (V_target, V_ground, V_sun) tuple
    """
    start_time = Time("2024-01-01 00:00:00", scale="utc")
    
    generator = VisibilityMatrixGenerator(
        n_satellites=K,
        n_targets=n_targets,
        n_ground_stations=n_ground,
        start_time=start_time,
        time_steps=T,
        dt_seconds=dt,
        seed=seed
    )
    
    return generator.generate_all_visibility_matrices()


if __name__ == "__main__":
    # Test the visibility generator
    print("\nTesting Visibility Matrix Generator")
    print("="*80)
    
    # Small test case
    K = 3  # 3 satellites
    T = 100  # 100 time steps
    dt = 100  # 100 seconds per step
    
    V_target, V_ground, V_sun = generate_quick_visibility(
        K=K, T=T, dt=dt, seed=42, n_targets=20, n_ground=3
    )
    
    print(f"\nTest Results:")
    print(f"V_target shape: {V_target.shape}")
    print(f"V_ground shape: {V_ground.shape}")
    print(f"V_sun shape: {V_sun.shape}")
    print(f"\nVisibility statistics:")
    print(f"  Target observations: {V_target.sum()} opportunities")
    print(f"  Ground downlinks: {V_ground.sum()} opportunities")
    print(f"  Sunlight time: {V_sun.sum()} / {V_sun.size} time steps")