import numpy as np

class ParticleData:
    """
    Manages particle data using Structure of Arrays (SoA) with numpy.
    Handles a dynamic number of active particles up to a fixed capacity.
    """
    def __init__(self, capacity: int):
        """
        Initializes data storage with a maximum capacity.
        Args:
            capacity: The maximum number of particles the structure can hold.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive.")
        self.capacity: int = capacity
        self.n_particles: int = 0  # Current number of active particles

        # Core physical properties (SoA)
        self.position: np.ndarray = np.zeros((capacity, 3), dtype=np.float64)
        self.velocity: np.ndarray = np.zeros((capacity, 3), dtype=np.float64)
        self.acceleration: np.ndarray = np.zeros((capacity, 3), dtype=np.float64)
        self.mass: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self.radius: np.ndarray = np.zeros(capacity, dtype=np.float64)

        # Management properties
        self.ids: np.ndarray = np.zeros(capacity, dtype=np.int64) # Unique identifier
        self.active: np.ndarray = np.zeros(capacity, dtype=bool) # Activity status
        self.merge_status: np.ndarray = np.full(capacity, -1, dtype=np.int64)

        self._next_id: int = 0

    def add_particle(self, pos, vel, mass, radius=0.0):
        """Adds a new particle to the simulation."""
        if self.n_particles >= self.capacity:
            raise MemoryError("ParticleData capacity exceeded.")

        idx = self.n_particles # Index where the new particle will be added

        self.position[idx] = np.asarray(pos, dtype=np.float64)
        self.velocity[idx] = np.asarray(vel, dtype=np.float64)
        self.mass[idx] = float(mass)
        self.radius[idx] = float(radius)
        self.acceleration[idx] = np.zeros(3, dtype=np.float64) # Init accel to zero
        self.ids[idx] = self._next_id
        self.active[idx] = True
        self.merge_status[idx] = idx

        self.n_particles += 1
        self._next_id += 1
        return idx # Return the index of the added particle

    def remove_particle(self, index: int):
        """
        Marks a particle as inactive. Does not immediately reclaim memory.
        Note: A separate 'compact' method would be needed to reclaim memory.
        """
        if 0 <= index < self.n_particles and self.active[index]:
            self.active[index] = False
            # Optional: Could swap with the last active particle and decrement n_particles
            # for faster iteration, but makes indices unstable.
            # Let's stick to the simple 'active' flag for now.
        # else: particle already removed or out of bounds (or never added)

    # --- Convenience properties to get active data ---
    # These return copies or views masked by the 'active' status. Be careful
    # if modifying the returned arrays directly is intended.

    @property
    def active_indices(self) -> np.ndarray:
        """Returns the indices of currently active particles."""
        # Note: np.where returns a tuple, we need the first element
        return np.where(self.active[:self.n_particles])[0]

    @property
    def num_active_particles(self) -> int:
        """Returns the count of currently active particles."""
        return int(np.sum(self.active[:self.n_particles]))

    def get_active_positions(self) -> np.ndarray:
        return self.position[self.active]

    def get_active_velocities(self) -> np.ndarray:
        return self.velocity[self.active]

    def get_active_masses(self) -> np.ndarray:
        return self.mass[self.active]

    def get_active_radii(self) -> np.ndarray:
        return self.radius[self.active]

    def get_active_ids(self) -> np.ndarray:
        return self.ids[self.active]

    def __len__(self) -> int:
        """Returns the number of active particles."""
        return self.num_active_particles

    def get_merge_status(self, index_x):
        idx = self.merge_status[index_x]
        if idx == -1 or idx == index_x:
            return idx
        self.merge_status[index_x] = self.get_merge_status(idx)
        return self.merge_status[index_x]

    def merge(self, index_x, index_y): 
        idx, idy = self.get_merge_status(index_x), self.get_merge_status(index_y)
        if (not self.active[idx]) or (not self.active[idy]) or idx == idy:
            return
        if self.mass[idx] > self.mass[idy]:
            idx, idy = idy, idx
        self.remove_particle(idx)
        self.merge_status[idx] = idy
        mx, my = self.mass[idx], self.mass[idy]
        rx, ry = mx / (mx + my), my / (mx + my)
        self.position[idy] = rx * self.position[idx] + ry * self.position[idy]
        self.velocity[idy] = rx * self.velocity[idx] + ry * self.velocity[idy]
        self.acceleration[idy] = rx * self.acceleration[idx] + ry * self.acceleration[idy]
        self.mass[idy] = self.mass[idx] + self.mass[idy]
        self.radius[idy] = (self.radius[idx]**3 + self.radius[idy]**3)**(1./3.)

        self.acceleration[idx] = np.zeros(3)

    def compact(self, verbose=True):
        """
        Removes inactive particles and compacts the arrays.
        This changes the indices of particles! Only use when necessary.
        """
        active_mask = self.active[:self.n_particles]
        num_active = int(np.sum(active_mask))

        if num_active == self.n_particles:
            return # Nothing to compact

        self.position[:num_active] = self.position[:self.n_particles][active_mask]
        self.velocity[:num_active] = self.velocity[:self.n_particles][active_mask]
        self.acceleration[:num_active] = self.acceleration[:self.n_particles][active_mask]
        self.mass[:num_active] = self.mass[:self.n_particles][active_mask]
        self.radius[:num_active] = self.radius[:self.n_particles][active_mask]
        self.ids[:num_active] = self.ids[:self.n_particles][active_mask]
        self.active[:num_active] = True
        self.active[num_active:self.n_particles] = False # Ensure rest are inactive
        self.merge_status[:num_active] = np.arange(0, num_active, 1)
        self.merge_status[num_active:self.n_particles] = -1

        # Clear the remaining (now unused) part of the arrays
        self.position[num_active:self.n_particles].fill(0)
        self.velocity[num_active:self.n_particles].fill(0)
        self.acceleration[num_active:self.n_particles].fill(0)
        self.mass[num_active:self.n_particles].fill(0)
        self.radius[num_active:self.n_particles].fill(0)
        self.ids[num_active:self.n_particles].fill(0)

        self.n_particles = num_active
        if verbose:
            print(f"Compacted arrays. New active particle count: {self.n_particles}")
