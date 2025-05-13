import numpy as np
from vispy import app, scene
from vispy.scene import visuals
from vispy.color import ColorArray

# To ensure the application can find the backend
# app.use_app('pyqt6') # Or 'pyqt5', 'pyside6', 'glfw', etc.
# Usually, VisPy auto-detects. If not, uncomment and specify.

class InteractiveVisualizerVisPy:
    def __init__(self, particles, title="N-Body Simulation", point_size_min=1, point_size_max=10):
        self.particles = particles
        self.point_size_min = point_size_min
        self.point_size_max = point_size_max
        self._paused = False

        # Create a canvas and a view
        self.canvas = scene.SceneCanvas(keys='interactive', title=title, size=(1024, 768), show=True)
        self.view = self.canvas.central_widget.add_view()

        # Create a marker visual for the particles
        # We'll set data later in the update method
        self.scatter = visuals.Markers()
        self.view.add(self.scatter)

        # Set up the camera
        # 'turntable' is good for orbiting, 'fly' for first-person, 'arcball' is also common
        self.view.camera = scene.cameras.TurntableCamera(fov=30, distance=10.0, elevation=30, azimuth=0, center=(0,0,0))
        # You might want to adjust distance based on your system scale
        # self.view.camera = scene.cameras.ArcballCamera(fov=30)


        # Add a 3D axis
        self.axis = visuals.XYZAxis(parent=self.view.scene)

        # Timer for animation updates
        self.timer = app.Timer('auto', connect=self.on_timer_update, start=False)
        # 'auto' tries to pick a good interval. Can specify e.g., 1/60.0 for 60 FPS target.

        # Connect key press event for pause/resume
        self.canvas.events.key_press.connect(self.on_key_press)

        self.simulation_step_callback = None # Will be set by the simulation


    def set_simulation_step_callback(self, callback):
        """Sets the function to call to advance the simulation by one step."""
        self.simulation_step_callback = callback

    def update_particle_data(self):
        """Updates the scatter plot data from the ParticleData object."""
        active_idx = self.particles.active_indices
        if active_idx.size == 0:
            self.scatter.set_data(np.empty((0, 3)), size=0)
            return

        pos = self.particles.position[active_idx]
        mass = self.particles.mass[active_idx]
        vel = self.particles.velocity[active_idx] # For color

        # --- Asteroid Visuals ---
        if pos.shape[0] > 0:
            # Color by speed
            speeds = np.linalg.norm(vel, axis=1)
            min_s, max_s = np.min(speeds) if speeds.size > 0 else 0, np.max(speeds) if speeds.size > 0 else 1
            normalized_speeds = (speeds - min_s) / (max_s - min_s + 1e-9) # Add epsilon to avoid div by zero
            
            # Use a colormap (e.g., 'viridis', 'coolwarm', 'autumn')
            # We need to map normalized_speeds (0-1) to RGBA colors
            # Example: simple colormap from blue (low speed) to red (high speed)
            colors = np.zeros((len(normalized_speeds), 4))
            colors[:, 0] = normalized_speeds  # Red channel
            colors[:, 2] = 1 - normalized_speeds # Blue channel
            colors[:, 3] = 0.8 # Alpha

            # Size by mass (log scale often works well)
            min_mass_display = 1e-15 # Avoid log(0)
            log_mass = np.log10(np.maximum(mass, min_mass_display))
            min_lm, max_lm = np.min(log_mass) if log_mass.size > 0 else -15, np.max(log_mass) if log_mass.size > 0 else -14
            
            normalized_sizes = (log_mass - min_lm) / (max_lm - min_lm + 1e-9)
            sizes = self.point_size_min + normalized_sizes * (self.point_size_max - self.point_size_min)
            sizes = np.clip(sizes, self.point_size_min, self.point_size_max)


            self.scatter.set_data(
                pos,
                size=sizes,
                face_color=ColorArray(colors), # Use ColorArray for per-particle colors
                edge_color=None # No edges for many small particles usually looks cleaner
            )
        else:
            self.scatter.set_data(np.empty((0, 3)), size=0) # No other particles

        # Auto-range camera once initially (or periodically if desired)
        # if not hasattr(self, '_ranged_once'):
        #     self.view.camera.set_range()
        #     self._ranged_once = True


    def on_timer_update(self, event):
        """Callback for the animation timer."""
        if not self._paused and self.simulation_step_callback:
            self.simulation_step_callback() # Advance simulation by one step
            self.update_particle_data()     # Update visuals
            self.canvas.update()            # Request a redraw

    def on_key_press(self, event):
        """Handle key presses."""
        if event.key == 'Space':
            self._paused = not self._paused
            print("Paused" if self._paused else "Resumed")
        elif event.key == 'r': # Reset camera
            self.view.camera.reset()
            # Autorange again if needed
            # self.view.camera.set_range(x=(np.min(all_pos[:,0]), np.max(all_pos[:,0])),
            #                           y=(np.min(all_pos[:,1]), np.max(all_pos[:,1])),
            #                           z=(np.min(all_pos[:,2]), np.max(all_pos[:,2])))
            # self.view.camera.viewbox.camera.set_range() # Another way to try to set range

    def start_animation(self):
        self.timer.start()
        self.canvas.app.run()

    def stop_animation(self):
        self.timer.stop()
        self.canvas.app.quit()
