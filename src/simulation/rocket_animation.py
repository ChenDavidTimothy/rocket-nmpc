import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon, Circle, Path, PathPatch, Rectangle
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Color palette
COLORS = {
    'main': {
        'background': '#252426',
        'primary1': '#F0A050',  # Orange
        'primary2': '#C07830',  # Dark orange
        'primary3': '#E0C0A0',  # Light orange
        'primary4': '#805030'   # Brown
    },
    'secondary': {
        'red': '#E04040',
        'green': '#50B050',
        'yellow': '#F0C040',
        'blue': '#4080C0',
        'purple': '#8040A0'
    }
}

@dataclass
class RocketAnimationParams:
    # Physical parameters
    mass: float
    g: float
    l: float  # Thrust moment arm
    J: float  # Moment of inertia
    
    # State constraints
    x_max: float
    x_min: float
    z_max: float
    z_min: float
    theta_max: float
    theta_min: float
    
    # Control constraints
    F_max: float
    F_min: float
    gimbal_max: float
    gimbal_min: float
    
    # Visualization scaling
    rocket_length: float = 2.0
    rocket_width: float = 0.3
    nozzle_length: float = 0.4
    nozzle_width: float = 0.2
    fin_length: float = 0.4
    fin_width: float = 0.2
    margin_factor: float = 1.2  
    ground_height_factor: float = 0.1  # Ground height relative to plot height

class RocketMPCAnimation:
    def __init__(self, params: RocketAnimationParams, fps: int = 30):
        self.fps = fps
        self.params = params

    def _create_rocket_shape(self, length: float, width: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nose_length = length * 0.25
        body_length = length * 0.75
        
        # Main body
        body = np.array([
            [-body_length/2, -width/2],
            [body_length/2, -width/2],
            [body_length/2, width/2],
            [-body_length/2, width/2],
        ])
        
        # Pointed nose cone
        nose_tip = np.array([body_length/2 + nose_length, 0])
        
        # Enhanced fins
        fin_length = width * 1.5
        fin_width = width * 0.8
        fin_offset = body_length/2 - width
        
        left_fin = np.array([
            [-fin_offset, -width/2],
            [-fin_offset - fin_length, -width/2 - fin_width],
            [-fin_offset - fin_length/2, -width/2],
        ])
        
        right_fin = np.array([
            [-fin_offset, width/2],
            [-fin_offset - fin_length, width/2 + fin_width],
            [-fin_offset - fin_length/2, width/2],
        ])
        
        return body, nose_tip, left_fin, right_fin

    def _create_thrust_flame(self, thrust_magnitude: float, length: float, width: float) -> np.ndarray:
        flame_length = thrust_magnitude * length
        
        flame_points = []
        n_points = 20
        for i in range(n_points):
            t = i / (n_points - 1)
            current_width = width * (1 - t) * (0.8 + 0.2 * np.sin(t * 8 * np.pi))
            x = -flame_length * t
            flame_points.extend([
                (x, current_width),
                (x, -current_width)
            ])
        
        return np.array(flame_points)

    def _prepare_interpolation(self, t: List[float], xx: np.ndarray, xx1: List[np.ndarray], 
                             u_cl: List[np.ndarray], T: float) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        # Convert to numpy arrays
        t = np.array(t)
        xx = np.array(xx)
        u_cl = np.array(u_cl) if len(u_cl) > 0 else np.zeros((len(t), 2))
        
        # Find unique time points
        unique_indices = []
        last_t = float('-inf')
        for i, current_t in enumerate(t):
            if current_t > last_t:
                unique_indices.append(i)
                last_t = current_t
        
        # Create arrays with unique time points
        t_unique = t[unique_indices]
        xx_unique = xx[:, unique_indices]
        
        # Calculate new time points for desired FPS
        t_final = t_unique[-1]
        n_frames = int(t_final * self.fps) + 1
        t_new = np.linspace(0, t_final, n_frames)
        
        # Interpolate state variables
        xx_interp = []
        for i in range(xx.shape[0]):
            if len(t_unique) < 4:
                f = interp1d(t_unique, xx_unique[i, :], kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
            else:
                try:
                    f = interp1d(t_unique, xx_unique[i, :], kind='cubic', 
                                bounds_error=False, fill_value='extrapolate')
                except ValueError:
                    f = interp1d(t_unique, xx_unique[i, :], kind='linear', 
                                bounds_error=False, fill_value='extrapolate')
            xx_interp.append(f(t_new))
        xx_new = np.vstack(xx_interp)
        
        # Interpolate control inputs
        u_cl_unique = u_cl[unique_indices[:-1]]
        t_control = t_unique[:-1]
        
        u_cl_interp = []
        for i in range(u_cl_unique.shape[1]):
            f = interp1d(t_control, u_cl_unique[:, i], kind='linear',
                        bounds_error=False, fill_value='extrapolate')
            u_cl_interp.append(f(t_new))
        
        u_cl_new = np.vstack(u_cl_interp).T
        
        # Handle predicted trajectories
        xx1_new = []
        n_original = len(xx1)
        for frame_idx in range(len(t_new)):
            original_frame_idx = min(int(frame_idx * n_original / len(t_new)), n_original - 1)
            xx1_new.append(xx1[original_frame_idx])
        
        return t_new, xx_new, xx1_new, u_cl_new

    def create_animation(self, t: List[float], xx: np.ndarray, xx1: List[np.ndarray],
                        u_cl: List[np.ndarray], xs: np.ndarray, N: int, T: float,
                        save_path: Optional[str] = None) -> animation.FuncAnimation:
        plt.style.use('dark_background')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor(COLORS['main']['background'])
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        
        # Main animation axes
        ax = fig.add_subplot(gs[0])
        ax.set_facecolor(COLORS['main']['background'])
        
        # Information panel axes
        ax_info = fig.add_subplot(gs[1])
        ax_info.set_facecolor(COLORS['main']['background'])
        ax_info.axis('off')
        
        # Prepare interpolated data
        t_new, xx_new, xx1_new, u_cl_new = self._prepare_interpolation(t, xx, xx1, u_cl, T)
        
        # Calculate plot limits with margin
        x_data = xx[0, :]
        z_data = xx[1, :]
        x_min = min(min(x_data), self.params.x_min) - self.params.margin_factor
        x_max = max(max(x_data), self.params.x_max) + self.params.margin_factor
        z_min = min(min(z_data), self.params.z_min) - self.params.margin_factor
        z_max = max(max(z_data), self.params.z_max) + self.params.margin_factor
        
        def animate(frame):
            ax.clear()
            ax_info.clear()
            
            # Ensure background color persists
            ax.set_facecolor(COLORS['main']['background'])
            ax_info.set_facecolor(COLORS['main']['background'])
            
            # Set white borders
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('white')
                spine.set_linewidth(1.0)
            
            # Set up main animation plot with dynamic limits
            ax.set_xlim(z_min, z_max)
            ax.set_ylim(x_min, x_max)
            ax.grid(True, alpha=0.2, color=COLORS['main']['primary3'])
            
            # Draw ground with dynamic size
            ground_height = abs(x_max - x_min) * self.params.ground_height_factor
            ground = Rectangle((z_min, x_min), 
                             z_max - z_min, 
                             ground_height,
                             fc=COLORS['main']['primary4'], 
                             alpha=0.5, 
                             ec='none')
            ax.add_patch(ground)
            
            # Draw trajectory with fade effect
            alpha_values = np.linspace(0.2, 1, frame+1)
            for i in range(frame):
                ax.plot(xx_new[1, i:i+2], xx_new[0, i:i+2], 
                       color=COLORS['secondary']['blue'],
                       alpha=alpha_values[i], linewidth=1.5)
            
            # Get current thrust and angle
            if frame < len(u_cl_new):
                thrust_magnitude = u_cl_new[frame][0]
                thrust_angle = -xx_new[4, frame] - u_cl_new[frame][1]
            else:
                thrust_magnitude = 0
                thrust_angle = -xx_new[4, frame]
            
            # Create rocket components with dynamic size
            body, nose_tip, left_fin, right_fin = self._create_rocket_shape(
                self.params.rocket_length, 
                self.params.rocket_width)
            
            # Rotation matrix
            theta = -xx_new[4, frame]
            R = np.array([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ])
            
            # Transform rocket components
            pos = np.array([xx_new[0, frame], xx_new[1, frame]])
            
            body_transformed = np.dot(body, R.T) + pos
            nose_transformed = np.dot(np.array([nose_tip]), R.T) + pos
            left_fin_transformed = np.dot(left_fin, R.T) + pos
            right_fin_transformed = np.dot(right_fin, R.T) + pos
            
            # Draw rocket components
            ax.fill(body_transformed[:, 1], body_transformed[:, 0],
                   COLORS['main']['primary3'], 
                   edgecolor=COLORS['main']['primary1'], linewidth=1)
            
            # Draw nose cone
            ax.fill([body_transformed[1, 1], body_transformed[2, 1], nose_transformed[0, 1]],
                   [body_transformed[1, 0], body_transformed[2, 0], nose_transformed[0, 0]],
                   COLORS['main']['primary2'], 
                   edgecolor=COLORS['main']['primary1'], linewidth=1)
            
            # Draw fins
            ax.fill(left_fin_transformed[:, 1], left_fin_transformed[:, 0],
                   COLORS['main']['primary2'],
                   edgecolor=COLORS['main']['primary1'], linewidth=1)
            ax.fill(right_fin_transformed[:, 1], right_fin_transformed[:, 0],
                   COLORS['main']['primary2'],
                   edgecolor=COLORS['main']['primary1'], linewidth=1)
            
            # Draw thrust flame if thrusting
            if thrust_magnitude > 0:
                flame = self._create_thrust_flame(
                    thrust_magnitude/self.params.F_max,
                    self.params.rocket_length,
                    self.params.rocket_width*0.4)
                
                R_thrust = np.array([
                    [np.cos(thrust_angle), -np.sin(thrust_angle)],
                    [np.sin(thrust_angle), np.cos(thrust_angle)]
                ])
                
                flame_rotated = np.dot(flame, R_thrust.T)
                bottom_point = np.dot(np.array([-self.params.rocket_length/2, 0]), R.T) + pos
                flame_transformed = flame_rotated + bottom_point
                
                n_points = len(flame)//2
                for i in range(n_points-1):
                    points = np.vstack((
                        flame_transformed[2*i:2*i+2],
                        flame_transformed[2*i+2:2*i+4][::-1]
                    ))
                    alpha = 1 - i/n_points
                    color = COLORS['secondary']['yellow'] if i < n_points//2 else COLORS['secondary']['red']
                    ax.fill(points[:, 1], points[:, 0], color=color, alpha=alpha)
            
            # Draw predicted trajectory
            pred_idx = min(frame, len(xx1_new)-1)
            if pred_idx < len(xx1_new):
                pred_traj = xx1_new[pred_idx]
                ax.plot(pred_traj[:, 1], pred_traj[:, 0],
                       color=COLORS['secondary']['green'],
                       linestyle=':', alpha=0.3, linewidth=1.5)
            
            # Main plot labels
            ax.set_xlabel('Horizontal position (m)', color=COLORS['main']['primary3'])
            ax.set_ylabel('Vertical position (m)', color=COLORS['main']['primary3'])
            ax.set_aspect('equal')
            ax.tick_params(colors=COLORS['main']['primary3'])
            
            # Information panel
            time_in_sec = frame / self.fps
            info_text = (
                f'Time: {time_in_sec:.1f}s\n\n'
                f'Position:\n'
                f'X: {xx_new[0, frame]:6.1f}m\n'
                f'Z: {xx_new[1, frame]:6.1f}m\n\n'
                f'Velocity:\n'
                f'U: {xx_new[2, frame]:6.1f}m/s\n'
                f'W: {xx_new[3, frame]:6.1f}m/s\n\n'
                f'Attitude:\n'
                f'θ: {np.rad2deg(xx_new[4, frame]):6.1f}°\n'
                f'ω: {np.rad2deg(xx_new[5, frame]):6.1f}°/s\n\n'
                f'Control:\n'
                f'Thrust: {thrust_magnitude/self.params.F_max*100:6.1f}%\n'
                f'Gimbal: {np.rad2deg(u_cl_new[frame][1] if frame < len(u_cl_new) else 0):6.1f}°\n\n'
                f'System Parameters:\n'
                f'Mass: {self.params.mass:.1f} kg\n'
                f'Moment of inertia: {self.params.J:.1f} kg⋅m²\n'
                f'Thrust arm: {self.params.l:.1f} m'
            )
            
            ax_info.text(0.05, 0.95, 'Rocket Telemetry',
                        fontsize=12, fontweight='bold',
                        color=COLORS['main']['primary1'],
                        transform=ax_info.transAxes)
            
            ax_info.text(0.05, 0.85, info_text,
                        fontfamily='monospace',
                        color=COLORS['main']['primary3'],
                        transform=ax_info.transAxes,
                        verticalalignment='top')
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], color=COLORS['secondary']['blue'],
                          label='Actual Trajectory'),
                plt.Line2D([0], [0], color=COLORS['secondary']['green'],
                          linestyle=':', label='Predicted Trajectory'),
                plt.Line2D([0], [0], color=COLORS['secondary']['yellow'],
                          label='Thrust')
            ]
            legend = ax_info.legend(handles=legend_elements,
                                  loc='lower center',
                                  bbox_to_anchor=(0.5, 0.02))
            plt.setp(legend.get_texts(), color=COLORS['main']['primary3'])
        
        # Calculate frames and interval
        frames = xx_new.shape[1]
        interval = 1000 / self.fps  # milliseconds between frames
        
        # Create animation
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=frames,
            interval=interval,
            blit=False
        )
        
        # Save animation if path provided
        if save_path:
            writer = animation.FFMpegWriter(fps=self.fps, bitrate=3000)
            ani.save(save_path, writer=writer,
                    savefig_kwargs={'facecolor': COLORS['main']['background']})
        
        return ani

    def show(self):
        plt.show()