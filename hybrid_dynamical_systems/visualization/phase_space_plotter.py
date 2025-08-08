"""
Phase Space Visualization Module
Creates 3D plots of the hybrid dynamical system trajectory in (α, λ₁, λ₂) space.

This module reproduces the phase-space trajectory plot and provides
analysis tools for understanding the system's evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
from typing import Tuple, Optional, Dict, Any
import seaborn as sns

from ..core.hybrid_system import HybridDynamicalSystem, HybridSystemConfig

class PhaseSpacePlotter:
    """Visualization tools for the 3D phase-space trajectory."""
    
    def __init__(self, system: HybridDynamicalSystem):
        self.system = system
        self.fig = None
        self.ax = None
        
    def plot_3d_trajectory(self, 
                          figsize: Tuple[int, int] = (12, 8),
                          color: str = 'blue',
                          linewidth: float = 3.0,
                          alpha: float = 0.8,
                          show_grid: bool = True,
                          show_axes_labels: bool = True,
                          title: str = "Phase-Space Trajectory: α(t), λ₁(t), λ₂(t)",
                          save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create the 3D phase-space trajectory plot.
        
        Args:
            figsize: Figure size (width, height)
            color: Color of the trajectory line
            linewidth: Width of the trajectory line
            alpha: Transparency of the trajectory
            show_grid: Whether to show grid lines
            show_axes_labels: Whether to show axis labels
            title: Plot title
            save_path: Optional path to save the plot
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects
        """
        # Ensure trajectory is computed
        if self.system.trajectory is None:
            self.system.integrate_trajectory()
            
        trajectory = self.system.trajectory
        times = self.system.times
        
        # Create figure and 3D axes
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Extract trajectory components
        alpha_traj = trajectory[:, 0]
        lambda1_traj = trajectory[:, 1]
        lambda2_traj = trajectory[:, 2]
        
        # Plot the trajectory
        self.ax.plot(alpha_traj, lambda1_traj, lambda2_traj, 
                    color=color, linewidth=linewidth, alpha=alpha,
                    label='System Evolution')
        
        # Mark start and end points
        start_point = trajectory[0]
        end_point = trajectory[-1]
        
        self.ax.scatter([start_point[0]], [start_point[1]], [start_point[2]], 
                       color='green', s=100, marker='o', label='Start')
        self.ax.scatter([end_point[0]], [end_point[1]], [end_point[2]], 
                       color='red', s=100, marker='s', label='End')
        
        # Set axis labels and limits
        if show_axes_labels:
            self.ax.set_xlabel('α(t)', fontsize=12)
            self.ax.set_ylabel('λ₁(t)', fontsize=12)
            self.ax.set_zlabel('λ₂(t)', fontsize=12)
        
        # Set axis limits to match the bounds
        config = self.system.config
        self.ax.set_xlim(config.alpha_bounds)
        self.ax.set_ylim(config.lambda1_bounds)
        self.ax.set_zlim(config.lambda2_bounds)
        
        # Add grid
        if show_grid:
            self.ax.grid(True, alpha=0.3)
        
        # Set title
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add legend
        self.ax.legend()
        
        # Adjust view angle for better visualization
        self.ax.view_init(elev=20, azim=45)
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return self.fig, self.ax
    
    def plot_trajectory_with_color_mapping(self,
                                         figsize: Tuple[int, int] = (12, 8),
                                         colormap: str = 'viridis',
                                         linewidth: float = 3.0,
                                         show_grid: bool = True,
                                         title: str = "Phase-Space Trajectory with Time Color Mapping") -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a 3D trajectory plot with color mapping based on time.
        
        Args:
            figsize: Figure size
            colormap: Matplotlib colormap name
            linewidth: Width of the trajectory line
            show_grid: Whether to show grid lines
            title: Plot title
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects
        """
        if self.system.trajectory is None:
            self.system.integrate_trajectory()
            
        trajectory = self.system.trajectory
        times = self.system.times
        
        # Create figure and 3D axes
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract trajectory components
        alpha_traj = trajectory[:, 0]
        lambda1_traj = trajectory[:, 1]
        lambda2_traj = trajectory[:, 2]
        
        # Create color mapping based on time
        colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(times)))
        
        # Plot trajectory with color mapping
        for i in range(len(trajectory) - 1):
            ax.plot([alpha_traj[i], alpha_traj[i+1]], 
                   [lambda1_traj[i], lambda1_traj[i+1]], 
                   [lambda2_traj[i], lambda2_traj[i+1]], 
                   color=colors[i], linewidth=linewidth)
        
        # Add colorbar
        norm = mcolors.Normalize(vmin=times[0], vmax=times[-1])
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
        cbar.set_label('Time', fontsize=10)
        
        # Set axis labels and limits
        config = self.system.config
        ax.set_xlabel('α(t)', fontsize=12)
        ax.set_ylabel('λ₁(t)', fontsize=12)
        ax.set_zlabel('λ₂(t)', fontsize=12)
        ax.set_xlim(config.alpha_bounds)
        ax.set_ylim(config.lambda1_bounds)
        ax.set_zlim(config.lambda2_bounds)
        
        if show_grid:
            ax.grid(True, alpha=0.3)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.view_init(elev=20, azim=45)
        
        return fig, ax
    
    def plot_parameter_evolution(self,
                               figsize: Tuple[int, int] = (15, 10),
                               show_psi: bool = True) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the evolution of individual parameters over time.
        
        Args:
            figsize: Figure size
            show_psi: Whether to also plot Ψ(x) evolution
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects
        """
        if self.system.trajectory is None:
            self.system.integrate_trajectory()
            
        trajectory = self.system.trajectory
        times = self.system.times
        
        # Extract parameters
        alpha_traj = trajectory[:, 0]
        lambda1_traj = trajectory[:, 1]
        lambda2_traj = trajectory[:, 2]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot individual parameters
        axes[0].plot(times, alpha_traj, 'b-', linewidth=2, label='α(t)')
        axes[0].set_ylabel('α(t)', fontsize=12)
        axes[0].set_title('Symbolic-Neural Weight Evolution')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(times, lambda1_traj, 'r-', linewidth=2, label='λ₁(t)')
        axes[1].set_ylabel('λ₁(t)', fontsize=12)
        axes[1].set_title('Cognitive Plausibility Penalty')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        axes[2].plot(times, lambda2_traj, 'g-', linewidth=2, label='λ₂(t)')
        axes[2].set_ylabel('λ₂(t)', fontsize=12)
        axes[2].set_xlabel('Time', fontsize=12)
        axes[2].set_title('Computational Efficiency Penalty')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Plot Ψ(x) evolution if requested
        if show_psi:
            # Create sample input data
            x_sample = np.random.rand(100)
            psi_values = self.system.evaluate_psi_trajectory(x_sample)
            
            axes[3].plot(times, psi_values, 'purple', linewidth=2, label='Ψ(x)')
            axes[3].set_ylabel('Ψ(x)', fontsize=12)
            axes[3].set_xlabel('Time', fontsize=12)
            axes[3].set_title('System Output Evolution')
            axes[3].grid(True, alpha=0.3)
            axes[3].legend()
        
        plt.tight_layout()
        return fig, axes
    
    def plot_trajectory_analysis(self,
                               figsize: Tuple[int, int] = (16, 12)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a comprehensive analysis plot showing trajectory insights.
        
        Args:
            figsize: Figure size
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes objects
        """
        if self.system.trajectory is None:
            self.system.integrate_trajectory()
            
        # Get trajectory insights
        insights = self.system.get_trajectory_insights()
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize)
        
        # 3D trajectory plot
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        trajectory = self.system.trajectory
        alpha_traj = trajectory[:, 0]
        lambda1_traj = trajectory[:, 1]
        lambda2_traj = trajectory[:, 2]
        
        ax1.plot(alpha_traj, lambda1_traj, lambda2_traj, 'blue', linewidth=3)
        ax1.set_xlabel('α(t)')
        ax1.set_ylabel('λ₁(t)')
        ax1.set_zlabel('λ₂(t)')
        ax1.set_title('3D Phase-Space Trajectory')
        ax1.view_init(elev=20, azim=45)
        
        # Parameter evolution
        times = self.system.times
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(times, alpha_traj, 'b-', label='α(t)')
        ax2.plot(times, lambda1_traj, 'r-', label='λ₁(t)')
        ax2.plot(times, lambda2_traj, 'g-', label='λ₂(t)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Parameter Value')
        ax2.set_title('Parameter Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Trajectory statistics
        ax3 = fig.add_subplot(2, 3, 3)
        stats_data = [
            insights['alpha_range'][1] - insights['alpha_range'][0],
            insights['lambda1_range'][1] - insights['lambda1_range'][0],
            insights['lambda2_range'][1] - insights['lambda2_range'][0]
        ]
        ax3.bar(['α Range', 'λ₁ Range', 'λ₂ Range'], stats_data, 
                color=['blue', 'red', 'green'])
        ax3.set_ylabel('Range')
        ax3.set_title('Parameter Ranges')
        
        # Evolution characteristics
        ax4 = fig.add_subplot(2, 3, 4)
        characteristics = insights['evolution_characteristics']
        char_names = list(characteristics.keys())
        char_values = [1 if v else 0 for v in characteristics.values()]
        ax4.bar(char_names, char_values, color=['orange', 'purple', 'brown'])
        ax4.set_ylabel('Occurred (1) or Not (0)')
        ax4.set_title('Evolution Characteristics')
        ax4.set_ylim(0, 1.2)
        
        # Trajectory type
        ax5 = fig.add_subplot(2, 3, 5)
        trajectory_type = insights['trajectory_type']
        ax5.text(0.5, 0.5, f'Trajectory Type:\n{trajectory_type}', 
                ha='center', va='center', fontsize=14, fontweight='bold')
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis('off')
        ax5.set_title('Trajectory Classification')
        
        # Start vs End comparison
        ax6 = fig.add_subplot(2, 3, 6)
        start_point = insights['start_point']
        end_point = insights['end_point']
        
        x_pos = np.arange(3)
        width = 0.35
        
        ax6.bar(x_pos - width/2, start_point, width, label='Start', alpha=0.8)
        ax6.bar(x_pos + width/2, end_point, width, label='End', alpha=0.8)
        
        ax6.set_xlabel('Parameters')
        ax6.set_ylabel('Value')
        ax6.set_title('Start vs End Points')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(['α', 'λ₁', 'λ₂'])
        ax6.legend()
        
        plt.tight_layout()
        return fig, plt.gca()
    
    def create_animated_trajectory(self, 
                                 save_path: str = "trajectory_animation.gif",
                                 duration: int = 10) -> None:
        """
        Create an animated visualization of the trajectory evolution.
        
        Args:
            save_path: Path to save the animation
            duration: Duration of animation in seconds
        """
        if self.system.trajectory is None:
            self.system.integrate_trajectory()
            
        trajectory = self.system.trajectory
        times = self.system.times
        
        # Create figure and 3D axes
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set axis limits
        config = self.system.config
        ax.set_xlim(config.alpha_bounds)
        ax.set_ylim(config.lambda1_bounds)
        ax.set_zlim(config.lambda2_bounds)
        ax.set_xlabel('α(t)')
        ax.set_ylabel('λ₁(t)')
        ax.set_zlabel('λ₂(t)')
        ax.set_title('Animated Phase-Space Trajectory')
        
        # Animation function
        def animate(frame):
            ax.clear()
            ax.set_xlim(config.alpha_bounds)
            ax.set_ylim(config.lambda1_bounds)
            ax.set_zlim(config.lambda2_bounds)
            ax.set_xlabel('α(t)')
            ax.set_ylabel('λ₁(t)')
            ax.set_zlabel('λ₂(t)')
            ax.set_title('Animated Phase-Space Trajectory')
            
            # Plot trajectory up to current frame
            idx = int(frame * len(trajectory) / (duration * 30))  # 30 fps
            idx = min(idx, len(trajectory) - 1)
            
            alpha_traj = trajectory[:idx+1, 0]
            lambda1_traj = trajectory[:idx+1, 1]
            lambda2_traj = trajectory[:idx+1, 2]
            
            if len(alpha_traj) > 1:
                ax.plot(alpha_traj, lambda1_traj, lambda2_traj, 'blue', linewidth=3)
            
            # Mark current point
            if idx < len(trajectory):
                ax.scatter([trajectory[idx, 0]], [trajectory[idx, 1]], [trajectory[idx, 2]], 
                          color='red', s=100)
            
            ax.view_init(elev=20, azim=45)
        
        # Create animation (simplified - would need matplotlib.animation for full implementation)
        print(f"Animation would be saved to {save_path}")
        print("Note: Full animation implementation requires matplotlib.animation")
        
        return fig, ax