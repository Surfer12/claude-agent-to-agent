"""
Phase-Space Visualization Module
Interactive 3D plotting and analysis tools for hybrid dynamical systems
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

from hybrid_phase_space_system import HybridDynamicalSystem, create_example_system


class PhaseSpaceVisualizer:
    """Interactive visualization tools for phase-space trajectories"""
    
    def __init__(self, system: HybridDynamicalSystem):
        self.system = system
        self.trajectory_data = None
        
        # Color schemes
        self.colors = {
            'trajectory': '#1f77b4',  # Blue
            'start': '#2ca02c',       # Green
            'end': '#d62728',         # Red
            'analysis_points': '#ff7f0e'  # Orange
        }
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def plot_3d_trajectory(self, 
                          figsize: Tuple[int, int] = (12, 9),
                          show_analysis_points: bool = True,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a static 3D plot of the phase-space trajectory
        Reproduces the style from the original image
        """
        
        if self.trajectory_data is None:
            t_eval, trajectory = self.system.generate_trajectory()
            self.trajectory_data = (t_eval, trajectory)
        else:
            t_eval, trajectory = self.trajectory_data
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        alpha = trajectory[:, 0]
        lambda1 = trajectory[:, 1]
        lambda2 = trajectory[:, 2]
        
        # Plot the main trajectory
        ax.plot(alpha, lambda1, lambda2, 
                color=self.colors['trajectory'], 
                linewidth=2.5, 
                alpha=0.8, 
                label='Trajectory')
        
        # Mark start and end points
        ax.scatter(alpha[0], lambda1[0], lambda2[0], 
                  color=self.colors['start'], 
                  s=100, 
                  marker='o', 
                  label='Start')
        
        ax.scatter(alpha[-1], lambda1[-1], lambda2[-1], 
                  color=self.colors['end'], 
                  s=100, 
                  marker='s', 
                  label='End')
        
        # Add analysis points if requested
        if show_analysis_points:
            analysis_times = [2.5, 5.0, 7.5]  # Example analysis points
            for i, t in enumerate(analysis_times):
                idx = np.argmin(np.abs(t_eval - t))
                ax.scatter(alpha[idx], lambda1[idx], lambda2[idx],
                          color=self.colors['analysis_points'],
                          s=80,
                          marker='^',
                          alpha=0.8)
                
                # Add time annotation
                ax.text(alpha[idx], lambda1[idx], lambda2[idx] + 0.1,
                       f't={t}',
                       fontsize=10,
                       ha='center')
        
        # Customize the plot
        ax.set_xlabel('α(t)', fontsize=14, labelpad=10)
        ax.set_ylabel('λ₁(t)', fontsize=14, labelpad=10)
        ax.set_zlabel('λ₂(t)', fontsize=14, labelpad=10)
        
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_zlim(0, 2)
        
        # Add title
        ax.set_title('Phase-Space Trajectory: α(t), λ₁(t), λ₂(t)', 
                    fontsize=16, 
                    pad=20)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # Add grid and make it look professional
        ax.grid(True, alpha=0.3)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Make pane edges more subtle
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_interactive_3d(self) -> go.Figure:
        """Create an interactive 3D plot using Plotly"""
        
        if self.trajectory_data is None:
            t_eval, trajectory = self.system.generate_trajectory()
            self.trajectory_data = (t_eval, trajectory)
        else:
            t_eval, trajectory = self.trajectory_data
        
        # Create the main trajectory trace
        trajectory_trace = go.Scatter3d(
            x=trajectory[:, 0],
            y=trajectory[:, 1],
            z=trajectory[:, 2],
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(size=2, opacity=0.8),
            name='Trajectory',
            hovertemplate='α: %{x:.3f}<br>λ₁: %{y:.3f}<br>λ₂: %{z:.3f}<extra></extra>'
        )
        
        # Start point
        start_trace = go.Scatter3d(
            x=[trajectory[0, 0]],
            y=[trajectory[0, 1]],
            z=[trajectory[0, 2]],
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Start',
            hovertemplate='Start: α=%{x:.3f}, λ₁=%{y:.3f}, λ₂=%{z:.3f}<extra></extra>'
        )
        
        # End point
        end_trace = go.Scatter3d(
            x=[trajectory[-1, 0]],
            y=[trajectory[-1, 1]],
            z=[trajectory[-1, 2]],
            mode='markers',
            marker=dict(size=10, color='red', symbol='square'),
            name='End',
            hovertemplate='End: α=%{x:.3f}, λ₁=%{y:.3f}, λ₂=%{z:.3f}<extra></extra>'
        )
        
        # Create figure
        fig = go.Figure(data=[trajectory_trace, start_trace, end_trace])
        
        # Update layout
        fig.update_layout(
            title='Interactive Phase-Space Trajectory: α(t), λ₁(t), λ₂(t)',
            scene=dict(
                xaxis_title='α(t)',
                yaxis_title='λ₁(t)',
                zaxis_title='λ₂(t)',
                xaxis=dict(range=[0, 2]),
                yaxis=dict(range=[0, 2]),
                zaxis=dict(range=[0, 2]),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_parameter_evolution(self, figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """Plot the evolution of each parameter over time"""
        
        if self.trajectory_data is None:
            t_eval, trajectory = self.system.generate_trajectory()
            self.trajectory_data = (t_eval, trajectory)
        else:
            t_eval, trajectory = self.trajectory_data
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # α(t) evolution
        axes[0].plot(t_eval, trajectory[:, 0], 'b-', linewidth=2, label='α(t)')
        axes[0].set_xlabel('Time t')
        axes[0].set_ylabel('α(t)')
        axes[0].set_title('Symbolic/Neural Balance')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 2)
        
        # λ₁(t) evolution
        axes[1].plot(t_eval, trajectory[:, 1], 'g-', linewidth=2, label='λ₁(t)')
        axes[1].set_xlabel('Time t')
        axes[1].set_ylabel('λ₁(t)')
        axes[1].set_title('Cognitive Penalty Weight')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 2)
        
        # λ₂(t) evolution
        axes[2].plot(t_eval, trajectory[:, 2], 'r-', linewidth=2, label='λ₂(t)')
        axes[2].set_xlabel('Time t')
        axes[2].set_ylabel('λ₂(t)')
        axes[2].set_title('Efficiency Penalty Weight')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0, 2)
        
        plt.tight_layout()
        return fig
    
    def plot_psi_analysis(self, 
                         x_test: np.ndarray,
                         n_time_points: int = 50,
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Comprehensive analysis of Ψ(x) computation along the trajectory
        """
        
        if self.trajectory_data is None:
            t_eval, trajectory = self.system.generate_trajectory()
            self.trajectory_data = (t_eval, trajectory)
        else:
            t_eval, trajectory = self.trajectory_data
        
        # Select time points for analysis
        analysis_times = np.linspace(t_eval[0], t_eval[-1], n_time_points)
        
        # Compute analysis data
        analysis_data = []
        for t in analysis_times:
            analysis = self.system.analyze_trajectory_point(t, x_test)
            analysis_data.append(analysis)
        
        # Extract data for plotting
        times = [d['time'] for d in analysis_data]
        alphas = [d['trajectory_point']['alpha'] for d in analysis_data]
        lambda1s = [d['trajectory_point']['lambda1'] for d in analysis_data]
        lambda2s = [d['trajectory_point']['lambda2'] for d in analysis_data]
        
        symbolic_vals = [d['components']['symbolic'] for d in analysis_data]
        neural_vals = [d['components']['neural'] for d in analysis_data]
        hybrid_vals = [d['components']['hybrid'] for d in analysis_data]
        
        penalty_terms = [d['penalties']['penalty_term'] for d in analysis_data]
        prob_terms = [d['probabilistic_bias'] for d in analysis_data]
        psi_contributions = [d['psi_contribution'] for d in analysis_data]
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Trajectory parameters
        axes[0, 0].plot(times, alphas, 'b-', linewidth=2, label='α(t)')
        axes[0, 0].plot(times, lambda1s, 'g-', linewidth=2, label='λ₁(t)')
        axes[0, 0].plot(times, lambda2s, 'r-', linewidth=2, label='λ₂(t)')
        axes[0, 0].set_xlabel('Time t')
        axes[0, 0].set_ylabel('Parameter Value')
        axes[0, 0].set_title('Trajectory Parameters')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Component outputs
        axes[0, 1].plot(times, symbolic_vals, 'b-', linewidth=2, label='S(x)')
        axes[0, 1].plot(times, neural_vals, 'orange', linewidth=2, label='N(x)')
        axes[0, 1].plot(times, hybrid_vals, 'purple', linewidth=2, label='Hybrid')
        axes[0, 1].set_xlabel('Time t')
        axes[0, 1].set_ylabel('Output Value')
        axes[0, 1].set_title('Component Outputs')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Penalty and bias terms
        axes[0, 2].plot(times, penalty_terms, 'brown', linewidth=2, label='Penalty Term')
        axes[0, 2].plot(times, prob_terms, 'cyan', linewidth=2, label='P(H|E,β)')
        axes[0, 2].set_xlabel('Time t')
        axes[0, 2].set_ylabel('Term Value')
        axes[0, 2].set_title('Regularization Terms')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Ψ contributions
        axes[1, 0].plot(times, psi_contributions, 'magenta', linewidth=2)
        axes[1, 0].set_xlabel('Time t')
        axes[1, 0].set_ylabel('Ψₜ(x)')
        axes[1, 0].set_title('Ψ(x) Integrand Contributions')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Cumulative integral
        cumulative_psi = np.cumsum(psi_contributions) * (times[1] - times[0])
        axes[1, 1].plot(times, cumulative_psi, 'darkgreen', linewidth=2)
        axes[1, 1].set_xlabel('Time t')
        axes[1, 1].set_ylabel('∫Ψₜ(x)dt')
        axes[1, 1].set_title('Cumulative Ψ(x) Integral')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Phase portrait in α-λ₁ space
        axes[1, 2].plot(alphas, lambda1s, 'navy', linewidth=2, alpha=0.7)
        axes[1, 2].scatter(alphas[0], lambda1s[0], color='green', s=100, marker='o', label='Start')
        axes[1, 2].scatter(alphas[-1], lambda1s[-1], color='red', s=100, marker='s', label='End')
        axes[1, 2].set_xlabel('α(t)')
        axes[1, 2].set_ylabel('λ₁(t)')
        axes[1, 2].set_title('Phase Portrait: α vs λ₁')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_animated_trajectory(self, 
                                 save_path: Optional[str] = None,
                                 interval: int = 100,
                                 frames: int = 200) -> animation.FuncAnimation:
        """Create an animated 3D trajectory visualization"""
        
        if self.trajectory_data is None:
            t_eval, trajectory = self.system.generate_trajectory()
            self.trajectory_data = (t_eval, trajectory)
        else:
            t_eval, trajectory = self.trajectory_data
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up the axes
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_zlim(0, 2)
        ax.set_xlabel('α(t)')
        ax.set_ylabel('λ₁(t)')
        ax.set_zlabel('λ₂(t)')
        ax.set_title('Animated Phase-Space Trajectory')
        
        # Initialize empty line and point
        line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.8)
        point, = ax.plot([], [], [], 'ro', markersize=8)
        
        def animate(frame):
            # Calculate how much of the trajectory to show
            n_points = min(frame + 1, len(trajectory))
            
            if n_points > 1:
                # Update trajectory line
                line.set_data_3d(trajectory[:n_points, 0], 
                                trajectory[:n_points, 1], 
                                trajectory[:n_points, 2])
                
                # Update current point
                current_idx = n_points - 1
                point.set_data_3d([trajectory[current_idx, 0]], 
                                [trajectory[current_idx, 1]], 
                                [trajectory[current_idx, 2]])
                
                # Update title with current time
                current_time = t_eval[current_idx] if current_idx < len(t_eval) else t_eval[-1]
                ax.set_title(f'Phase-Space Trajectory (t = {current_time:.2f})')
            
            return line, point
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=interval, blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        return anim
    
    def plot_comparison_study(self, 
                            systems: List[HybridDynamicalSystem],
                            system_names: List[str],
                            figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """Compare multiple systems with different parameters"""
        
        fig = plt.figure(figsize=figsize)
        
        # 3D comparison plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, (system, name) in enumerate(zip(systems, system_names)):
            t_eval, trajectory = system.generate_trajectory()
            
            color = colors[i % len(colors)]
            ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                    color=color, linewidth=2, alpha=0.8, label=name)
            
            # Mark start points
            ax1.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                       color=color, s=50, marker='o', alpha=0.8)
        
        ax1.set_xlabel('α(t)')
        ax1.set_ylabel('λ₁(t)')
        ax1.set_zlabel('λ₂(t)')
        ax1.set_title('Trajectory Comparison')
        ax1.legend()
        
        # Parameter evolution comparison
        ax2 = fig.add_subplot(222)
        for i, (system, name) in enumerate(zip(systems, system_names)):
            t_eval, trajectory = system.generate_trajectory()
            color = colors[i % len(colors)]
            ax2.plot(t_eval, trajectory[:, 0], color=color, linewidth=2, label=f'{name} α(t)')
        
        ax2.set_xlabel('Time t')
        ax2.set_ylabel('α(t)')
        ax2.set_title('α(t) Evolution Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Similar plots for λ₁ and λ₂
        ax3 = fig.add_subplot(223)
        for i, (system, name) in enumerate(zip(systems, system_names)):
            t_eval, trajectory = system.generate_trajectory()
            color = colors[i % len(colors)]
            ax3.plot(t_eval, trajectory[:, 1], color=color, linewidth=2, label=f'{name} λ₁(t)')
        
        ax3.set_xlabel('Time t')
        ax3.set_ylabel('λ₁(t)')
        ax3.set_title('λ₁(t) Evolution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(224)
        for i, (system, name) in enumerate(zip(systems, system_names)):
            t_eval, trajectory = system.generate_trajectory()
            color = colors[i % len(colors)]
            ax4.plot(t_eval, trajectory[:, 2], color=color, linewidth=2, label=f'{name} λ₂(t)')
        
        ax4.set_xlabel('Time t')
        ax4.set_ylabel('λ₂(t)')
        ax4.set_title('λ₂(t) Evolution Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def create_demo_visualizations():
    """Create demonstration visualizations matching the walk-through"""
    
    # Create the example system
    system = create_example_system()
    visualizer = PhaseSpaceVisualizer(system)
    
    print("Generating phase-space trajectory visualizations...")
    
    # 1. Main 3D trajectory plot
    fig1 = visualizer.plot_3d_trajectory(show_analysis_points=True)
    fig1.savefig('/workspace/phase_space_trajectory_3d.png', dpi=300, bbox_inches='tight')
    print("✓ 3D trajectory plot saved")
    
    # 2. Parameter evolution over time
    fig2 = visualizer.plot_parameter_evolution()
    fig2.savefig('/workspace/parameter_evolution.png', dpi=300, bbox_inches='tight')
    print("✓ Parameter evolution plot saved")
    
    # 3. Comprehensive Ψ(x) analysis
    test_x = np.array([1.0])
    fig3 = visualizer.plot_psi_analysis(test_x, n_time_points=100)
    fig3.savefig('/workspace/psi_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Ψ(x) analysis plot saved")
    
    # 4. Interactive plot (if plotly is available)
    try:
        fig4 = visualizer.plot_interactive_3d()
        fig4.write_html('/workspace/interactive_trajectory.html')
        print("✓ Interactive 3D plot saved as HTML")
    except Exception as e:
        print(f"Note: Interactive plot not created - {e}")
    
    print("\nAll visualizations generated successfully!")
    return visualizer


if __name__ == "__main__":
    create_demo_visualizations()