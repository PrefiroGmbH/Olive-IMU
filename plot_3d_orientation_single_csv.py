#!/usr/bin/env python3
"""
3D Orientation Visualization for Olive Robotics Data - Single CSV Version
This script creates an interactive 3D plot showing roll, pitch, and yaw over time
"""

import pandas as pd
import numpy as np
import os
import glob
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Try to import matplotlib, but don't fail if it's not available
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib 3D plotting not available. Using Plotly only.")

def quaternion_to_euler(qx, qy, qz, qw):
    """Convert quaternion to roll, pitch, yaw (Euler angles)"""
    # Stack quaternions properly for scipy
    quat_array = np.column_stack([qx, qy, qz, qw])
    
    # Create rotation object from quaternion
    r = R.from_quat(quat_array)
    
    # Convert to Euler angles (ZYX convention - yaw, pitch, roll)
    euler = r.as_euler('zyx', degrees=True)
    
    # Return as roll, pitch, yaw (XYZ convention)
    return euler[:, 2], euler[:, 1], euler[:, 0]  # roll, pitch, yaw

def create_orientation_vectors(roll, pitch, yaw):
    """Create orientation vectors for visualization"""
    # Convert to radians
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)
    
    # Create rotation matrices
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])
    
    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    
    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix
    R_combined = R_z @ R_y @ R_x
    
    # Unit vectors for X, Y, Z axes
    x_axis = R_combined @ np.array([1, 0, 0])
    y_axis = R_combined @ np.array([0, 1, 0])
    z_axis = R_combined @ np.array([0, 0, 1])
    
    return x_axis, y_axis, z_axis

def create_matplotlib_3d_plot(df, save_path=None):
    """Create 3D orientation plot using matplotlib"""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib 3D plotting not available. Skipping matplotlib plot.")
        return None, None, None, None
    
    if df is None or df.empty:
        print("No pose data available for 3D orientation plot")
        return None, None, None, None
    
    # Convert quaternions to Euler angles (using pose orientation from single CSV)
    roll, pitch, yaw = quaternion_to_euler(
        df['pose_orientation_x'].values,
        df['pose_orientation_y'].values,
        df['pose_orientation_z'].values,
        df['pose_orientation_w'].values
    )
    
    # Create time array
    time = df['timestamp'].values - df['timestamp'].iloc[0]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # 3D orientation plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot orientation vectors at regular intervals
    step = max(1, len(df) // 50)  # Show max 50 orientation vectors
    for i in range(0, len(df), step):
        x_axis, y_axis, z_axis = create_orientation_vectors(roll[i], pitch[i], yaw[i])
        
        # Position in 3D space (using time as one dimension)
        pos = np.array([time[i], 0, 0])
        
        # Plot axes with different colors
        ax1.quiver(pos[0], pos[1], pos[2], x_axis[0], x_axis[1], x_axis[2], 
                  color='red', alpha=0.7, length=0.1, label='X-axis' if i == 0 else "")
        ax1.quiver(pos[0], pos[1], pos[2], y_axis[0], y_axis[1], y_axis[2], 
                  color='green', alpha=0.7, length=0.1, label='Y-axis' if i == 0 else "")
        ax1.quiver(pos[0], pos[1], pos[2], z_axis[0], z_axis[1], z_axis[2], 
                  color='blue', alpha=0.7, length=0.1, label='Z-axis' if i == 0 else "")
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Orientation Over Time')
    ax1.legend()
    
    # Roll over time
    ax2 = fig.add_subplot(222)
    ax2.plot(time, roll, 'r-', linewidth=2, label='Roll')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Roll (degrees)')
    ax2.set_title('Roll Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Pitch over time
    ax3 = fig.add_subplot(223)
    ax3.plot(time, pitch, 'g-', linewidth=2, label='Pitch')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Pitch (degrees)')
    ax3.set_title('Pitch Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Yaw over time
    ax4 = fig.add_subplot(224)
    ax4.plot(time, yaw, 'b-', linewidth=2, label='Yaw')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Yaw (degrees)')
    ax4.set_title('Yaw Over Time')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D orientation plot saved to: {save_path}")
    
    plt.show()
    
    return roll, pitch, yaw, time

def create_plotly_interactive_plot(df, save_path=None):
    """Create interactive 3D orientation plot using Plotly"""
    if df is None or df.empty:
        print("No pose data available for interactive 3D orientation plot")
        return
    
    # Convert quaternions to Euler angles (using pose orientation from single CSV)
    roll, pitch, yaw = quaternion_to_euler(
        df['pose_orientation_x'].values,
        df['pose_orientation_y'].values,
        df['pose_orientation_z'].values,
        df['pose_orientation_w'].values
    )
    
    # Create time array
    time = df['timestamp'].values - df['timestamp'].iloc[0]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('3D Orientation Over Time', 'Roll Over Time', 
                       'Pitch Over Time', 'Yaw Over Time'),
        specs=[[{"type": "scatter3d"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 3D orientation plot
    step = max(1, len(df) // 100)  # Show max 100 orientation vectors
    for i in range(0, len(df), step):
        x_axis, y_axis, z_axis = create_orientation_vectors(roll[i], pitch[i], yaw[i])
        
        # Position in 3D space
        pos = np.array([time[i], 0, 0])
        
        # Add orientation vectors
        fig.add_trace(go.Scatter3d(
            x=[pos[0], pos[0] + x_axis[0] * 0.1],
            y=[pos[1], pos[1] + x_axis[1] * 0.1],
            z=[pos[2], pos[2] + x_axis[2] * 0.1],
            mode='lines',
            line=dict(color='red', width=3),
            name='X-axis' if i == 0 else None,
            showlegend=(i == 0)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0], pos[0] + y_axis[0] * 0.1],
            y=[pos[1], pos[1] + y_axis[1] * 0.1],
            z=[pos[2], pos[2] + y_axis[2] * 0.1],
            mode='lines',
            line=dict(color='green', width=3),
            name='Y-axis' if i == 0 else None,
            showlegend=(i == 0)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter3d(
            x=[pos[0], pos[0] + z_axis[0] * 0.1],
            y=[pos[1], pos[1] + z_axis[1] * 0.1],
            z=[pos[2], pos[2] + z_axis[2] * 0.1],
            mode='lines',
            line=dict(color='blue', width=3),
            name='Z-axis' if i == 0 else None,
            showlegend=(i == 0)
        ), row=1, col=1)
    
    # Roll over time
    fig.add_trace(go.Scatter(
        x=time, y=roll,
        mode='lines',
        name='Roll',
        line=dict(color='red', width=2)
    ), row=1, col=2)
    
    # Pitch over time
    fig.add_trace(go.Scatter(
        x=time, y=pitch,
        mode='lines',
        name='Pitch',
        line=dict(color='green', width=2)
    ), row=2, col=1)
    
    # Yaw over time
    fig.add_trace(go.Scatter(
        x=time, y=yaw,
        mode='lines',
        name='Yaw',
        line=dict(color='blue', width=2)
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title="Interactive 3D Orientation Analysis - Olive Robotics",
        height=800,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Roll (degrees)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Pitch (degrees)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Yaw (degrees)", row=2, col=2)
    
    # Update 3D scene
    fig.update_scenes(
        xaxis_title="Time (s)",
        yaxis_title="Y",
        zaxis_title="Z",
        row=1, col=1
    )
    
    if save_path:
        fig.write_html(save_path)
        print(f"Interactive 3D orientation plot saved to: {save_path}")
    
    # Show the plot
    fig.show()
    
    return roll, pitch, yaw, time

def find_latest_data_directory():
    """Find the most recent data directory"""
    data_dirs = glob.glob("olive_data_*")
    if not data_dirs:
        raise FileNotFoundError("No data directories found. Run the data logger first.")
    return max(data_dirs, key=os.path.getctime)

def main():
    """Main function to create 3D orientation plots"""
    try:
        # Find the latest data directory
        data_dir = find_latest_data_directory()
        print(f"Loading data from: {data_dir}")
        
        # Load unified CSV file
        csv_file = os.path.join(data_dir, 'all_imu_data.csv')
        if not os.path.exists(csv_file):
            print("No unified CSV data found!")
            return
        
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} records")
        
        # Create plots directory
        plots_dir = os.path.join(data_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Convert quaternions to Euler angles for statistics
        roll, pitch, yaw = quaternion_to_euler(
            df['pose_orientation_x'].values,
            df['pose_orientation_y'].values,
            df['pose_orientation_z'].values,
            df['pose_orientation_w'].values
        )
        time = df['timestamp'].values - df['timestamp'].iloc[0]
        
        if MATPLOTLIB_AVAILABLE:
            print("\nCreating matplotlib 3D orientation plot...")
            create_matplotlib_3d_plot(
                df, 
                os.path.join(plots_dir, '3d_orientation_matplotlib.png')
            )
        
        print("\nCreating interactive Plotly 3D orientation plot...")
        create_plotly_interactive_plot(
            df, 
            os.path.join(plots_dir, '3d_orientation_interactive.html')
        )
        
        # Print orientation statistics
        print("\n" + "="*60)
        print("ORIENTATION ANALYSIS SUMMARY")
        print("="*60)
        print(f"Duration: {time[-1]:.2f} seconds")
        print(f"Data Points: {len(df)}")
        print()
        print("Roll (degrees):")
        print(f"  Range: {roll.min():.2f} to {roll.max():.2f}")
        print(f"  Mean: {roll.mean():.2f} ± {roll.std():.2f}")
        print()
        print("Pitch (degrees):")
        print(f"  Range: {pitch.min():.2f} to {pitch.max():.2f}")
        print(f"  Mean: {pitch.mean():.2f} ± {pitch.std():.2f}")
        print()
        print("Yaw (degrees):")
        print(f"  Range: {yaw.min():.2f} to {yaw.max():.2f}")
        print(f"  Mean: {yaw.mean():.2f} ± {yaw.std():.2f}")
        print("="*60)
        
        print(f"\nPlots saved to: {plots_dir}")
        print("Open the HTML file in a web browser for interactive 3D visualization!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

