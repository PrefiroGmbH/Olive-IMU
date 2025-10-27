#!/usr/bin/env python3
"""
Olive Robotics Data Plotting Script - Single CSV Version
This script reads the single unified CSV data file and creates various plots
for analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import glob
from datetime import datetime

def find_latest_data_directory():
    """Find the most recent data directory"""
    data_dirs = glob.glob("olive_data_*")
    if not data_dirs:
        raise FileNotFoundError("No data directories found. Run the data logger first.")
    return max(data_dirs, key=os.path.getctime)

def load_data(data_dir):
    """Load the single unified CSV file from the data directory"""
    # Look for the unified CSV file
    csv_file = os.path.join(data_dir, 'all_imu_data.csv')
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"Loaded unified data: {len(df)} records")
    print(f"Columns: {', '.join(df.columns)}")
    
    return df

def plot_imu_data(df, save_path=None):
    """Plot IMU data (orientation, angular velocity, linear acceleration)"""
    if df is None or df.empty:
        print("No IMU data available")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('IMU Data Analysis', fontsize=16)
    
    # Plot orientation (quaternion)
    axes[0].plot(df['timestamp'], df['imu_orientation_x'], label='X', alpha=0.7)
    axes[0].plot(df['timestamp'], df['imu_orientation_y'], label='Y', alpha=0.7)
    axes[0].plot(df['timestamp'], df['imu_orientation_z'], label='Z', alpha=0.7)
    axes[0].plot(df['timestamp'], df['imu_orientation_w'], label='W', alpha=0.7)
    axes[0].set_title('Orientation (Quaternion)')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot angular velocity
    axes[1].plot(df['timestamp'], df['imu_angular_velocity_x'], label='X', alpha=0.7)
    axes[1].plot(df['timestamp'], df['imu_angular_velocity_y'], label='Y', alpha=0.7)
    axes[1].plot(df['timestamp'], df['imu_angular_velocity_z'], label='Z', alpha=0.7)
    axes[1].set_title('Angular Velocity (rad/s)')
    axes[1].set_ylabel('rad/s')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot linear acceleration
    axes[2].plot(df['timestamp'], df['imu_linear_acceleration_x'], label='X', alpha=0.7)
    axes[2].plot(df['timestamp'], df['imu_linear_acceleration_y'], label='Y', alpha=0.7)
    axes[2].plot(df['timestamp'], df['imu_linear_acceleration_z'], label='Z', alpha=0.7)
    axes[2].set_title('Linear Acceleration (m/s²)')
    axes[2].set_ylabel('m/s²')
    axes[2].set_xlabel('Timestamp')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_acceleration_data(df, save_path=None):
    """Plot acceleration data"""
    if df is None or df.empty:
        print("No acceleration data available")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(df['timestamp'], df['accel_x'], label='X', alpha=0.7)
    ax.plot(df['timestamp'], df['accel_y'], label='Y', alpha=0.7)
    ax.plot(df['timestamp'], df['accel_z'], label='Z', alpha=0.7)
    ax.set_title('Acceleration Data')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_xlabel('Timestamp')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_pose_data(df, save_path=None):
    """Plot pose data (position and orientation)"""
    if df is None or df.empty:
        print("No pose data available")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Pose Data Analysis', fontsize=16)
    
    # Plot position
    axes[0].plot(df['timestamp'], df['pose_position_x'], label='X', alpha=0.7)
    axes[0].plot(df['timestamp'], df['pose_position_y'], label='Y', alpha=0.7)
    axes[0].plot(df['timestamp'], df['pose_position_z'], label='Z', alpha=0.7)
    axes[0].set_title('Position')
    axes[0].set_ylabel('Position (m)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot orientation
    axes[1].plot(df['timestamp'], df['pose_orientation_x'], label='X', alpha=0.7)
    axes[1].plot(df['timestamp'], df['pose_orientation_y'], label='Y', alpha=0.7)
    axes[1].plot(df['timestamp'], df['pose_orientation_z'], label='Z', alpha=0.7)
    axes[1].plot(df['timestamp'], df['pose_orientation_w'], label='W', alpha=0.7)
    axes[1].set_title('Orientation (Quaternion)')
    axes[1].set_ylabel('Value')
    axes[1].set_xlabel('Timestamp')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_velocity_data(df, save_path=None):
    """Plot velocity data"""
    if df is None or df.empty:
        print("No velocity data available")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Velocity Data Analysis', fontsize=16)
    
    # Plot linear velocity
    axes[0].plot(df['timestamp'], df['velocity_linear_x'], label='X', alpha=0.7)
    axes[0].plot(df['timestamp'], df['velocity_linear_y'], label='Y', alpha=0.7)
    axes[0].plot(df['timestamp'], df['velocity_linear_z'], label='Z', alpha=0.7)
    axes[0].set_title('Linear Velocity')
    axes[0].set_ylabel('Velocity (m/s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot angular velocity
    axes[1].plot(df['timestamp'], df['velocity_angular_x'], label='X', alpha=0.7)
    axes[1].plot(df['timestamp'], df['velocity_angular_y'], label='Y', alpha=0.7)
    axes[1].plot(df['timestamp'], df['velocity_angular_z'], label='Z', alpha=0.7)
    axes[1].set_title('Angular Velocity')
    axes[1].set_ylabel('Angular Velocity (rad/s)')
    axes[1].set_xlabel('Timestamp')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_temperature_data(df, save_path=None):
    """Plot temperature data"""
    if df is None or df.empty:
        print("No temperature data available")
        return
    
    # Check if temperature data has any non-zero values
    if 'temperature' not in df.columns or df['temperature'].sum() == 0:
        print("No temperature data recorded")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(df['timestamp'], df['temperature'], 'r-', alpha=0.7)
    ax.set_title('Temperature Data')
    ax.set_ylabel('Temperature (°C)')
    ax.set_xlabel('Timestamp')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def calculate_distances(df):
    """Calculate total distances travelled in 3D and 2D space"""
    if df is None or df.empty:
        return None
    
    # Calculate 3D distance
    df_3d = df[['pose_position_x', 'pose_position_y', 'pose_position_z']].diff()
    df_3d = df_3d.dropna()
    distance_3d = np.sqrt((df_3d['pose_position_x']**2 + 
                           df_3d['pose_position_y']**2 + 
                           df_3d['pose_position_z']**2)).sum()
    
    # Calculate 2D XY distance
    df_xy = df[['pose_position_x', 'pose_position_y']].diff()
    df_xy = df_xy.dropna()
    distance_xy = np.sqrt((df_xy['pose_position_x']**2 + df_xy['pose_position_y']**2)).sum()
    
    # Calculate 2D XZ distance
    df_xz = df[['pose_position_x', 'pose_position_z']].diff()
    df_xz = df_xz.dropna()
    distance_xz = np.sqrt((df_xz['pose_position_x']**2 + df_xz['pose_position_z']**2)).sum()
    
    # Calculate 2D YZ distance
    df_yz = df[['pose_position_y', 'pose_position_z']].diff()
    df_yz = df_yz.dropna()
    distance_yz = np.sqrt((df_yz['pose_position_y']**2 + df_yz['pose_position_z']**2)).sum()
    
    # Calculate straight-line distances
    start_pos = df.iloc[0]
    end_pos = df.iloc[-1]
    
    straight_3d = np.sqrt((end_pos['pose_position_x'] - start_pos['pose_position_x'])**2 + 
                         (end_pos['pose_position_y'] - start_pos['pose_position_y'])**2 + 
                         (end_pos['pose_position_z'] - start_pos['pose_position_z'])**2)
    
    straight_xy = np.sqrt((end_pos['pose_position_x'] - start_pos['pose_position_x'])**2 + 
                         (end_pos['pose_position_y'] - start_pos['pose_position_y'])**2)
    
    return {
        'total_3d': distance_3d,
        'total_xy': distance_xy,
        'total_xz': distance_xz,
        'total_yz': distance_yz,
        'straight_3d': straight_3d,
        'straight_xy': straight_xy
    }

def create_3d_trajectory_plot(df, save_path=None):
    """Create a 3D trajectory plot from pose data with distance calculations"""
    if df is None or df.empty:
        print("No pose data available for 3D plot")
        return
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Calculate distances
        distances = calculate_distances(df)
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(df['pose_position_x'], df['pose_position_y'], df['pose_position_z'], 
                'b-', alpha=0.7, linewidth=2)
        ax.scatter(df['pose_position_x'].iloc[0], df['pose_position_y'].iloc[0], 
                   df['pose_position_z'].iloc[0], color='green', s=100, label='Start')
        ax.scatter(df['pose_position_x'].iloc[-1], df['pose_position_y'].iloc[-1], 
                   df['pose_position_z'].iloc[-1], color='red', s=100, label='End')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(f'3D Trajectory\nTotal Distance: {distances["total_3d"]:.3f}m | Straight Distance: {distances["straight_3d"]:.3f}m')
        ax.legend()
        
        # Add distance information as text
        info_text = f"""Distance Summary:
3D Total: {distances['total_3d']:.3f}m
3D Straight: {distances['straight_3d']:.3f}m
Efficiency: {((distances['straight_3d'] / distances['total_3d']) * 100):.1f}%"""
        
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                 fontsize=9, verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return distances
    except ImportError:
        print("3D plotting not available. Creating 2D trajectory plot instead.")
        return create_2d_trajectory_plot(df, save_path)

def create_2d_trajectory_plot(df, save_path=None):
    """Create a 2D trajectory plot from pose data with distance calculations"""
    if df is None or df.empty:
        print("No pose data available for 2D plot")
        return
    
    # Calculate distances
    distances = calculate_distances(df)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('2D Trajectory Analysis with Distance Calculations', fontsize=16)
    
    # XY trajectory
    axes[0,0].plot(df['pose_position_x'], df['pose_position_y'], 'b-', alpha=0.7, linewidth=2)
    axes[0,0].scatter(df['pose_position_x'].iloc[0], df['pose_position_y'].iloc[0], 
                     color='green', s=100, label='Start')
    axes[0,0].scatter(df['pose_position_x'].iloc[-1], df['pose_position_y'].iloc[-1], 
                     color='red', s=100, label='End')
    axes[0,0].set_xlabel('X Position (m)')
    axes[0,0].set_ylabel('Y Position (m)')
    axes[0,0].set_title(f'XY Trajectory\nTotal Distance: {distances["total_xy"]:.3f}m\nStraight Distance: {distances["straight_xy"]:.3f}m')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # XZ trajectory
    axes[0,1].plot(df['pose_position_x'], df['pose_position_z'], 'b-', alpha=0.7, linewidth=2)
    axes[0,1].scatter(df['pose_position_x'].iloc[0], df['pose_position_z'].iloc[0], 
                     color='green', s=100, label='Start')
    axes[0,1].scatter(df['pose_position_x'].iloc[-1], df['pose_position_z'].iloc[-1], 
                     color='red', s=100, label='End')
    axes[0,1].set_xlabel('X Position (m)')
    axes[0,1].set_ylabel('Z Position (m)')
    axes[0,1].set_title(f'XZ Trajectory\nTotal Distance: {distances["total_xz"]:.3f}m')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # YZ trajectory
    axes[1,0].plot(df['pose_position_y'], df['pose_position_z'], 'b-', alpha=0.7, linewidth=2)
    axes[1,0].scatter(df['pose_position_y'].iloc[0], df['pose_position_z'].iloc[0], 
                     color='green', s=100, label='Start')
    axes[1,0].scatter(df['pose_position_y'].iloc[-1], df['pose_position_z'].iloc[-1], 
                     color='red', s=100, label='End')
    axes[1,0].set_xlabel('Y Position (m)')
    axes[1,0].set_ylabel('Z Position (m)')
    axes[1,0].set_title(f'YZ Trajectory\nTotal Distance: {distances["total_yz"]:.3f}m')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Distance summary
    axes[1,1].axis('off')
    distance_text = f"""Distance Summary:
    
3D Total Distance: {distances['total_3d']:.3f} m
3D Straight Distance: {distances['straight_3d']:.3f} m

2D XY Total Distance: {distances['total_xy']:.3f} m
2D XY Straight Distance: {distances['straight_xy']:.3f} m

2D XZ Total Distance: {distances['total_xz']:.3f} m
2D YZ Total Distance: {distances['total_yz']:.3f} m

Path Efficiency (3D):
{((distances['straight_3d'] / distances['total_3d']) * 100):.1f}%

Path Efficiency (XY):
{((distances['straight_xy'] / distances['total_xy']) * 100):.1f}%"""
    
    axes[1,1].text(0.1, 0.9, distance_text, transform=axes[1,1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return distances

def main():
    """Main function to load data and create plots"""
    try:
        # Find the latest data directory
        data_dir = find_latest_data_directory()
        print(f"Loading data from: {data_dir}")
        
        # Load unified data
        df = load_data(data_dir)
        
        if df is None or df.empty:
            print("No data found!")
            return
        
        # Create plots directory
        plots_dir = os.path.join(data_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate plots
        print("\nGenerating plots...")
        
        # IMU plots
        plot_imu_data(df, os.path.join(plots_dir, 'imu_analysis.png'))
        
        # Acceleration plots
        plot_acceleration_data(df, os.path.join(plots_dir, 'acceleration_analysis.png'))
        
        # Pose plots
        plot_pose_data(df, os.path.join(plots_dir, 'pose_analysis.png'))
        
        # 3D trajectory with distances
        distances = create_3d_trajectory_plot(df, os.path.join(plots_dir, 'trajectory_3d_plot.png'))
        
        # Print distance summary to console
        if distances:
            print("\n" + "="*60)
            print("DISTANCE ANALYSIS SUMMARY")
            print("="*60)
            print(f"3D Total Distance Travelled: {distances['total_3d']:.3f} m")
            print(f"3D Straight-line Distance:   {distances['straight_3d']:.3f} m")
            print(f"3D Path Efficiency:          {((distances['straight_3d'] / distances['total_3d']) * 100):.1f}%")
            print()
            print(f"2D XY Total Distance:        {distances['total_xy']:.3f} m")
            print(f"2D XY Straight Distance:     {distances['straight_xy']:.3f} m")
            print(f"2D XY Path Efficiency:       {((distances['straight_xy'] / distances['total_xy']) * 100):.1f}%")
            print()
            print(f"2D XZ Total Distance:        {distances['total_xz']:.3f} m")
            print(f"2D YZ Total Distance:        {distances['total_yz']:.3f} m")
            print("="*60)
        
        # Velocity plots
        plot_velocity_data(df, os.path.join(plots_dir, 'velocity_analysis.png'))
        
        # Temperature plots (if available)
        plot_temperature_data(df, os.path.join(plots_dir, 'temperature_analysis.png'))
        
        print(f"\nPlots saved to: {plots_dir}")
        print("Data analysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

