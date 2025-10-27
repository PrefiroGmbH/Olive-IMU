#!/usr/bin/env python3
"""
3D Pose Visualization for Olive Robotics Data - Single CSV Version
Creates an interactive 3D plot of the robot's trajectory using Plotly
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import glob

def create_3d_pose_plot(csv_file, save_path=None):
    """Create a 3D plot of the robot's pose trajectory using Plotly"""
    
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Extract position data (already in meters from single CSV)
    x = df['pose_position_x']
    y = df['pose_position_y']
    z = df['pose_position_z']
    
    # Calculate trajectory statistics
    total_distance = np.sqrt(np.sum(np.diff(x)**2 + np.diff(y)**2 + np.diff(z)**2))
    straight_distance = np.sqrt((x.iloc[-1] - x.iloc[0])**2 + 
                                (y.iloc[-1] - y.iloc[0])**2 + 
                                (z.iloc[-1] - z.iloc[0])**2)
    efficiency = (straight_distance / total_distance) * 100 if total_distance > 0 else 0
    
    # Create the 3D plot
    fig = go.Figure()
    
    # Add trajectory line
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines+markers',
        line=dict(color='blue', width=4),
        marker=dict(size=3, color='blue', opacity=0.6),
        name='Trajectory',
        hovertemplate='<b>Position</b><br>' +
                      'X: %{x:.4f} m<br>' +
                      'Y: %{y:.4f} m<br>' +
                      'Z: %{z:.4f} m<br>' +
                      '<extra></extra>'
    ))
    
    # Add start point
    fig.add_trace(go.Scatter3d(
        x=[x.iloc[0]], y=[y.iloc[0]], z=[z.iloc[0]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='circle'),
        name='Start',
        hovertemplate='<b>Start Point</b><br>' +
                      'X: %{x:.4f} m<br>' +
                      'Y: %{y:.4f} m<br>' +
                      'Z: %{z:.4f} m<br>' +
                      '<extra></extra>'
    ))
    
    # Add end point
    fig.add_trace(go.Scatter3d(
        x=[x.iloc[-1]], y=[y.iloc[-1]], z=[z.iloc[-1]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='square'),
        name='End',
        hovertemplate='<b>End Point</b><br>' +
                      'X: %{x:.4f} m<br>' +
                      'Y: %{y:.4f} m<br>' +
                      'Z: %{z:.4f} m<br>' +
                      '<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Olive Robotics 3D Trajectory<br><sub>Position data in meters</sub>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)',
            zaxis_title='Z Position (m)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=60)
    )
    
    # Add annotations with statistics
    fig.add_annotation(
        text=f"<b>Statistics:</b><br>" +
             f"Total Distance: {total_distance:.4f} m<br>" +
             f"Straight Distance: {straight_distance:.4f} m<br>" +
             f"Efficiency: {efficiency:.1f}%<br>" +
             f"Data Points: {len(df)}",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="black",
        borderwidth=1,
        align="left"
    )
    
    # Save the plot if path is provided
    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, width=1000, height=800, scale=2)
        print(f"3D plot saved to: {save_path}")
    
    return fig

def find_latest_data_directory():
    """Find the most recent data directory"""
    data_dirs = glob.glob("olive_data_*")
    if not data_dirs:
        raise FileNotFoundError("No data directories found. Run the data logger first.")
    return max(data_dirs, key=os.path.getctime)

def main():
    """Main function to create 3D pose plot"""
    try:
        # Find the latest data directory
        data_dir = find_latest_data_directory()
        print(f"Creating 3D plot from: {data_dir}")
        
        # Load unified CSV file
        csv_file = os.path.join(data_dir, 'all_imu_data.csv')
        if not os.path.exists(csv_file):
            print("No unified CSV data found!")
            return
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(data_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create 3D plot
        html_path = os.path.join(plots_dir, '3d_trajectory_interactive.html')
        fig = create_3d_pose_plot(csv_file, html_path)
        
        # Show the plot
        fig.show()
        
        print("\nOpen the HTML file in a web browser for interactive 3D visualization!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

