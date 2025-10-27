# Single CSV Data Logger - Usage Guide

This guide covers the new single CSV data logger and its associated plotting scripts.

## Data Collection

### Script: `olive_ros2_single_csv_logger.py`

Collects all IMU sensor data into a single unified CSV file.

**Usage:**
```bash
python3 olive_ros2_single_csv_logger.py
# or
./olive_ros2_single_csv_logger.py
```

**Features:**
- Calls `setPose` service at startup to reset position to zero
- Waits for position to reach zero (1cm tolerance) before collecting data
- 30-second timeout - starts collecting anyway if zero isn't reached
- All data from all topics stored in one file: `all_imu_data.csv`
- Data includes:
  - IMU orientation (quaternion: x, y, z, w)
  - IMU angular velocity (x, y, z)
  - IMU linear acceleration (x, y, z)
  - Acceleration (x, y, z)
  - Pose position (x, y, z)
  - Pose orientation (quaternion: x, y, z, w)
  - Velocity linear (x, y, z)
  - Velocity angular (x, y, z)
  - Temperature
  - Status

**Output:**
- Creates timestamped directory: `olive_data_YYYYMMDD_HHMMSS/`
- Single CSV file: `all_imu_data.csv`

## Data Visualization

### 1. Main Plotting Script: `plot_single_csv_data.py`

Comprehensive plotting script that generates all standard plots and distance calculations.

**Usage:**
```bash
python3 plot_single_csv_data.py
# or
./plot_single_csv_data.py
```

**Generates:**
- IMU analysis plot (orientation, angular velocity, linear acceleration)
- Acceleration analysis plot
- Pose analysis plot (position and orientation)
- 3D trajectory plot with distance calculations
- 2D trajectory plots (XY, XZ, YZ projections)
- Velocity analysis plot (linear and angular)
- Temperature plot (if available)

**Output:**
- All plots saved to: `olive_data_YYYYMMDD_HHMMSS/plots/`
- Distance summary printed to console

**Distance Calculations:**
- 3D total distance travelled
- 3D straight-line distance (start to end)
- Path efficiency (straight/total * 100%)
- 2D projections (XY, XZ, YZ)

### 2. Interactive 3D Pose Plot: `plot_3d_pose_single_csv.py`

Creates an interactive 3D trajectory visualization using Plotly.

**Usage:**
```bash
python3 plot_3d_pose_single_csv.py
# or
./plot_3d_pose_single_csv.py
```

**Features:**
- Interactive 3D plot (rotate, zoom, pan)
- Hover to see exact coordinates
- Start and end points marked
- Distance statistics displayed
- Opens in web browser

**Output:**
- `plots/3d_trajectory_interactive.html`

### 3. Interactive 3D Orientation Plot: `plot_3d_orientation_single_csv.py`

Visualizes orientation (roll, pitch, yaw) over time with 3D representation.

**Usage:**
```bash
python3 plot_3d_orientation_single_csv.py
# or
./plot_3d_orientation_single_csv.py
```

**Features:**
- Converts quaternions to Euler angles (roll, pitch, yaw)
- 3D orientation vectors visualization
- Individual plots for roll, pitch, and yaw over time
- Interactive Plotly visualization
- Statistical analysis (range, mean, std dev)

**Output:**
- `plots/3d_orientation_interactive.html` (interactive)
- `plots/3d_orientation_matplotlib.png` (static, if matplotlib available)
- Orientation statistics printed to console

## CSV File Format

The unified CSV file contains all data with the following columns:

```
timestamp                    - Unix timestamp
imu_orientation_x/y/z/w     - IMU orientation quaternion
imu_angular_velocity_x/y/z  - IMU angular velocity (rad/s)
imu_linear_acceleration_x/y/z - IMU linear acceleration (m/s²)
accel_x/y/z                  - Acceleration data (m/s²)
pose_position_x/y/z          - Position (meters)
pose_orientation_x/y/z/w     - Pose orientation quaternion
velocity_linear_x/y/z        - Linear velocity (m/s)
velocity_angular_x/y/z       - Angular velocity (rad/s)
temperature                  - Temperature (°C)
status                       - Status string
```

## Quick Start Workflow

1. **Collect Data:**
   ```bash
   ./olive_ros2_single_csv_logger.py
   ```
   Press Ctrl+C when done.

2. **Generate All Plots:**
   ```bash
   ./plot_single_csv_data.py
   ```

3. **View Interactive 3D Trajectory:**
   ```bash
   ./plot_3d_pose_single_csv.py
   ```
   Open the generated HTML file in your browser.

4. **Analyze Orientation:**
   ```bash
   ./plot_3d_orientation_single_csv.py
   ```

## Advantages of Single CSV Format

- **Simplicity:** One file instead of 6 separate files
- **Time-synchronized:** All data aligned by timestamp
- **Easier analysis:** Can correlate data across all sensors
- **Reduced file management:** Simpler directory structure
- **Better for ML/analysis:** Direct input to pandas/numpy

## Example Output

```
DISTANCE ANALYSIS SUMMARY
============================================================
3D Total Distance Travelled: 0.380 m
3D Straight-line Distance:   0.321 m
3D Path Efficiency:          84.6%

2D XY Total Distance:        0.380 m
2D XY Straight Distance:     0.321 m
2D XY Path Efficiency:       84.6%

2D XZ Total Distance:        0.015 m
2D YZ Total Distance:        0.375 m
============================================================
```

## Dependencies

Required Python packages (see `requirements.txt`):
- rclpy (ROS2)
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scipy (for orientation conversion)

## Notes

- All scripts automatically find the most recent data directory
- Plots are saved with high resolution (300 DPI for PNG)
- Interactive HTML plots can be shared and viewed in any browser
- The logger uses threading locks for thread-safe data writing

