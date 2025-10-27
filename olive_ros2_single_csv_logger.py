#!/usr/bin/env python3
"""
Olive Robotics ROS2 Data Logger - Single CSV Version
This script subscribes to all Olive Robotics topics and saves all data 
to a single unified CSV file for analysis and plotting.
"""

import rclpy
from rclpy.node import Node
import csv
import os
import time
from datetime import datetime
import threading
import signal
import sys

# Import ROS2 message types
from sensor_msgs.msg import Imu
from geometry_msgs.msg import AccelStamped, PoseStamped, TwistStamped
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

class OliveSingleCSVLogger(Node):
    def __init__(self):
        super().__init__('olive_single_csv_logger')
        
        # Create data directory
        self.data_dir = f"olive_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Single CSV file setup
        self.csv_filepath = os.path.join(self.data_dir, 'all_imu_data.csv')
        self.csv_file = None
        self.csv_writer = None
        self.lock = threading.Lock()
        
        # Latest values from each topic
        self.latest_values = {
            'imu_orientation_x': 0.0,
            'imu_orientation_y': 0.0,
            'imu_orientation_z': 0.0,
            'imu_orientation_w': 0.0,
            'imu_angular_velocity_x': 0.0,
            'imu_angular_velocity_y': 0.0,
            'imu_angular_velocity_z': 0.0,
            'imu_linear_acceleration_x': 0.0,
            'imu_linear_acceleration_y': 0.0,
            'imu_linear_acceleration_z': 0.0,
            'accel_x': 0.0,
            'accel_y': 0.0,
            'accel_z': 0.0,
            'pose_position_x': 0.0,
            'pose_position_y': 0.0,
            'pose_position_z': 0.0,
            'pose_orientation_x': 0.0,
            'pose_orientation_y': 0.0,
            'pose_orientation_z': 0.0,
            'pose_orientation_w': 0.0,
            'velocity_linear_x': 0.0,
            'velocity_linear_y': 0.0,
            'velocity_linear_z': 0.0,
            'velocity_angular_x': 0.0,
            'velocity_angular_y': 0.0,
            'velocity_angular_z': 0.0,
            'temperature': 0.0,
            'status': ''
        }
        
        # Pose setting functionality
        self.set_pose_client = self.create_client(Trigger, '/olive/olixSense/x1/id001/setPose')
        
        # Data collection control
        self.data_collection_started = False
        self.zero_position_reached = False
        self.start_time = time.time()
        self.max_wait_time = 30.0  # Maximum time to wait for zero position (30 seconds)
        
        # Initialize CSV file
        self.setup_csv_file()
        
        # Create subscribers
        self.imu_sub = self.create_subscription(
            Imu,
            '/olive/olixSense/x1/id001/imu',
            self.imu_callback,
            10
        )
        
        self.accel_sub = self.create_subscription(
            AccelStamped,
            '/olive/olixSense/x1/id001/acceleration',
            self.accel_callback,
            10
        )
        
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/olive/olixSense/x1/id001/pose',
            self.pose_callback,
            10
        )
        
        self.velocity_sub = self.create_subscription(
            TwistStamped,
            '/olive/olixSense/x1/id001/velocity',
            self.velocity_callback,
            10
        )
        
        self.temperature_sub = self.create_subscription(
            Float64,
            '/olive/olixSense/x1/id001/temperature',
            self.temperature_callback,
            10
        )
        
        self.status_sub = self.create_subscription(
            String,
            '/olive/olixSense/x1/id001/status',
            self.status_callback,
            10
        )
        
        self.get_logger().info('Subscribed to all topics')
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Call setPose service to reset current position to zero
        self.call_set_pose_service()
        
        self.get_logger().info('Olive Robotics Single CSV Data Logger started')
        self.get_logger().info(f'Data will be saved to: {self.csv_filepath}')
    
    def setup_csv_file(self):
        """Initialize the single CSV file and writer"""
        # Define headers for all data fields
        headers = [
            'timestamp',
            # IMU data
            'imu_orientation_x', 'imu_orientation_y', 'imu_orientation_z', 'imu_orientation_w',
            'imu_angular_velocity_x', 'imu_angular_velocity_y', 'imu_angular_velocity_z',
            'imu_linear_acceleration_x', 'imu_linear_acceleration_y', 'imu_linear_acceleration_z',
            # Acceleration data
            'accel_x', 'accel_y', 'accel_z',
            # Pose data
            'pose_position_x', 'pose_position_y', 'pose_position_z',
            'pose_orientation_x', 'pose_orientation_y', 'pose_orientation_z', 'pose_orientation_w',
            # Velocity data
            'velocity_linear_x', 'velocity_linear_y', 'velocity_linear_z',
            'velocity_angular_x', 'velocity_angular_y', 'velocity_angular_z',
            # Temperature and status
            'temperature', 'status'
        ]
        
        self.csv_file = open(self.csv_filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(headers)
        self.csv_file.flush()
    
    def get_timestamp(self):
        """Get current timestamp in seconds"""
        return time.time()
    
    def write_data_row(self):
        """Write a row with all current latest values"""
        if self.data_collection_started:
            timestamp = self.get_timestamp()
            row = [
                timestamp,
                # IMU data
                self.latest_values['imu_orientation_x'],
                self.latest_values['imu_orientation_y'],
                self.latest_values['imu_orientation_z'],
                self.latest_values['imu_orientation_w'],
                self.latest_values['imu_angular_velocity_x'],
                self.latest_values['imu_angular_velocity_y'],
                self.latest_values['imu_angular_velocity_z'],
                self.latest_values['imu_linear_acceleration_x'],
                self.latest_values['imu_linear_acceleration_y'],
                self.latest_values['imu_linear_acceleration_z'],
                # Acceleration data
                self.latest_values['accel_x'],
                self.latest_values['accel_y'],
                self.latest_values['accel_z'],
                # Pose data
                self.latest_values['pose_position_x'],
                self.latest_values['pose_position_y'],
                self.latest_values['pose_position_z'],
                self.latest_values['pose_orientation_x'],
                self.latest_values['pose_orientation_y'],
                self.latest_values['pose_orientation_z'],
                self.latest_values['pose_orientation_w'],
                # Velocity data
                self.latest_values['velocity_linear_x'],
                self.latest_values['velocity_linear_y'],
                self.latest_values['velocity_linear_z'],
                self.latest_values['velocity_angular_x'],
                self.latest_values['velocity_angular_y'],
                self.latest_values['velocity_angular_z'],
                # Temperature and status
                self.latest_values['temperature'],
                self.latest_values['status']
            ]
            self.csv_writer.writerow(row)
            self.csv_file.flush()
    
    def imu_callback(self, msg):
        """Callback for IMU data"""
        with self.lock:
            # Update latest IMU values
            self.latest_values['imu_orientation_x'] = msg.orientation.x
            self.latest_values['imu_orientation_y'] = msg.orientation.y
            self.latest_values['imu_orientation_z'] = msg.orientation.z
            self.latest_values['imu_orientation_w'] = msg.orientation.w
            self.latest_values['imu_angular_velocity_x'] = msg.angular_velocity.x
            self.latest_values['imu_angular_velocity_y'] = msg.angular_velocity.y
            self.latest_values['imu_angular_velocity_z'] = msg.angular_velocity.z
            self.latest_values['imu_linear_acceleration_x'] = msg.linear_acceleration.x
            self.latest_values['imu_linear_acceleration_y'] = msg.linear_acceleration.y
            self.latest_values['imu_linear_acceleration_z'] = msg.linear_acceleration.z
            
            # Write a row with all current values
            self.write_data_row()
    
    def accel_callback(self, msg):
        """Callback for acceleration data"""
        with self.lock:
            # Update latest acceleration values
            self.latest_values['accel_x'] = msg.accel.linear.x
            self.latest_values['accel_y'] = msg.accel.linear.y
            self.latest_values['accel_z'] = msg.accel.linear.z
            
            # Write a row with all current values
            self.write_data_row()
    
    def pose_callback(self, msg):
        """Callback for pose data"""
        with self.lock:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Check if position is at zero (within small tolerance)
            tolerance = 0.01  # 1cm tolerance
            is_at_zero = (abs(msg.pose.position.x) < tolerance and 
                         abs(msg.pose.position.y) < tolerance and 
                         abs(msg.pose.position.z) < tolerance)
            
            # If we haven't started data collection and position is at zero, start collecting
            if not self.data_collection_started and is_at_zero:
                self.data_collection_started = True
                self.zero_position_reached = True
                self.get_logger().info('Position reached zero - starting data collection')
            
            # If we haven't started data collection and timeout reached, start anyway
            elif not self.data_collection_started and elapsed_time > self.max_wait_time:
                self.data_collection_started = True
                self.get_logger().warn(f'Timeout reached ({self.max_wait_time}s) - starting data collection anyway')
                self.get_logger().info(f'Current position: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}')
            
            # Update latest pose values
            self.latest_values['pose_position_x'] = msg.pose.position.x
            self.latest_values['pose_position_y'] = msg.pose.position.y
            self.latest_values['pose_position_z'] = msg.pose.position.z
            self.latest_values['pose_orientation_x'] = msg.pose.orientation.x
            self.latest_values['pose_orientation_y'] = msg.pose.orientation.y
            self.latest_values['pose_orientation_z'] = msg.pose.orientation.z
            self.latest_values['pose_orientation_w'] = msg.pose.orientation.w
            
            # Write a row with all current values
            self.write_data_row()
    
    def velocity_callback(self, msg):
        """Callback for velocity data"""
        with self.lock:
            # Update latest velocity values
            self.latest_values['velocity_linear_x'] = msg.twist.linear.x
            self.latest_values['velocity_linear_y'] = msg.twist.linear.y
            self.latest_values['velocity_linear_z'] = msg.twist.linear.z
            self.latest_values['velocity_angular_x'] = msg.twist.angular.x
            self.latest_values['velocity_angular_y'] = msg.twist.angular.y
            self.latest_values['velocity_angular_z'] = msg.twist.angular.z
            
            # Write a row with all current values
            self.write_data_row()
    
    def temperature_callback(self, msg):
        """Callback for temperature data"""
        with self.lock:
            # Update latest temperature value
            self.latest_values['temperature'] = msg.data
            
            # Write a row with all current values
            self.write_data_row()
    
    def status_callback(self, msg):
        """Callback for status data"""
        with self.lock:
            # Update latest status value
            self.latest_values['status'] = msg.data
            
            # Write a row with all current values
            self.write_data_row()
    
    def call_set_pose_service(self):
        """
        Call the setPose service to reset current position to zero
        """
        try:
            if not self.set_pose_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error('setPose service not available')
                return False
            
            request = Trigger.Request()
            future = self.set_pose_client.call_async(request)
            
            # Wait for the response
            rclpy.spin_until_future_complete(self, future)
            
            if future.result() is not None:
                response = future.result()
                if response.success:
                    self.get_logger().info('setPose service called successfully - position reset to zero')
                    return True
                else:
                    self.get_logger().error(f'setPose service failed: {response.message}')
                    return False
            else:
                self.get_logger().error('setPose service call failed')
                return False
                
        except Exception as e:
            self.get_logger().error(f'Error calling setPose service: {str(e)}')
            return False
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.get_logger().info('Shutting down data logger...')
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Close CSV file"""
        if self.csv_file:
            self.csv_file.close()
        self.get_logger().info(f'Data saved to {self.csv_filepath}')

def main(args=None):
    rclpy.init(args=args)
    
    try:
        logger = OliveSingleCSVLogger()
        rclpy.spin(logger)
    except KeyboardInterrupt:
        logger.get_logger().info('Data logger interrupted by user')
    except Exception as e:
        logger.get_logger().error(f'Error: {str(e)}')
    finally:
        if 'logger' in locals():
            logger.cleanup()
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()

