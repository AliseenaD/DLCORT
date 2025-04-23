import unittest
import pandas as pd
import numpy as np
from io import StringIO
import tkinter as tk
from unittest.mock import MagicMock
from objectLoc import RegionMarker
import cv2
import math

# Simplified version of InteractionAnalyzerUI for testing
class InteractionAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.fps = tk.StringVar(value="30")
        self.pixel_ratio = tk.StringVar(value="10")
        self.average_speed = 0
        self.speeds = []
        
    def calculate_average_speed__per_frame(self, csv_file):
        # Open the file and skip the first two rows
        df = pd.read_csv(csv_file, skiprows=2)
        # speeds = []
        first_loc = (0,0)
        second_loc = (0,0)
        
        # Iterate through every row, calculate speed
        for i in range(1, len(df)):
            prev_row = df.iloc[i-1]
            curr_row = df.iloc[i]

            # Get all the position coordinates
            first_rx, first_ry = (float(prev_row.iloc[4]), float(prev_row.iloc[5]))
            first_r_ear_prob = float(prev_row.iloc[6])
            first_lx, first_ly = (float(prev_row.iloc[7]), float(prev_row.iloc[8]))
            first_l_ear_prob = float(prev_row.iloc[9])
            second_rx, second_ry = (float(curr_row.iloc[4]), float(curr_row.iloc[5]))
            second_r_ear_prob = float(curr_row.iloc[6])
            second_lx, second_ly = (float(curr_row.iloc[7]), float(curr_row.iloc[8]))
            second_l_ear_prob = float(curr_row.iloc[9])
            
            # If either left or right ear prob less than threshold then continue
            probability_threshold = 0.6
            if first_l_ear_prob < probability_threshold or first_r_ear_prob < probability_threshold or second_l_ear_prob < probability_threshold or second_r_ear_prob < probability_threshold:
                continue
            
            # Calculate midpoints and find distance and speed
            first_mid_x = (first_lx + first_rx) / 2
            first_mid_y = (first_ly + first_ry) / 2
            first_loc = (first_mid_x, first_mid_y)
            second_mid_x = (second_lx + second_rx) / 2
            second_mid_y = (second_ly + second_ry) / 2
            second_loc = (second_mid_x, second_mid_y)
            distance = np.sqrt(((second_loc[0] - first_loc[0]) / float(self.pixel_ratio.get()))**2 + 
                             ((second_loc[1] - first_loc[1]) / float(self.pixel_ratio.get()))**2)


            print(f'Raw first right ear: ({first_rx},{first_ry})\n'+
                  f'Raw first left ear: ({first_lx},{first_ly})\n'+
                  f'First midpoint: {first_loc}\n'+
                  f'Raw second right ear: ({second_rx},{second_ry})\n'+
                  f'Raw second left ear: ({second_lx},{second_ly})\n'+
                  f'second midpoint: {second_loc}\n'+
                  f'Distance: {distance}')
            # Calculate speed and add to list if above threshold
            speed = distance * float(self.fps.get())
            print(f"Calculated speed: {speed}")
            if speed >= 3:
                self.speeds.append(speed)
                
        self.average_speed = np.mean(self.speeds) if self.speeds else 0
        print(f"Average speed: {self.average_speed}")


class TestSpeedCalculation(unittest.TestCase):
    def setUp(self):
        # Create a mock root window
        self.root = tk.Tk()
        # Initialize the UI with mock values
        self.analyzer = InteractionAnalyzerUI(self.root)
        # Set default values for fps and pixel ratio 
        self.analyzer.fps.set("30")  # 30 frames per second
        self.analyzer.pixel_ratio.set("10")  # 10 pixels per cm for easy math
        
    def create_test_csv(self, data):
        """Creates a CSV with proper DLC format"""
        # Create header rows matching the actual format
        header1 = "scorer,DLC_resnet50_ObjectTestNov21shuffle1_250000,DLC_resnet50_ObjectTestNov21shuffle1_250000.1,DLC_resnet50_ObjectTestNov21shuffle1_250000.2\n"
        header2 = "bodyparts,nose,nose,nose,leftear,leftear,leftear,rightear,rightear,rightear\n"
        header3 = "coords,x,y,likelihood,x,y,likelihood,x,y,likelihood\n"
        return StringIO(header1 + header2 + header3 + data)
    
    def test_basic_speed_calculation(self):
        """Test that speed is calculated correctly for a simple movement"""
        # Create test data where movement should result in 15 cm/sec
        # Moving 5 pixels in x direction (0.5 cm at 10 pixels/cm)
        # At 30 fps, this equals 15 cm/sec
        data = "0,100,100,0.9,105,100,0.9,95,100,0.9\n"  # First frame
        data += "1,100,100,0.9,110,100,0.9,100,100,0.9"  # Second frame - moved right ear 5 pixels
        
        csv_file = self.create_test_csv(data)
        self.analyzer.calculate_average_speed__per_frame(csv_file)
        self.assertEqual(round(self.analyzer.average_speed, 2), 15.00)
    
    def test_below_threshold_speed(self):
        """Test that speeds below 3 cm/sec are excluded"""
        # Create movement that results in 2 cm/sec (should be excluded)
        data = "0,100,100,0.9,102,100,0.9,98,100,0.9\n"  # First frame
        data += "0.5,100,100,0.9,102.5,100,0.9,98.5,100,0.9"   # Small movement (< 3 cm/sec)
        
        csv_file = self.create_test_csv(data)
        self.analyzer.calculate_average_speed__per_frame(csv_file)
        self.assertEqual(len(self.analyzer.speeds), 0)  # No speeds should be included
    
    def test_probability_threshold(self):
        """Test that frames with low probability are excluded"""
        # Create movement with low probability scores
        data = "0,100,100,0.5,105,100,0.9,95,100,0.9\n"  # Low nose probability
        data += "1,100,100,0.9,110,100,0.5,100,100,0.9"  # Low right ear probability
        
        csv_file = self.create_test_csv(data)
        self.analyzer.calculate_average_speed__per_frame(csv_file)
        self.assertEqual(len(self.analyzer.speeds), 0)  # No speeds should be included due to low probability
    
    def test_average_calculation(self):
        """Test that average is calculated correctly from multiple speeds"""
        # Create three movements: 15 cm/sec, 20 cm/sec, and 25 cm/sec
        data = "0,100,100,0.9,105,100,0.9,95,100,0.9\n"   # Start
        data += "1,100,100,0.9,110,100,0.9,100,100,0.9\n" # 15 cm/sec
        data += "2,100,100,0.9,116,100,0.9,106,100,0.9\n" # 18 cm/sec
        data += "3,100,100,0.9,123,100,0.9,113,100,0.9"   # 21 cm/sec
        
        csv_file = self.create_test_csv(data)
        self.analyzer.calculate_average_speed__per_frame(csv_file)
        self.assertEqual(round(self.analyzer.average_speed, 2), 18.00)  # Average of 15, 18, and 21
    
    def test_diagonal_movement(self):
        """Test that diagonal movement speed is calculated correctly using Pythagorean theorem"""
        # Create diagonal movement (3,4,5 triangle)
        data = "0,100,100,0.9,105,100,0.9,95,100,0.9\n"      # Start
        data += "1,100,100,0.9,108,104,0.9,98,104,0.9"       # Diagonal movement (3,4 units = 5 units total)
        
        csv_file = self.create_test_csv(data)
        self.analyzer.calculate_average_speed__per_frame(csv_file)
        # 5 cm movement at 30 fps = 15 cm/sec
        expected_speed = 5 * 3  # 5 units * 3 (fps multiplier for cm/sec)
        self.assertAlmostEqual(self.analyzer.average_speed, expected_speed, places=2)
     
    def test_varying_probabilities(self):
        """Test behavior with probabilities near the threshold"""
        data = "0,100,100,0.61,105,100,0.9,95,100,0.9\n"   # Just above threshold
        data += "1,100,100,0.59,110,100,0.59,100,100,0.9"   # Just below threshold
        
        csv_file = self.create_test_csv(data)
        self.analyzer.calculate_average_speed__per_frame(csv_file)
        self.assertEqual(self.analyzer.average_speed, 0)  # Should exclude due to one frame below threshold
    
    def test_mixed_speeds_with_below_threshold(self):
        """Test that speeds below 3 cm/sec are excluded when calculating average from multiple frames"""
        # Create four movements:
        # - First movement: 15 cm/sec (included)
        # - Second movement: 2 cm/sec (should be excluded as below threshold)
        # - Third movement: 18 cm/sec (included)
        # - Fourth movement: 21 cm/sec (included)
        
        data = "0,100,100,0.9,105,100,0.9,95,100,0.9\n"   # Start
        data += "1,100,100,0.9,110,100,0.9,100,100,0.9\n" # 15 cm/sec (included)
        data += "2,100,100,0.9,110.67,100,0.9,100.67,100,0.9\n" # 2 cm/sec (should be excluded)
        data += "3,100,100,0.9,116.67,100,0.9,106.67,100,0.9\n" # 18 cm/sec (included)
        data += "4,100,100,0.9,123.67,100,0.9,113.67,100,0.9"   # 21 cm/sec (included)
        
        csv_file = self.create_test_csv(data)
        self.analyzer.calculate_average_speed__per_frame(csv_file)
        
        # Only three speeds should be included (15, 18, 21)
        self.assertEqual(len(self.analyzer.speeds), 3)
        
        # Average should be (15 + 18 + 21) / 3 = 18
        self.assertAlmostEqual(self.analyzer.average_speed, 18.00, places=2)
        
        # Verify the 2 cm/sec movement was excluded
        expected_speeds = [15.0, 18.0, 21.0]
        for i, speed in enumerate(self.analyzer.speeds):
            self.assertAlmostEqual(speed, expected_speeds[i], places=1)
    
    def tearDown(self):
        self.root.destroy()

if __name__ == '__main__':
    unittest.main()