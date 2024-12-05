import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import cv2
import json
from objectLoc import RegionMarker
import math
import numpy as np
import pandas as pd

class InteractionAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ineraction analyzer")
        self.root.geometry("800x600")

        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # File selection
        ttk.Label(main_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.csv_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.csv_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_csv).grid(row=0, column=2)
        
        ttk.Label(main_frame, text="Video File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.video_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.video_path, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_video).grid(row=1, column=2)
        
        ttk.Label(main_frame, text="Output Video:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.output_path = tk.StringVar(value="output.mp4")
        ttk.Entry(main_frame, textvariable=self.output_path, width=50).grid(row=2, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(row=2, column=2)
        
        # Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        param_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(param_frame, text="Radius:").grid(row=0, column=0, sticky=tk.W)
        self.radius = tk.StringVar(value="75")
        ttk.Entry(param_frame, textvariable=self.radius, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(param_frame, text="Angle:").grid(row=0, column=2, sticky=tk.W, padx=(20,0))
        self.angle = tk.StringVar(value="30")
        ttk.Entry(param_frame, textvariable=self.angle, width=10).grid(row=0, column=3, padx=5)
        
        # Regions
        regions_frame = ttk.LabelFrame(main_frame, text="Regions", padding="10")
        regions_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        self.regions = tk.StringVar(value="left,right")
        ttk.Label(regions_frame, text="Region Names (comma-separated):").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(regions_frame, textvariable=self.regions, width=40).grid(row=0, column=1, padx=5)
        
        # Run button
        ttk.Button(main_frame, text="Run Analysis", command=self.run_analysis).grid(row=5, column=0, columnspan=3, pady=20)
        
    def browse_csv(self):
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.csv_path.set(filename)

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select video file",
            filetypes=(("Video files", "*.mp4 *.avi"), ("All files", "*.*"))
        )
        if filename:
            self.video_path.set(filename)

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="Save output video as",
            defaultextension=".mp4",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )
        if filename:
            self.output_path.set(filename)
    


     # Calculate the angle between the nose and the object using vectors
    def calculate_angle_between_vectors(self, v1, v2):
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        # Calculate dot product
        dot_product = np.dot(v1_norm, v2_norm)
        
        # Ensure dot product is in valid range [-1, 1]
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate angle in degrees
        angle = math.degrees(math.acos(dot_product))

        return angle

    # Function that checks to see if nose is within radius and pointing at the center of an object
    def check_interaction(self, nose_cords, left_cords, right_cords, object_regions):
        nx, ny = nose_cords
        lx, ly = left_cords
        rx, ry = right_cords
        # Calculate midpoint between ears
        mid_x = (lx + rx) / 2
        mid_y = (ly + ry) / 2
        # Calculate nose direction vector (from midpoint to nose)
        nose_vector = np.array([nx - mid_x, ny - mid_y])
        # Calculate the angle change per pixel so that angle of nose can adjust based on distance
        angle_diff = 90 - float(self.angle.get())
        angle_per_pixel = angle_diff / float(self.radius.get())

        # Iterate through each object and check to see if nose is within radius passed in
        for object in object_regions:
            ox, oy = object["center"]
            distance = math.sqrt((ox - nx)**2 + (oy - ny)**2)
            # If within radius then check angle of nose as well
            if distance < object['radius']:
                # Create vector from nose to center of object
                object_vector = np.array([ox - nx, oy - ny])
                # Find difference in distance between radius and actual distance of nose
                distance_difference = float(self.radius.get()) - distance
                # Check if angle of head is within angle of threshold given the distance to object
                if self.calculate_angle_between_vectors(nose_vector, object_vector) <= distance_difference * angle_per_pixel + float(self.angle.get()):
                    # Increment the interaction count of the object
                    object["interaction_count"] += 1
                    return object["title"]
        return None

    # Go through and check for interaction for every frame of location csv
    def count_interactions(self, csv_file, object_regions):
        df = pd.read_csv(csv_file, skiprows=2)

        # Iterate through each row and check interaction
        for index, row in df.iterrows():
            # Get all of the position coords
            nose_coords = (float(row.iloc[1]), float(row.iloc[2]))
            nose_prob = float(row.iloc[3])
            r_ear_coords = (float(row.iloc[4]), float(row.iloc[5]))
            r_ear_prob = float(row.iloc[6])
            l_ear_coords = (float(row.iloc[7]), float(row.iloc[8]))
            l_ear_prob = float(row.iloc[9])
            # Skip frame if the probabliities or too low for any of the labeled parts
            probabilitiy_threshold = 0.6
            if (nose_prob < probabilitiy_threshold or l_ear_prob < probabilitiy_threshold or r_ear_prob < probabilitiy_threshold):
                continue

            # Check the interaction for each frame
            self.check_interaction(nose_coords, l_ear_coords, r_ear_coords, object_regions)

    # Mark every interaction on a new video to output
    def process_video_interactions(self, csv_file, video_path, object_regions, output_path):
        # Read the CSV file
        df = pd.read_csv(csv_file, skiprows=2)

        # Open the input video for labeling
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get the video frame properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Create a video write
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_index = 0
        while cap.isOpened() and frame_index < len(df):
            ret, frame = cap.read()
            if not ret:
                break

            # Get coordinates for current frame
            row = df.iloc[frame_index]
            nose_coords = (float(row.iloc[1]), float(row.iloc[2]))
            nose_prob = float(row.iloc[3])
            l_ear_coords = (float(row.iloc[4]), float(row.iloc[5]))
            l_ear_prob = float(row.iloc[6])
            r_ear_coords = (float(row.iloc[7]), float(row.iloc[8]))
            r_ear_prob = float(row.iloc[9])

            # Skip if the probability is not within threshold
            probability_threshold = 0.6
            if (nose_prob < probability_threshold or l_ear_prob < probability_threshold or r_ear_prob < probability_threshold):
                frame_index += 1
                out.write(frame)
                continue

            interaction = self.check_interaction(nose_coords, l_ear_coords, r_ear_coords, object_regions)

            # If interaction then add label
            if interaction:
                # Find the interacting object and draw circle for label
                for object in object_regions:
                    if object["title"] == interaction:
                        cv2.circle(frame, object['center'], object['radius'], (0, 0, 255), 2)
                        # Add a region title
                        cv2.putText(frame, object['title'], object['center'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        break
            
            # Write the frame
            out.write(frame)
            frame_index += 1
            # Show progress
            if frame_index % 100 == 0:
                print(f"Processed {frame_index}/{len(df)} frames")

        cap.release()
        out.release()
        print("Video processing complete!")
        return output_path
    
    # Run the analysis
    def run_analysis(self):
        try:
            # Validate the inputs
            if not self.csv_path.get() or not self.video_path.get():
                messagebox.showerror("Error", "Please select both CSV and video files")
                return
            if not self.radius.get().isdigit() or not self.angle.get().isdigit():
                messagebox.showerror("Error", "Radius and angle must be numbers")
                return
            
            # Get the parameters
            regions = [r.strip() for r in self.regions.get().split(",")]
            radius = int(self.radius.get())
            angle = int(self.angle.get())

            # Create a region marker
            marker = RegionMarker(self.video_path.get(), radius, regions, angle)
            marked_regions = marker.mark_regions()

            # Count the interactions
            self.count_interactions(self.csv_path.get(), marked_regions)

            # Process video
            output_path = self.process_video_interactions(self.csv_path.get(), self.video_path.get(), marked_regions, self.output_path.get())

            # Show the results
            results = "\n".join([
                f"{obj['title']}: {obj['interaction_count']} frames "
                f"({obj['interaction_count'] / 30:.2f} seconds)"
                for obj in marked_regions
            ])

            messagebox.showinfo("Analysis Complete", 
                              f"Analysis completed successfully!\n\n"
                              f"Results:\n{results}\n\n"
                              f"Output video saved to:\n{output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = InteractionAnalyzerUI(root)
    root.mainloop()



if __name__ == '__main__':
    main()
