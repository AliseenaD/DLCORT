import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from objectLoc import RegionMarker
import math
import numpy as np
import pandas as pd

class InteractionAnalyzerUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ineraction analyzer")
        self.root.geometry("800x800")

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

        # Frames per second
        frames_frame = ttk.LabelFrame(main_frame, text="Frames Per Second", padding="10")
        frames_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(frames_frame, text="Video FPS:").grid(row=0, column=0, sticky=tk.W)
        self.fps = tk.StringVar(value="30")
        ttk.Entry(frames_frame, textvariable=self.fps, width=10).grid(row=0, column=1, padx=5)

        # Pixels per cm
        pixel_frame = ttk.LabelFrame(main_frame, text="Pixels Per CM", padding="10")
        pixel_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(pixel_frame, text="Pixel/Cm:").grid(row=0, column=0, sticky=tk.W)
        self.pixel_ratio = tk.StringVar(value="30")
        ttk.Entry(pixel_frame, textvariable=self.pixel_ratio, width=10).grid(row=0, column=1, padx=5)
        self.average_speed = 0

        # Crop video if needed
        crop_video_frame = ttk.LabelFrame(main_frame, text="Crop video if no analysis performed yet (not required for previously analyzed videos)", padding="10")
        crop_video_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        # x1 coordinate
        ttk.Label(crop_video_frame, text="x1:").grid(row=0, column=0, sticky=tk.W)
        self.x1 = tk.StringVar(value="")
        ttk.Entry(crop_video_frame, textvariable=self.x1, width=10).grid(row=0, column=1, padx=5)
        # y1 coordinate
        ttk.Label(crop_video_frame, text="y1:").grid(row=0, column=2, sticky=tk.W)
        self.y1 = tk.StringVar(value="")
        ttk.Entry(crop_video_frame, textvariable=self.y1, width=10).grid(row=0, column=3, padx=5)
        # x2 coordinate
        ttk.Label(crop_video_frame, text="x2:").grid(row=0, column=4, sticky=tk.W)
        self.x2 = tk.StringVar(value="")
        ttk.Entry(crop_video_frame, textvariable=self.x2, width=10).grid(row=0, column=5, padx=5)
        # y2 coordinate
        ttk.Label(crop_video_frame, text="y2:").grid(row=0, column=6, sticky=tk.W)
        self.y2 = tk.StringVar(value="")
        ttk.Entry(crop_video_frame, textvariable=self.y2, width=10).grid(row=0, column=7, padx=5)
        ttk.Button(crop_video_frame, text="Crop Video", command=self.crop_video).grid(row=0, column=9, padx=5)

        # Frame trimming
        trim_frame = ttk.LabelFrame(main_frame, text="Trim to a designated number of frames", padding="10")
        trim_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(trim_frame, text="Frame Trim").grid(row=0, column=0, sticky=tk.W)
        self.frame_count = tk.StringVar(value='')
        ttk.Entry(trim_frame, textvariable=self.frame_count, width=10).grid(row=0, column=1, padx=5)
        ttk.Button(trim_frame, text="Trim CSV", command=self.trim_csv).grid(row=0, column=2, padx=5)

        # Run button
        ttk.Button(main_frame, text="Run Analysis", command=self.run_analysis).grid(row=10, column=0, columnspan=3, pady=20)
        ttk.Button(main_frame, text="Calculate Average Speed", command=self.calculate_average_speed__per_frame).grid(row=11, column=0, columnspan=3, pady=20)
        
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

    # Function that will crop the video and set the video path to that for analysis
    def crop_video(self):
        # First ensure all values are present
        if not self.video_path.get() or not all([self.x1.get().isdigit(), 
                                                 self.x2.get().isdigit(), 
                                                 self.y1.get().isdigit(), 
                                                 self.y2.get().isdigit()]):
            messagebox.showerror("Error", "Please provide a valid video path")
            return

        # Setup output path and coordinates
        output_path = self.video_path.get().rsplit('.', 1)[0] + '_cropped.mp4'
        x1 = int(self.x1.get())
        x2 = int(self.x2.get())
        y1 = int(self.y1.get())
        y2 = int(self.y2.get())
        try:
            # Get video properties and set up video writer
            cap = cv2.VideoCapture(self.video_path.get())
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (x2 - x1, y2- y1)
            )

            # Crop video
            while True:
                ret, frame = cap.read()
                if not ret:
                    break;

                cropped_frame = frame[y1:y2, x1:x2]
                out.write(cropped_frame)
            
            cap.release()
            out.release()

            # Notify of successful cropping
            messagebox.showinfo("Success", f"Successfully saved cropped video file as: {output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while cropping the video: {str(e)}")

    # Function that trims the CSV based on the number of frames provided
    def trim_csv(self):
        # Check to ensure csv path and number of frames are present
        if not self.csv_path.get():
            messagebox.showerror("Error", "Please input a valid CSV path first")
        if not self.frame_count.get().isdigit():
            messagebox.showerror("Error", "Please provide a valid digit for number of frames to trim to")

        try:
            # Save new csv with the number of frames desired
            df = pd.read_csv(self.csv_path.get(), skiprows=2)
            frame_number = int(self.frame_count.get())
            trimmed_df = df.head(frame_number)

            # Generate file name and save it
            output_filename = self.csv_path.get().rsplit('.', 1)[0] + '_' + str(frame_number) + '_trimmed.csv'
            trimmed_df.to_csv(output_filename, index=False)

            messagebox.showinfo("Success", "Successfully saved trimmed CSV file")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while trimming the CSV: {str(e)}")

    # Function that calculates speed per frame and then 
    def calculate_average_speed__per_frame(self):
        # first ensure csv_path is not empty
        if not self.csv_path.get():
            messagebox.showerror("Error", "Please input a valid CSV path first")

        # Open the file and skip the first two rows
        df = pd.read_csv(self.csv_path.get(), skiprows=2)
        speeds = []
        first_loc = (0,0)
        second_loc = (0,0)
        # Iterate through every row, calculate speed, if above 3 cm/sec then add to speeds
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

            # If either left or right ear prob less than threshold then continue until next frame
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
            distance = np.sqrt(((second_loc[0] - first_loc[0]) / float(self.pixel_ratio.get()))**2 + ((second_loc[1] - first_loc[1]) / float(self.pixel_ratio.get()))**2)

            # Now calculate speed and add to array if greater than or equal to 3
            speed = distance * float(self.fps.get())
            if speed >= 3:
                speeds.append(speed)

        self.average_speed = np.mean(speeds) if speeds else 0
        messagebox.showinfo('Speed Analysis Complete',
                            "🏃 MOVEMENT ANALYSIS\n"
                            f"• Average speed: {self.average_speed:.2f} cm/sec\n")
    
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

        # Make sure to set interaction counts to zero at the start
        print("Printing out the object interaction counts")
        for object in object_regions:
            print(object["interaction_count"])

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
        
        print("Printing out the object interaction counts")
        for object in object_regions:
            print(object["interaction_count"])

    '''
    Takes in the coordinates of the left and right ear and checks which quadrant the mouse is 
    in within the box. Returns the quadrant object with updated frame count for the mouse in that quadrant
    '''
    def check_quadrant_interaction(self, left_cords, right_cords, video_width, video_height, quadrant_object):
        lx, ly = left_cords
        rx, ry = right_cords
        # Calculate midpoint between ears
        mid_x = (lx + rx) / 2
        mid_y = (ly + ry) / 2

        # Calculate the quadrant borders
        x_quadrant_border = video_width / 2
        y_quadrant_border = video_height / 2

        # Now check which quadrant mouse is in and update the frame count
        if mid_x <= x_quadrant_border and mid_y <= y_quadrant_border:
            quadrant_object["upper-left"] += 1
            return "upper-left"
        elif mid_x > x_quadrant_border and mid_y <= y_quadrant_border:
            quadrant_object["upper-right"] += 1
            return "upper-right"
        elif mid_x > x_quadrant_border and mid_y > y_quadrant_border:
            quadrant_object["bottom-right"] += 1
            return "bottom-right"
        elif mid_x < x_quadrant_border and mid_y > y_quadrant_border:
            quadrant_object["bottom-left"] += 1
            return "bottom-left"
    
    '''
    Takes in a CSV file and extracts the mouses ear locations, then checks which quadrant
    the mouse was in for every frame and returns an object with the count of each quadrant
    '''
    def count_quadrant_interactions(self, csv_file, video_width, video_height):
        df = pd.read_csv(csv_file, skiprows=2)

        # Define the quadrant object
        quadrant_object = {
            "upper-left": 0,
            "upper-right": 0,
            "bottom-left": 0,
            "bottom-right": 0
        }

        # Go through every frame, extract ear locations, and check quadrant location
        for index, row in df.iterrows():
            # Get position coords and probability
            r_ear_coords = (float(row.iloc[4]), float(row.iloc[5]))
            r_ear_prob = float(row.iloc[6])
            l_ear_coords = (float(row.iloc[7]), float(row.iloc[8]))
            l_ear_prob = float(row.iloc[9])
            # Skip frame if the probabliities or too low for any of the labeled parts
            probabilitiy_threshold = 0.6
            if (l_ear_prob < probabilitiy_threshold or r_ear_prob < probabilitiy_threshold):
                continue

            # Check quadrant loc
            self.check_quadrant_interaction(left_cords=l_ear_coords, 
                                            right_cords=r_ear_coords,
                                            video_height=video_height,
                                            video_width=video_width,  
                                            quadrant_object=quadrant_object)
        
        # Return the quadrant object
        return quadrant_object
    
    '''
    Takes in the csv file, video path, output path, the video capturer, and video dimensions and generates a new video showing a live
    count of the number of frames the mouse has spent in each quadrant
    '''
    def process_video_quadrant_interactions(self, csv_file, output_path, cap, video_width, video_height):
        # Read the CSV file and remaining video props (fps)
        df = pd.read_csv(csv_file, skiprows=2)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(output_path.split('.')[0])
        reformatted_output_name = output_path.split('.')[0] + 'quadrant_analysis'
        reformatted_output = reformatted_output_name + '.' + output_path.split('.')[1]

        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(reformatted_output, fourcc, fps, (video_width, video_height))

        # Initialize display counters
        quadrant_object = {
            "upper-left": 0,
            "upper-right": 0,
            "bottom-left": 0,
            "bottom-right": 0
        }

        frame_index = 0
        while cap.isOpened() and frame_index < len(df):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get the coordinates for the current frame
            row = df.iloc[frame_index]
            l_ear_coords = (float(row.iloc[4]), float(row.iloc[5]))
            l_ear_prob = float(row.iloc[6])
            r_ear_coords = (float(row.iloc[7]), float(row.iloc[8]))
            r_ear_prob = float(row.iloc[9])

            # Skip if the probability is not within threshold
            probability_threshold = 0.6
            if (l_ear_prob < probability_threshold or r_ear_prob < probability_threshold):
                frame_index += 1
                out.write(frame)
                continue

            # Calculate quadrant borders
            x_quadrant_border = video_width // 2
            y_quadrant_border = video_height // 2
            
            # Draw quadrant lines
            cv2.line(frame, (x_quadrant_border, 0), (x_quadrant_border, video_height), (255, 255, 0), 1)
            cv2.line(frame, (0, y_quadrant_border), (video_width, y_quadrant_border), (255, 255, 0), 1)
        

            # Check loc within quadrants
            current_quadrant = self.check_quadrant_interaction(left_cords=l_ear_coords, 
                                            right_cords=r_ear_coords,
                                            video_height=video_height,
                                            video_width=video_width,  
                                            quadrant_object=quadrant_object)
            
            # Highlight the current quadrant
            if current_quadrant == "upper-left":
                cv2.rectangle(frame, (0, 0), (x_quadrant_border, y_quadrant_border), (0, 255, 0), 2)
            elif current_quadrant == "upper-right":
                cv2.rectangle(frame, (x_quadrant_border, 0), (video_width, y_quadrant_border), (0, 255, 0), 2)
            elif current_quadrant == "bottom-right":
                cv2.rectangle(frame, (x_quadrant_border, y_quadrant_border), (video_width, video_height), (0, 255, 0), 2)
            elif current_quadrant == "bottom-left":
                cv2.rectangle(frame, (0, y_quadrant_border), (x_quadrant_border, video_height), (0, 255, 0), 2)
            

            # Display the counts on the video
            y_offset = 20  
            for title, count in quadrant_object.items():
                text = f"{title}: {count} frames ({count/fps:.1f}s)"
                # Draw black background for better visibility
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (10, y_offset - text_height), 
                            (10 + text_width, y_offset + 5), 
                            (0, 0, 0), -1)
                # Draw text in white
                cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                y_offset += 20  # Move down for next counter

            # Write the frame
            out.write(frame)
            frame_index += 1
            # Show progress
            if frame_index % 3000 == 0:
                print(f"Processed {frame_index}/{len(df)} frames")

        cap.release()
        out.release()
        print("Video processing complete!")
        return reformatted_output

    # Mark every interaction on the video and keep track of frames interacted too
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

        # Create a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Initialize display counters
        display_counts = {obj["title"]: 0 for obj in object_regions}

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
            
            # Check interaction with temporary objects (duplicates with 0 for interaction count)
            temp_regions = [{"title": obj["title"], "center": obj["center"], "radius": obj["radius"], "interaction_count": 0} for obj in object_regions]
            interaction = self.check_interaction(nose_coords, l_ear_coords, r_ear_coords, temp_regions)

            # Update display counts if there's an interaction
            if interaction:
                display_counts[interaction] += 1
                
                # Find the interacting object and draw circle for label
                for object in object_regions:
                    if object["title"] == interaction:
                        cv2.circle(frame, object['center'], object['radius'], (0, 0, 255), 2)
                        # Add a region title
                        cv2.putText(frame, object['title'], object['center'], 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        break

            # Draw running counters
            y_offset = 30  
            for title, count in display_counts.items():
                text = f"{title}: {count} frames ({count/fps:.1f}s)"
                # Draw black background for better visibility
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (10, y_offset - text_height), 
                            (10 + text_width, y_offset + 5), 
                            (0, 0, 0), -1)
                # Draw text in white
                cv2.putText(frame, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30  # Move down for next counter
            
            # Write the frame
            out.write(frame)
            frame_index += 1
            # Show progress
            if frame_index % 3000 == 0:
                print(f"Processed {frame_index}/{len(df)} frames")

        cap.release()
        out.release()
        print("Video processing complete!")
        return output_path
    
    '''Takes in the marked_region object and returns the quadrant that object is in, whether it be
    upper-left, upper-right, bottom-left, or bottom-right'''
    def check_quadrant(self, marked_region, video_width, video_height):
        center_x, center_y = marked_region['center']
         # Calculate the quadrant borders
        x_quadrant_border = video_width / 2
        y_quadrant_border = video_height / 2

        # Now check which quadrant mouse is in and update the frame count
        if center_x <= x_quadrant_border and center_y <= y_quadrant_border:
            return "upper-left"
        elif center_x > x_quadrant_border and center_y <= y_quadrant_border:
            return "upper-right"
        elif center_x > x_quadrant_border and center_y > y_quadrant_border:
            return "bottom-right"
        elif center_x < x_quadrant_border and center_y > y_quadrant_border:
            return "bottom-left"
        
    
    '''Runs the quadrant analysis by checking the mouses location in respect to the quadrants of the box for every frame of video'''
    def run_quadrant_analysis(self, marked_regions):
        try:
            # Validate the inputs
            if not self.csv_path.get() or not self.video_path.get():
                messagebox.showerror("Error", "Please select both CSV and video files")
                return
            
            # Get the video dimensions
            # Open the input video for labeling
            cap = cv2.VideoCapture(self.video_path.get())
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get the video frame properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Go through and obtain the quadrants each object is in
            object_regions = {}
            for object in marked_regions:
                object_quadrant = self.check_quadrant(object, video_width=frame_width, video_height=frame_height)
                object_regions[object['title']] = object_quadrant

            # Count the interactions in csv
            quadrant_interactions = self.count_quadrant_interactions(self.csv_path.get(), video_width=frame_width, video_height=frame_height)
            # Create the output video
            output_path = self.process_video_quadrant_interactions(self.csv_path.get(),
                                                                self.output_path.get(), cap=cap, video_width=frame_width,
                                                                video_height=frame_height)
            
            # Format the quadrant interaction results
            interaction_results = "\n".join([
                f"• {title}: {count} frames ({count / float(self.fps.get()):.2f} seconds)"
                for title, count in quadrant_interactions.items()
            ])
            
            # Format the object quadrant locations
            object_quadrants = "\n".join([
                f"• {title} object: {quadrant} quadrant"
                for title, quadrant in object_regions.items()
            ])
            
            # Create the formatted message
            message = (
                "🧩 QUADRANT ANALYSIS\n"
                f"{interaction_results}\n\n"
                "📍 OBJECT LOCATIONS\n"
                f"{object_quadrants}\n\n"
                "🎬 QUADRANT VIDEO\n"
                f"• Saved to: {output_path}\n\n"
            )
            
            return message
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
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

            # Obtain the quadrant analysis results
            quadrant_results = self.run_quadrant_analysis(marked_regions)

            # Count the interactions
            self.count_interactions(self.csv_path.get(), marked_regions)

            # Process video
            output_path = self.process_video_interactions(self.csv_path.get(), self.video_path.get(), marked_regions, self.output_path.get())

            # Format interaction results
            interaction_results = "\n".join([
                f"• {obj['title']}: {obj['interaction_count']} frames ({obj['interaction_count'] / float(self.fps.get()):.2f} seconds)"
                for obj in marked_regions
            ])

            # Create the formatted message
            message = (
                "═════════ ANALYSIS RESULTS ═════════\n\n"
                
                "📊 INTERACTION SUMMARY\n"
                f"{interaction_results}\n\n"
                
                "🎥 OUTPUT VIDEO\n"
                f"• Saved to: {output_path}\n\n"
                
                f"{quadrant_results}"
            )
            
            messagebox.showinfo("Analysis Complete", message)

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

def main():
    root = tk.Tk()
    app = InteractionAnalyzerUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
