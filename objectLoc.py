import cv2
from pathlib import Path

class RegionMarker:
    # Initialize a new region marker, takes a video path as well as a radius to calculate the regions of the objects
    def __init__(self, video_path, radius, regions, angle):
        self.video_path = Path(video_path)
        self.radius = radius
        self.regions = regions
        self.angle = angle
        self.current_region_index = 0
        self.region_areas = []

        # Get the first frame of the video
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Read the first frame of the video
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read the first frame of the video")
        
        # Get frame width and height
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define event that occurs when mouse is clicked, this function 
    # only occurs if the passed index is less than length of number of regions passed in
    def mouse_click(self, event, x, y, flags, param):
        if self.current_region_index < len(self.regions):
            # If left mouse click, assign the centerpoint and associated region as title
            if event == cv2.EVENT_LBUTTONDOWN:
                self.region_areas.append({
                    'title': self.regions[self.current_region_index],
                    'center': (x,y),
                    'radius': self.radius,
                    'angle': self.angle,
                    'interaction_count': 0
                })
                print(f"Added object {self.regions[self.current_region_index]} at ({x}, {y})")

                # Increment the region AFTER printing the message
                self.current_region_index += 1

                # If we've marked all objects, show completion message
                if self.current_region_index >= len(self.regions):
                    print("All objects have been marked please exit out by pressing any key")

                # Update display
                frame_copy = self.frame.copy()
                self.draw_regions(frame_copy)
                cv2.imshow('Object Marker', frame_copy)
    
    # Draws the regions on the frame based on where the user put the center point for the regions
    def draw_regions(self, frame):
        print('Current index:', self.current_region_index)
        for region in self.region_areas:
            # Draw a circle
            cv2.circle(frame, region['center'], region['radius'], (0, 0, 255), 2)
            # Draw a center point
            cv2.circle(frame, region['center'], 5, (0, 0, 255), 1)
            # Add a region title
            cv2.putText(frame, region['title'], region['center'], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Show labels based on current index and whether we need to mark more regions
        if self.current_region_index >= len(self.regions):
            # All objects marked, show escape message
            cv2.putText(frame, "Press Escape to exit frame",
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2)
        else:
            # Still have objects to mark, show current object
            cv2.putText(frame, f"Mark object {self.regions[self.current_region_index]}",
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255, 255, 255), 
                2)

    # Go through and mark all regions 
    def mark_regions(self):
        # Create window for marking
        cv2.namedWindow('Object Marker')
        cv2.setMouseCallback('Object Marker', self.mouse_click)

        print(f"Please mark all objects in order: {', '.join(self.regions)}")

        while True:
            frame_to_show = self.frame.copy()
            self.draw_regions(frame_to_show)
            cv2.imshow('Object Marker', frame_to_show)

            # Break if all objects have been marked and the ESC key has been pressed
            key = cv2.waitKey(0)
            if key==27 and self.current_region_index >= len(self.regions) - 1:
                break
            # Delete the last click if delete button pressed
            if key==8 and self.current_region_index > 0:
                self.current_region_index -= 1
                self.region_areas.pop()
                self.draw_regions(frame_to_show)

        cv2.destroyAllWindows()
        return self.region_areas
    
    def __del__(self):
        """Clean up video capture object"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()