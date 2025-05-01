# DeepLabCut Object Recognition Test Analysis

This software enables analysis of object recognition tests using labeled DeepLabCut mouse videos. Users can place multiple customizable object markers within the video’s field of view. The analysis measures mouse-object interactions by calculating the distance between the mouse’s head and the center of each object, ensuring that the head is both within the object's radius and oriented toward the object (with a manually adjustable angle threshold). Additional features include the calculation of the mouse’s average speed, automatically excluding periods when movement is below 3 cm/sec. For preprocessing, users also have the ability to crop video frames to a specific area and trim the experiment duration to focus the analysis on a desired timeframe. It is important to notice that this analysis is meant to be completed AFTER experiment videos have been labeled via DeepLabCut. To create the DeepLabCut-labeled videos, read the instructions below.

## Labeling Position with DeepLabCut

1. First, ensure the raw videos that have been recorded are uploaded into Google Drive
2. Navigate to the following Colab Notebook: https://colab.research.google.com/drive/1prVDdYurjfN0ELowvLjMPzLgC5igRwH3?usp=sharing
3. Run the first cell to install all required dependencies for the notebook
4. Edit the ProjectFolderName and videofile_path variables to make sure they point to the correct location in which the raw videos are stored in the Drive, then run the cell
5. Scroll down to the 'Labeling New Video' cell and ensure the video_files path is pointing correctly to all of the videos to be processed, then run the cell. This cell will output the CSV used within the rest of the analysis pipeline.
6. Run the 'Plot Trajectories' and 'Make Labeled Video' cells to obtain graphs of the mouse's location as well as the labeled video to be used within the rest of this analysis pipeline. 
7. Download the CSV as well as the labeled video to the device in which DLCORT is installed.

## Environment Setup

1. Clone the code from this repository.
2. Ensure you have Anaconda installed on your device. You can download it [here](https://www.anaconda.com/download/success).
3. Create a new Conda environment:
   ```bash
   conda create --name dlcort python=3.11
   ```
4. Activate the environment:
   ```bash
   conda activate dlcort
   ```
   To deactivate it at any time:
   ```bash
   conda deactivate
   ```
5. Install all dependencies by navigating to the repository folder in your terminal and running:
   ```bash
   pip install -r requirements.txt
   ```

## Using the Analysis Tool

1. Navigate to the repository folder in your terminal.
2. Run the analysis script:
   ```bash
   python analyzeLoc.py
   ```

### Analysis Options

- **Full Analysis (Interaction Times and Quadrant Analysis)**  
  - Input the DeepLabCut CSV file, the video file, the name for the output video you would like, as well as the path to the results CSV to write the results to.
  - Edit the object parameters and set the correct frames per second (FPS) of the video as well as the correct pixels per centimeter.
  - Click **"Run Analysis"**.
  - A new video will be generated at the output path you specify along with all of the results being added to the output CSV you provided.

- **Video Cropping**  
  - Useful when analyzing a video for the first time (to focus only on the surface area of the box).
  - Click and drag your mouse across the region of the video you want to crop.
  - Click **"Crop Video"**.
  - A new cropped video will be created.

- **Trim CSV to a Specific Number of Frames**  
  - Input a CSV file.
  - Enter the number of frames you want to keep.
  - Click **"Trim CSV"**.
  - A new trimmed CSV file will be generated.
