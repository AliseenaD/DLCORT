# DeepLabCut Object Recognition Test Analysis

This software enables analysis of object recognition tests using labeled DeepLabCut mouse videos. Users can place multiple customizable object markers within the video’s field of view. The analysis measures mouse-object interactions by calculating the distance between the mouse’s head and the center of each object, ensuring that the head is both within the object's radius and oriented toward the object (with a manually adjustable angle threshold). Additional features include the calculation of the mouse’s average speed, automatically excluding periods when movement is below 3 cm/sec. For preprocessing, users also have the ability to crop video frames to a specific area and trim the experiment duration to focus the analysis on a desired timeframe. It is important to notice that this analysis is meant to be completed **AFTER** experiment videos have been labeled via DeepLabCut. To create the DeepLabCut-labeled videos, read the instructions below.

## Labeling Position with DeepLabCut

1. First, ensure the raw videos that have been recorded are uploaded into Google Drive  
2. Navigate to the following Colab Notebook: https://colab.research.google.com/drive/1prVDdYurjfN0ELowvLjMPzLgC5igRwH3?usp=sharing  
3. Run the first cell to install all required dependencies for the notebook  
4. Edit the `ProjectFolderName` and `videofile_path` variables to make sure they point to the correct location in which the raw videos are stored in the Drive, then run the cell  
5. Scroll down to the 'Labeling New Video' cell and ensure the `video_files` path is pointing correctly to all of the videos to be processed, then run the cell. This cell will output the CSV used within the rest of the analysis pipeline.  
6. Run the 'Plot Trajectories' and 'Make Labeled Video' cells to obtain graphs of the mouse's location as well as the labeled video to be used within the rest of this analysis pipeline.  
7. Download the CSV as well as the labeled video to the device in which DLCORT is installed.  

## Environment Setup

1. Clone the code from this repository.  
2. Ensure you have Anaconda installed on your device. You can download it [here](https://www.anaconda.com/download/success).  
3. Create a new Conda environment (only have to do on first time ever using DLCORT):  
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
5. Install all dependencies by navigating to the repository folder in your terminal and running (only have to do on first time ever using DLCORT):  
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
  - Input the video to crop in the **Video File** text field.  
  - Input the CSV file that you would like to trim in the **CSV File** text field.  
  - Enter the name you would like for the output video and quadrant video to have in the **Output Video** text field.  
  - If you already have a CSV file with previously ran analyses, enter that file in the **Output CSV File** field. If this is your first time running the analysis, or you want to save the results to a new CSV file, enter a new CSV file name.  
  - The parameter values are already set to the default values used in prior experiments. An interaction will only be counted within a frame if both parameters are met.  
    - **Radius** signifies the pixel distance cutoff the mouse’s head can be from an object to be considered for interaction.  
    - **Angle** indicates the maximum angle in which the mouse’s head can be pointed away from the center of the object to be considered as an interaction. This angle scales 
    as the mouse gets further away from the object. It starts at 90˚ and decreases with distance from the object until it reaches the inputted angle at the edge of the radius. A higher radius allows for a wider range of angles. 
    - **Speed Cutoff** indicates the minimum speed the mouse must be going (in cm/sec) out for it to be counted towards the final speed calculation
    - **Frame Span Cutoff** indicates the miminum number of frames in a row the mouse must be interacted with an object for it to be included in the frame spans output file.
  - Enter the object names you would like the objects to have in the **Region Names** field. These names should be comma-separated. The default is “left” and “right”.  
  - Enter the frames per second the video was captured in the **Video FPS** text field. The default is 30 FPS.  
  - Enter the **Pixel/CM** value. You can calculate this using ImageJ by measuring the pixel distance across the box and dividing by its real-world length in centimeters.  
  - Click on the **Run Analysis** button.  

  **Outputs include:**  
  - A video with an interaction counter (same name as output video).  
  - A video with a quadrant counter (same name as output video with `_quadrant_analysis` suffix).  
  - A newly created CSV file containing all of the frame spans of mouse interaction and the object that was interacted with during those spans.
  - An updated or newly created CSV file with:
    - Interaction time with each object  
    - Time spent in each quadrant  
    - Quadrant assignment of each object  
    - Mouse average speed (excluding frames under 3 cm/sec)

- **Video Cropping**  
  - Input the video to crop in the **Video File** text field.  
  - Click **Crop Video**.  
  - The cropped video should appear in the same directory as the inputted **Video File**. The video will apear with the same name as the **Video File** with `_cropped` at the end. If it is not in the same directory, search your documents folder.  

- **Trim CSV to a Specific Number of Frames**  
  - Input the CSV file in the **CSV File** text field.  
  - Enter the number of frames to trim to in the **Frame Trim** text field.  
  - Click **Trim CSV**.  
  - The trimmed CSV file will appear in the same directory as the input file, with the name ending in `_[NumberOfFrames]_trimmed`.
