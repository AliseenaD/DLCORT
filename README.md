# DeepLabCut Object Recognition Test Analysis

This software enables analysis of object recognition tests using labeled DeepLabCut mouse videos. Users can place multiple customizable object markers within the video’s field of view. The analysis measures mouse-object interactions by calculating the distance between the mouse’s head and the center of each object, ensuring that the head is both within the object's radius and oriented toward the object (with a manually adjustable angle threshold). Additional features include the calculation of the mouse’s average speed, automatically excluding periods when movement is below 3 cm/sec. For preprocessing, users also have the ability to crop video frames to a specific area and trim the experiment duration to focus the analysis on a desired timeframe.

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
  - Input the CSV file and the corresponding video file.
  - Edit the object parameters and set the correct frames per second (FPS) of the video.
  - Click **"Run Analysis"**.
  - A new video will be generated at the output path you specify.

- **Speed Analysis**  
  - Input the CSV file and video file.
  - Ensure that the FPS and pixels per centimeter values are correctly set.
  - Click **"Calculate Average Speed"**.

- **Video Cropping**  
  - Useful when analyzing a video for the first time (to focus only on the surface area of the box).
  - Enter the pixel values for the corners you wish to crop.
  - Click **"Crop Video"**.
  - A new cropped video will be created.

- **Trim CSV to a Specific Number of Frames**  
  - Input a CSV file.
  - Enter the number of frames you want to keep.
  - Click **"Trim CSV"**.
  - A new trimmed CSV file will be generated.
