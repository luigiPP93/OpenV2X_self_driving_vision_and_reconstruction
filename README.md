# OpenV2X Framework

## Description

**OpenV2X** is a modular cyber-physical framework designed for real-time environmental perception and interactive feedback in software-defined vehicles. The system integrates vision-based perception with V2X communication capabilities to enable comprehensive scene reconstruction and human-in-the-loop feedback mechanisms.

This framework addresses critical challenges in autonomous driving by providing:
- **Modular architecture**: Independent, swappable components for object detection, lane detection, and communication
- **Real-time performance**: Optimized for embedded edge deployment with minimal latency
- **Dual lane detection approaches**: Support for both classical computer vision pipelines and deep learning models (UFLD)
- **V2X communication**: MQTT-based vehicle-to-everything messaging for cooperative perception
- **Interactive feedback system**: User interface for anomaly reporting and continuous system improvement

The complete methodology and validation results are detailed in the accompanying IEEE paper (included in this repository).

## Key Features

- **Object Detection Module**: YOLOv5-based vehicle detection with integrated orientation classification (frontal, rear, lateral)
- **Lane Detection Module**: Dual implementation supporting both classical vision pipeline and Ultra Fast Lane Detection (UFLD)
- **V2X Communication Layer**: MQTT-enabled bidirectional messaging for infrastructure integration
- **Environment Reconstruction**: Real-time semantic scene representation combining perception and V2X data
- **Anomaly Reporting Interface**: GUI for driver feedback collection to improve perception models post-deployment
- **Energy Profiling**: CodeCarbon integration for sustainability assessment
- **Embedded-Ready**: Validated on Raspberry Pi for edge deployment scenarios

## System Architecture

The framework operates in three functional layers:
1. **Perception Module**: Processes monocular RGB camera input for object and lane detection
2. **Communication Module**: Handles V2X message publishing/subscribing via MQTT broker
3. **Environment Reconstruction Module**: Fuses perception and V2X data for semantic scene representation with user feedback capabilities

## Dataset and Pre-trained Models

The system uses pre-trained weights for vehicle orientation recognition based on the [Vehicle Orientation Dataset](https://github.com/sekilab/VehicleOrientationDataset).

### Lane Detection Module

The Ultra Fast Lane Detection (UFLD) approach requires pre-trained models for TuSimple and CULane datasets:

#### TuSimple Model
- **Download**: [tusimple_18.pth](https://thehikari.file.core.windows.net/permanent/github/co-pilot-driving-system-with-opencv/models/tusimple_18.pth?sv=2023-01-03&st=2024-06-09T17%3A16%3A12Z&se=9999-12-30T15%3A00%3A00Z&sr=f&sp=r&sig=o71lN9sfny8lWKxLh6zVc1q3Ed18siMn%2BMUwhQ1uxQM%3D)
- **Destination**: Place in the `models/` directory

#### CULane Model
- **Download**: [culane_18.pth](https://thehikari.file.core.windows.net/permanent/github/co-pilot-driving-system-with-opencv/models/culane_18.pth?sv=2023-01-03&st=2024-06-09T17%3A15%3A06Z&se=9999-12-30T15%3A00%3A00Z&sr=f&sp=r&sig=vwaO8qFSnPo%2BX7bUp6LyEOAKnjmbXgZWadloHx2umVk%3D)
- **Destination**: Place in the `models/` directory

**Note**: Both models must be downloaded and placed in the `models/` directory for the UFLD lane detection module to function properly.


## Libraries

The project uses the following libraries:

- `PyTorch `
- `ultralytics` used for YOLO (car detection with [DataSet](https://github.com/sekilab/VehicleOrientationDataset))
- `tkinter`
- `codecarbon`
- `matplotlib`
- `opencv-python`
- `numpy`

## Images

<img src="./img/image1.png" alt="Image description" width="500"/>
Example image of a driving scenario

<img src="./img/image2.png" alt="Image description" width="500"/>
Second example image of a driving scenario

<img src="./img/gui.png" alt="Image description" width="400"/>
The GUI interface allows users to report issues by providing text input and simultaneously save the current video frame. This enables users to capture and document the exact moment an issue occurs, which can then be sent to the automotive manufacturer for further examination. 


### Prerequisites

Make sure you have the following software and libraries installed:

- Python 3.8.x
- Required Python packages (listed in `requirements.txt`)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/self_driving_vision_and_reconstruction.git
   ```
   ```bash
   cd self_driving_vision_and_reconstruction
   ```
    ```bash
   pip install -r requirements.txt
   ```
Additionally, download the dataset and model weights:

- Download the dataset from [this link](https://drive.google.com/drive/folders/1VkKwxuK8DOx7EsH9ZD5z_-nThg8BMyFE).
- Download the file `best.pt`, rename it to `yolov5_vehicle_oriented.pt`, and place it in the `yolo` directory.


   ```bash
   python Open_V2X_fremewor.py
   ```

An advanced system for 3D environment reconstruction for autonomous driving, similar to Tesla Vision.
# Configuration Options

## 1. Selecting Lane Detection Method

The framework supports two lane detection approaches to demonstrate modularity:

- **Classical Vision Pipeline** (traditional computer vision with ROI, Sobel/HLS filters)
- **Ultra Fast Lane Detection (UFLD)** (deep learning-based approach)

To switch between methods, modify the `pipeline_check` variable in `Open_V2X_framework.py`:

```python
pipeline_check = True   # Use classical vision pipeline
pipeline_check = False  # Use UFLD (deep learning)
```

#### 2. **Video Source Selection**

By default, the system uses a YouTube video (line 213). To test different environmental conditions, modify line 228:
```python
video_path = video_path[0]  # YouTube video (default)
video_path = video_path[4]  # Night conditions
video_path = video_path[5]  # Rain conditions
video_path = video_path[6]  # Daylight conditions
```

#### 3. **Pipeline-Specific Configuration**

If using the **classical vision pipeline** (`pipeline_check = True`), you must specify the environmental condition matching your selected video:
```python
img, lane, parameters = pipeline(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                                 mtx, dist, "Day")  # Options: "Day", "Night", "Rain"
```

Ensure this parameter matches your video selection from step 2.

## Adding Custom Videos

### For Classical Vision Pipeline

To add a custom video when using the pipeline approach:

1. **Camera Calibration**: Perform calibration for your specific video source
2. **Configure ROI**: In `modules/Object_detection_module/lane_detection_pipeline.py`:
   - Add your configuration to `configurazione_base`
   - Define the polygon for the Region of Interest (ROI) adapted to your video's perspective
   - The pipeline requires manual adaptation of these parameters for each new video source

### For UFLD Approach

When using Ultra Fast Lane Detection (`pipeline_check = False`):
- **No calibration required**
- Simply select the appropriate video type as described in [Configuration Options](#configuration-options)
- The deep learning model generalizes across different scenarios without manual tuning

## Performance Metrics

Based on validation results from the IEEE paper:

- **Object Detection**: 84.3% accuracy with orientation classification at 23 FPS
- **Lane Detection (Pipeline)**: 0.87 IoU (daylight), 0.73 IoU (rain)
- **Lane Detection (UFLD)**: 0.90 IoU (daylight), 0.60 IoU (rain)
- **V2X Communication**: <1ms average latency, zero message loss
- **Energy Consumption**: 0.002783 kWh per inference cycle

## Visual Examples

### Scene Reconstruction
The framework provides real-time semantic reconstruction combining lane geometry, detected vehicles with orientation, distance estimation, and V2X event overlays.

### Anomaly Reporting Interface
The GUI allows users to report system anomalies by:
- Providing textual descriptions
- Capturing the current video frame
- Sending reports to OEMs for continuous model improvement

## Future Development

This work remains open to extensions and improvements, including:

- Integration of additional perception modules (traffic sign recognition, pedestrian detection)
- Enhanced V2X communication protocols (C-V2X, DSRC)
- Expanded anomaly reporting categories
- Additional sensor fusion capabilities
- ASIL-compliant safety-critical module development

   