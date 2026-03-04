Autonomous Robot Navigation and Vision System

📌 Project Overview

This project implements an integrated robotics system combining autonomous mobile robot navigation with computer vision-based fruit quality detection.

The system consists of two primary modules:
1. Navigation Module (eBot)
2. Vision Module (OpenCV – Fruit Detection)
The objective is to demonstrate perception-driven robotics using predefined motion control and real-time image processing techniques.

🚗 Navigation Module (Mobile Robot)

🔹 Objective
To implement autonomous navigation of a mobile robot along a predefined path with basic shape detection and obstacle avoidance.

🔹 Features
- Predefined path execution
- Waypoint-based motion control
- Basic obstacle avoidance during traversal
- Square and triangle shape detection (experimental accuracy)

🔹 Working Principle
- A path is predefined using coordinate waypoints.
- The robot follows the path sequentially.
- During movement:
      The system monitors obstacles in the robot’s trajectory.
      If an obstacle is detected, avoidance behavior is triggered.
- The robot detects geometric shapes (square and triangle) using contour-    based vision processing.
- Shape classification is currently basic and not fully optimized.

🔹 Algorithms & Techniques Used
- Waypoint-based navigation logic
- Basic obstacle detection logic
- Contour detection
- Polygon approximation for shape classification

🔹 Limitations
- Shape detection accuracy is not fully optimized.
- Path is predefined (not dynamically planned).
- No advanced SLAM or localization implemented.

👁 Vision Module (OpenCV – Fruit Detection)

🔹 Objective
To classify fruits as good or bad based on color characteristics using HSV color space segmentation.

🔹 Features
- HSV-based color segmentation
- Fruit detection using contour extraction
- Good fruit and bad fruit classification
- Visual marking of detected fruits

🔹 Working Principle
- Input image is converted from BGR to HSV color space.
- HSV thresholds are defined for:
      Good fruit
      Bad fruit
- Binary masks are generated based on HSV ranges.
- Contours are extracted from segmented regions.
- Detected fruits are classified and labeled:
      Good fruits marked visually
      Bad fruits marked distinctly

🔹 Techniques Used
- HSV color thresholding
- Mask generation
- Morphological operations (if applied)
- Contour detection
- Bounding box annotation

🔹 Limitations
- Classification depends strictly on predefined HSV values.
- Sensitive to lighting conditions.
- No machine learning-based classification used.

🛠 Technologies Used
- Python
- OpenCV
- Mobile robot platform (eBot)
- Image processing techniques
- Basic motion control logic# AgroVision-eBot-Navigation
