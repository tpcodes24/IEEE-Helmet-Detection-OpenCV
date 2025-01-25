Automatic Detection of Motorcyclists with and without Safety Helmets

Author of the IEEE paper: https://ieeexplore.ieee.org/document/9215415

Abstract
Motorcycle usage is rapidly increasing in developing countries like India, which has led to a significant rise in motorcycle-related accidents. To address this issue, we propose a real-time system to detect motorcyclists with and without safety helmets using traffic surveillance footage.

Our approach involves:

Vehicle detection and tracking: Implemented using OpenCV and pipelined with image processing tools.
Feature extraction: Utilizing the Histogram of Oriented Gradients (HOG) descriptor.
Classification: Employing the Linear Support Vector Classifier (LinearSVC) to identify whether a rider is wearing a safety helmet.
Data storage and visualization: Storing results in a MySQL database with timestamps and visualizing through a tabular and graphical desktop interface.
With an 87.6% model accuracy, this solution enhances traffic safety measures and offers a time-efficient approach to enforcing helmet usage.

Key Features
Real-time helmet detection: Process live traffic surveillance footage to identify motorcyclists.
Machine learning techniques: Leverages HOG descriptors and LinearSVC for classification.
Database integration: Automatically logs results with timestamps in a MySQL database.
User-friendly visualization: Provides tabular and graphical data representation in a desktop application.