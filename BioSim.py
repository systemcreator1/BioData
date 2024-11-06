# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:17:23 2024

@author: Legoboy
"""

import pandas as pd
import cv2
import numpy as np
from collections import Counter

# Step 1: Read the CSV file to find the most common cell type
def get_most_common_cell_type(file_path):
    try:
        data = pd.read_csv(file_path)
        cell_types = data['cell_type'].tolist()
        if cell_types:
            most_common_cell_type = Counter(cell_types).most_common(1)[0][0]
            return most_common_cell_type
        else:
            print("No cell data found in the log.")
            return None
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
        return None

# Step 2: Generate a microscope-like structure based on the cell type
def draw_cell_under_microscope(cell_type):
    img = np.zeros((300, 300, 3), dtype=np.uint8)  # Black background as in microscope view

    if cell_type == "Normal":
        # Normal cells: Draw circular cells with a nucleus
        for i in range(3):  # Multiple normal cells
            center = (np.random.randint(80, 220), np.random.randint(80, 220))
            cv2.circle(img, center, 30, (255, 255, 200), -1)  # Light yellow cell body
            cv2.circle(img, center, 10, (150, 150, 100), -1)  # Nucleus in darker color

    elif cell_type == "Cancerous":
        # Cancerous cells: Draw irregular shapes with prominent nuclei
        for i in range(2):  # Fewer but larger cells
            points = np.array([[np.random.randint(70, 230), np.random.randint(70, 230)] for _ in range(8)], np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.fillPoly(img, [points], (255, 100, 100))  # Red-tinted irregular cell
            nucleus_center = tuple(points[np.random.randint(0, len(points))][0])
            cv2.circle(img, nucleus_center, 12, (100, 50, 50), -1)  # Darker nucleus

    elif cell_type == "Bacteria":
        # Bacteria cells: Small, oval-shaped cells with simple structure
        for i in range(5):  # Multiple bacteria
            center = (np.random.randint(50, 250), np.random.randint(50, 250))
            cv2.ellipse(img, center, (20, 8), angle=np.random.randint(0, 180), startAngle=0, endAngle=360, color=(200, 200, 150), thickness=-1)

    else:
        cv2.putText(img, "Unknown Cell Type", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img

# Step 3: Display the most common cell structure under the microscope
def display_microscope_view(file_path):
    cell_type = get_most_common_cell_type(file_path)
    if cell_type:
        print(f"Most common cell type: {cell_type}")
        cell_image = draw_cell_under_microscope(cell_type)
        cv2.imshow("Microscope View - Most Common Cell Structure", cell_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Could not determine the most common cell type.")

# Run the function to display the structure
display_microscope_view('cell_detection_log.csv')
