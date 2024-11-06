import cv2
import pandas as pd
import datetime
import random
from collections import Counter
from Bio.Seq import Seq
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import numpy as np
# Simulated cell types and their corresponding danger levels
CELL_TYPES = ["Normal", "Cancerous", "Bacteria"]
DANGER_LEVELS = {
    "Normal": "Low",
    "Cancerous": "High",
    "Bacteria": "Moderate"
}

# Initialize variables to store results for final conclusion and plotting
all_cell_types = []
all_danger_levels = []
cells_detected_over_time = []
timestamps = []
is_running = True

# Function to preprocess the image (using basic image processing techniques)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    edged = cv2.Canny(blurred, 50, 150)  # Use Canny edge detection
    return edged

# Function to detect contours (potential cells) in the frame
def detect_cells(image):
    processed_image = preprocess_image(image)  # Preprocess the image
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    return contours

# Function to assign a random cell type and corresponding danger level
def assign_cell_type():
    cell_type = random.choice(CELL_TYPES)
    danger_level = DANGER_LEVELS[cell_type]
    return cell_type, danger_level

# BioPython function to simulate DNA analysis
def dna_analysis(cell_type):
    dna_sequences = {
        "Normal": Seq("ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"),
        "Cancerous": Seq("ATGGCGGCTTAGTGAAGCCGCTGAAAGGGTGACCGATAG"),
        "Bacteria": Seq("ATGCCTGCGTACGGCTAGTCAGAGCTGAGCGATCCTG")
    }
    # Get the DNA sequence for the cell type
    dna_seq = dna_sequences[cell_type]
    rev_complement = dna_seq.reverse_complement()  # Get reverse complement of DNA
    return str(dna_seq), str(rev_complement)

# Function to log the detected cells and DNA sequences to a CSV file
def log_data(contours_count, cell_type, danger_level, dna_seq, rev_complement):
    data = {
        'timestamp': [datetime.datetime.now()], 
        'cells_detected': [contours_count],
        'cell_type': [cell_type],
        'danger_level': [danger_level],
        'dna_sequence': [dna_seq],
        'reverse_complement': [rev_complement]
    }
    df = pd.DataFrame(data)
    df.to_csv('cell_detection_log.csv', mode='a', header=False, index=False)  # Append to CSV file

# Function to plot the data after detection
def plot_data():
    if timestamps and cells_detected_over_time:
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, cells_detected_over_time, marker='o', label="Cells Detected")
        plt.title("Cells Detected Over Time")
        plt.xlabel("Time")
        plt.ylabel("Number of Cells Detected")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # Save the plot as an image to be displayed in OpenCV
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_cv
    else:
        return None

# Initialize webcam
cap = cv2.VideoCapture(0)

# Setup the CSV file with headers (run this once at the start)
headers = ['timestamp', 'cells_detected', 'cell_type', 'danger_level', 'dna_sequence', 'reverse_complement']
df = pd.DataFrame(columns=headers)
df.to_csv('cell_detection_log.csv', mode='w', header=True, index=False)

# Main loop for cell detection
while is_running:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect cells in the frame
    contours = detect_cells(frame)
    contours_count = len(contours)  # Count the number of contours (possible cells)

    # Randomly assign a cell type and danger level
    cell_type, danger_level = assign_cell_type()

    # Perform DNA analysis
    dna_seq, rev_complement = dna_analysis(cell_type)

    # Draw the contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # Draw green contours

    # Display the number of detected "cells", cell type, danger level, and DNA
    display_text = f"Cells: {contours_count} | Type: {cell_type} | Danger: {danger_level}"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Show danger level with color coding
    if danger_level == "High":
        danger_color = (0, 0, 255)  # Red for high danger
    elif danger_level == "Moderate":
        danger_color = (0, 255, 255)  # Yellow for moderate danger
    else:
        danger_color = (0, 255, 0)  # Green for low danger

    # Draw a rectangle to indicate the danger level
    cv2.rectangle(frame, (10, 50), (300, 90), danger_color, -1)
    cv2.putText(frame, f"Danger Level: {danger_level}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Instructions for stopping detection
    cv2.putText(frame, "Press 'q' to Stop Detection", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Cell Detection with Danger Level', frame)

    # Log the data (number of detected cells, cell type, danger level, and DNA info)
    log_data(contours_count, cell_type, danger_level, dna_seq, rev_complement)

    # Store data for final conclusion and plotting
    all_cell_types.append(cell_type)
    all_danger_levels.append(danger_level)
    cells_detected_over_time.append(contours_count)
    timestamps.append(datetime.datetime.now().strftime("%H:%M:%S"))

    # Press 'q' to manually quit (besides the button)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        is_running = False

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Final Conclusion after detection stops
def generate_conclusion():
    if all_cell_types:
        most_common_cell_type = Counter(all_cell_types).most_common(1)[0][0]
        most_common_danger_level = Counter(all_danger_levels).most_common(1)[0][0]
        conclusion_text = f"Final Conclusion:\nMost Common Cell Type: {most_common_cell_type}\nMost Common Danger Level: {most_common_danger_level}"
    else:
        conclusion_text = "No cells detected during the session."
    
    print(conclusion_text)

generate_conclusion()

# Show the plot in the output window (console or Jupyter)
plt.figure(figsize=(10, 6))
plt.plot(timestamps, cells_detected_over_time, marker='o', label="Cells Detected")
plt.title("Cells Detected Over Time")
plt.xlabel("Time")
plt.ylabel("Number of Cells Detected")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()

# Display the plot as an image in OpenCV window
plot_image = plot_data()
if plot_image is not None:
    cv2.imshow("Detection Summary", plot_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
