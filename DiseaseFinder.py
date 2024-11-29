# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:04:49 2024

@author: Legoboy
"""

import cv2
import pandas as pd
import datetime
import random
from Bio.Seq import Seq
from collections import Counter

# Database of microbes and diseases
MICROBES_DATABASE = {
    "Streptococcus": {"Disease": "Strep Throat", "Symptoms": "Sore throat, fever", "Risk": "Moderate"},
    "Candida": {"Disease": "Oral Thrush", "Symptoms": "White patches in mouth, discomfort", "Risk": "Moderate"},
    "H. pylori": {"Disease": "Peptic Ulcer", "Symptoms": "Stomach pain, nausea", "Risk": "High"},
    "E. coli": {"Disease": "Food Poisoning", "Symptoms": "Diarrhea, cramps", "Risk": "High"},
    "Normal Flora": {"Disease": "None", "Symptoms": "Healthy microbiota", "Risk": "Low"}
}

# Initialize variables
all_cell_types = []
all_diseases = []
all_risks = []
timestamps = []
cells_detected_over_time = []

# Function to preprocess the image
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged

# Function to detect contours (potential cells) in the frame
def detect_cells(image):
    processed_image = preprocess_image(image)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Function to identify disease based on microbe
def identify_microbe(cell_type):
    if cell_type in MICROBES_DATABASE:
        microbe_info = MICROBES_DATABASE[cell_type]
        return (
            microbe_info["Disease"],
            microbe_info["Symptoms"],
            microbe_info["Risk"]
        )
    return "Unknown", "Unknown symptoms", "Unknown risk"

# Function to simulate DNA analysis
def dna_analysis(microbe):
    dna_sequences = {
        "Streptococcus": Seq("ATGCCATTAGTGCTAGCTGCTGCTGA"),
        "Candida": Seq("ATGCGTACCGATCGTAGCTAGCTAGT"),
        "H. pylori": Seq("ATGGCCATTGTAATGGGCCGCTGAAA"),
        "E. coli": Seq("ATGCCTGCGTACGGCTAGTCAGAGCT"),
        "Normal Flora": Seq("ATGCCTGCGTACGGCTAGTCAGAGCT")
    }
    dna_seq = dna_sequences.get(microbe, Seq(""))
    rev_complement = dna_seq.reverse_complement()
    return str(dna_seq), str(rev_complement)

# Function to log data to a CSV file
def log_data(cell_count, microbe, disease, symptoms, risk, dna_seq, rev_complement):
    data = {
        "timestamp": [datetime.datetime.now()],
        "cells_detected": [cell_count],
        "microbe": [microbe],
        "disease": [disease],
        "symptoms": [symptoms],
        "risk": [risk],
        "dna_sequence": [dna_seq],
        "reverse_complement": [rev_complement]
    }
    df = pd.DataFrame(data)
    df.to_csv("disease_detection_log.csv", mode="a", header=False, index=False)

# Function to process detection and generate a report
def detect_and_report(frame, contours):
    cell_count = len(contours)
    cell_type = random.choice(list(MICROBES_DATABASE.keys()))  # Simulate detected microbe
    disease, symptoms, risk = identify_microbe(cell_type)
    dna_seq, rev_complement = dna_analysis(cell_type)

    # Display the information
    display_text = f"Cells: {cell_count} | Microbe: {cell_type} | Disease: {disease}"
    risk_text = f"Symptoms: {symptoms} | Risk: {risk}"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, risk_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Log the data
    log_data(cell_count, cell_type, disease, symptoms, risk, dna_seq, rev_complement)

    # Store data for final summary
    all_cell_types.append(cell_type)
    all_diseases.append(disease)
    all_risks.append(risk)
    cells_detected_over_time.append(cell_count)
    timestamps.append(datetime.datetime.now().strftime("%H:%M:%S"))

# Initialize webcam
cap = cv2.VideoCapture(0)

# Setup CSV file with headers
headers = ["timestamp", "cells_detected", "microbe", "disease", "symptoms", "risk", "dna_sequence", "reverse_complement"]
df = pd.DataFrame(columns=headers)
df.to_csv("disease_detection_log.csv", mode="w", header=True, index=False)

# Start detection loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    contours = detect_cells(frame)
    detect_and_report(frame, contours)

    # Display the frame
    cv2.imshow("Disease Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Final Summary
if all_cell_types:
    most_common_microbe = Counter(all_cell_types).most_common(1)[0][0]
    most_common_disease = Counter(all_diseases).most_common(1)[0][0]
    most_common_risk = Counter(all_risks).most_common(1)[0][0]
    print(f"Final Report:\nMost Common Microbe: {most_common_microbe}\nDisease: {most_common_disease}\nRisk: {most_common_risk}")
else:
    print("No microbes detected.")
