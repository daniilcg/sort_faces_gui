import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import face_recognition
from tqdm import tqdm
from sklearn.cluster import DBSCAN
import numpy as np

# Supported image formats
SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.heic')

# Function to scan all images in the source folder
def scan_images(source_folder):
    image_paths = []
    for root, _, files in os.walk(source_folder):
        for file in files:
            if file.lower().endswith(SUPPORTED_FORMATS):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Function to extract face encodings from all images
def extract_face_encodings(image_paths):
    encodings = []
    image_data = []

    for path in tqdm(image_paths, desc="Detecting faces"):
        try:
            image = face_recognition.load_image_file(path)
            face_encs = face_recognition.face_encodings(image)

            for face in face_encs:
                encodings.append(face)
                image_data.append((path, face))  # Store path and encoding
        except Exception as e:
            print(f"Error processing {path}: {e}")

    return encodings, image_data

# Function to group similar faces using clustering (DBSCAN)
def cluster_faces(encodings):
    if not encodings:
        return []

    clustering = DBSCAN(metric='euclidean', n_jobs=-1, eps=0.6, min_samples=1)
    labels = clustering.fit_predict(encodings)
    return labels

# Function to copy images into folders by person
def sort_and_copy(image_data, labels, output_folder):
    clusters = {}

    # Group image paths by label
    for (path, _), label in zip(image_data, labels):
        clusters.setdefault(label, set()).add(path)

    # Copy each image to the corresponding folder
    for label, paths in tqdm(clusters.items(), desc="Copying sorted photos"):
        person_folder = os.path.join(output_folder, f"Person_{label + 1}")
        os.makedirs(person_folder, exist_ok=True)

        for img_path in paths:
            try:
                filename = os.path.basename(img_path)
                shutil.copy(img_path, os.path.join(person_folder, filename))
            except Exception as e:
                print(f"Failed to copy {img_path}: {e}")

# GUI: Open folder selector
def select_folder(title):
    return filedialog.askdirectory(title=title)

# GUI: Start button callback
def start_sorting():
    source = select_folder("Select source folder with photos")
    if not source:
        return

    target = select_folder("Select output folder")
    if not target:
        return

    image_paths = scan_images(source)
    if not image_paths:
        messagebox.showerror("Error", "No supported image files found.")
        return

    encodings, image_data = extract_face_encodings(image_paths)
    if not encodings:
        messagebox.showinfo("Done", "No faces were detected.")
        return

    labels = cluster_faces(encodings)
    sort_and_copy(image_data, labels, target)

    messagebox.showinfo("Done", f"Photos sorted into {len(set(labels))} people folders.")

# Main GUI setup
def create_gui():
    root = tk.Tk()
    root.title("Face Sorter")
    root.geometry("400x300")
    root.resizable(False, False)

    # Logo or icon (optional)
    label = tk.Label(root, text="📸 Face Sorter", font=("Arial", 18))
    label.pack(pady=30)

    btn = tk.Button(root, text="Start Sorting Faces", command=start_sorting, font=("Arial", 14), bg="#4CAF50", fg="white", padx=20, pady=10)
    btn.pack(pady=20)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
