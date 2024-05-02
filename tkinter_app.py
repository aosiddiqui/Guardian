## Tkinter Application 


##-------------------------------------------

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import numpy as np
import torch
import torchaudio
from moviepy.editor import VideoFileClip
from transformers import VideoMAEImageProcessor, VideoMAEModel, ASTFeatureExtractor, ASTModel
import torch.nn as nn
import time
import threading
import matplotlib.pyplot as plt

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to read video frames using MoviePy
def read_video_moviepy(video_clip, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]

    for i, frame in enumerate(video_clip.iter_frames(fps=video_clip.fps)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    video_clip.reader.close()
    return np.stack(frames)

# Function to sample frame indices
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

# Initialize pretrained models
image_processor = VideoMAEImageProcessor()
video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)
audio_processor = ASTFeatureExtractor()
audio_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)

# Load your PyTorch model
class EncoderModel(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=1):
        super().__init__()
        self.video_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.video_encoder = nn.TransformerEncoder(self.video_encoder_layer, num_layers=num_layers)
        self.audio_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.audio_encoder = nn.TransformerEncoder(self.audio_encoder_layer, num_layers=num_layers)

    def forward(self, x, y):
        x = self.video_encoder(x)
        y = self.audio_encoder(y)
        return x, y

class KidsGuardModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.encoder = EncoderModel(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.linear_video = nn.Linear(d_model, 120)
        self.linear_audio = nn.Linear(d_model, 120)
        self.linear1 = nn.Linear(120*120, 1024)
        self.linear2 = nn.Linear(1024, 4)

    def forward(self, x, y):
        x, y = self.encoder(x, y)
        x, y = self.linear_video(x), self.linear_audio(y)
        x, y = x.unsqueeze(2), y.unsqueeze(1)
        input = torch.matmul(x, y)
        output = self.relu(self.linear1(input.view(input.size(0), -1)))
        output = self.linear2(output)
        # output = self.softmax(output)
        return output

# Path to your trained model
model_path = '/home/orchid/IP/KidsGuard/results/wo_pretrain/outer_product_2_layers/best_model.pth'
# Load the PyTorch model
pytorch_model = KidsGuardModel(d_model=768, nhead=8, num_layers=1)
pytorch_model.load_state_dict(torch.load(model_path, map_location=device))

def process_video(video_path, treeview, video_label, progress_label, generate_bar_graph):
    global predictions_table  # Make predictions_table global
    
    try:
        # Input video segment path
        video_clip = VideoFileClip(video_path)

        # Get total duration of the video
        total_duration = int(video_clip.duration)

        # Define clip length and frame sample rate
        clip_len = 16
        frame_sample_rate = 1

        # Display video
        video_label.config(text="Video")
        video_label.update()
        
        # Start a thread to display video
        video_thread = threading.Thread(target=display_video, args=(video_clip, video_label))
        video_thread.start()

        # Initialize predictions table
        predictions_table = []

        # Start progress bar
        progress_label.config(text="Processing video, please wait...")
        progress_label.update()
        progress_bar = ttk.Progressbar(progress_label, orient="horizontal", mode="indeterminate")
        progress_bar.pack()
        progress_bar.start()

        # Loop over 1-second intervals of the video
        for start_time in range(0, total_duration, 1):
            end_time = start_time + 1
            if end_time > total_duration:
                break
            
            # Sample frame indices
            indices = sample_frame_indices(clip_len, frame_sample_rate, int(video_clip.fps * (end_time - start_time)))
            
            # Read video frames
            video_frames = read_video_moviepy(video_clip.subclip(start_time, end_time), indices)
            
            # Prepare video for the model
            inputs = image_processor(list(video_frames), return_tensors="pt").to(device)
            
            # Forward pass for video
            with torch.no_grad():
                outputs = video_model(**inputs)
            last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze()
            video_embedding = last_hidden_states.numpy()  # Changed to CPU compatible data

            # Extract audio from video clip
            audio_clip = video_clip.subclip(start_time, end_time).audio

            # # Generate a random number for the file name
            # random_number = np.random.randint(1000, 9999)

            # Path to save audio
            output_audio_path = f'KidsGuard/saved_audio/audio_{start_time}-{end_time}.wav'

            try:
                # Write audio file
                audio_clip.write_audiofile(output_audio_path, codec="pcm_s32le")
                print("Audio file saved successfully.")

                # Load stereo audio
                stereo_audio, sample_rate = torchaudio.load(output_audio_path)
                # Convert stereo to mono
                mono_audio = stereo_audio.mean(dim=0)
                # Resample to 16 kHz
                resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                mono_audio_16khz = resample(mono_audio)

                # Prepare audio for the model
                inputs = audio_processor(mono_audio_16khz, sampling_rate=16000, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = audio_model(**inputs)
                last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze()
                audio_embedding = last_hidden_states.numpy()  # Changed to CPU compatible data

                # Preprocess embeddings if necessary
                video_embedding_tensor = torch.tensor(video_embedding).unsqueeze(0)  # Assuming batch size 1
                audio_embedding_tensor = torch.tensor(audio_embedding).unsqueeze(0)  # Assuming batch size 1

                # Make predictions
                with torch.no_grad():
                    predictions = pytorch_model(video_embedding_tensor, audio_embedding_tensor)

                # Convert predictions to probabilities
                probabilities = torch.softmax(predictions, dim=1)

                # Get the predicted class
                predicted_class = torch.argmax(probabilities, dim=1).item()

                # Decode class labels
                class_labels = ["None", "Sexual", "Violent", "Both Sexual and Violent"]
                predicted_label = class_labels[predicted_class]

                # Add prediction to the table
                predictions_table.append([f"{start_time} - {end_time}", predicted_label])

                # Insert data into Treeview
                treeview.insert("", "end", values=[f"{start_time} - {end_time}", predicted_label])

            except Exception as e:
                print("Error occurred:", e)

        # Stop progress bar
        progress_label.config(text="Video Processed")
        progress_bar.stop()
        progress_bar.pack_forget()

    except Exception as e:
        print("Error occurred while processing the video:", e)

# Function to generate bar graph
def generate_bar_graph():
    # Show the graph only if predictions_table is available
    if 'predictions_table' in globals():
        # Get data for the bar graph
        label_counts = {"None": 0, "Sexual": 0, "Violent": 0, "Both Sexual and Violent": 0}  # Initialize counts for each label

        # Loop through the predictions table and update label counts
        for interval, label in predictions_table:
            label_counts[label] += 1

        # Extract durations and labels
        durations = list(label_counts.values())
        labels = list(label_counts.keys())

        # Plot bar graph
        plt.figure(figsize=(8, 6))
        plt.bar(labels, durations, color='blue')
        plt.xlabel('Predicted Labels')
        plt.ylabel('Duration in seconds')
        plt.title('Duration of each of the Predicted Labels')
        plt.show()
    else:
        print("Predictions table not available.")

# Function to display video
def display_video(video_clip, video_label):
    for frame in video_clip.iter_frames():
        frame = Image.fromarray(frame)
        frame = frame.resize((400, 225))
        frame = ImageTk.PhotoImage(frame)
        video_label.config(image=frame)
        video_label.image = frame
        video_label.update_idletasks()
        time.sleep(0.03)  # Add a delay of 0.03 seconds (approximately 30 frames per second)

# Function to select video
def select_video():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_video(file_path, treeview, video_label, progress_label, generate_bar_graph)

# Function to exit the application
def exit_app():
    root.destroy()

# Create tkinter app
root = tk.Tk()
root.title("Fine-Grained Child Harmful Content Detection")

# Styling
root.configure(bg="#f0f0f0")  # Set background color
button_style = {"font": ("Arial", 12), "bg": "#007bff", "fg": "white"}  # Button style

# Label for Video Analysis heading
video_analysis_heading = tk.Label(root, text="Fine-Grained Child Harmful Content Detection", font=("Times New Roman", 14, "bold"), bg="#f0f0f0")
video_analysis_heading.pack()

# Create a button to select a video file
select_button = tk.Button(root, text="Select Video", command=select_video, **button_style)
select_button.pack(pady=20)

# Label to display video
video_label = tk.Label(root, text="Video Here", bg="#f0f0f0")
video_label.pack()

# Label to display progress
progress_label = tk.Label(root, text="", bg="#f0f0f0")
progress_label.pack()

# Create a Treeview widget to display predictions in tabular format
columns = ("Time Interval(in sec)", "Prediction")
treeview = ttk.Treeview(root, columns=columns, show="headings")
treeview.heading("Time Interval(in sec)", text="Time Interval(in sec)")
treeview.heading("Prediction", text="Prediction")
treeview.pack(pady=20)

# Bottom navigation bar
bottom_frame = tk.Frame(root, bg="#f0f0f0")
bottom_frame.pack(side="bottom", fill="x")

# # Generate Graph button
# generate_graph_button = tk.Button(bottom_frame, text="Generate Graph", command=generate_bar_graph, **button_style)
# generate_graph_button.pack(side="left", padx=10, pady=5)

# # Exit App button
# exit_button = tk.Button(bottom_frame, text="Exit App", command=exit_app, **button_style)
# exit_button.pack(side="right", padx=10, pady=5)




# Styling
root.configure(bg="#f0f0f0")  # Set background color
button_style = {"font": ("Arial", 12), "fg": "white"}  # Common button style

# Select Video button style
select_button_style = button_style.copy()
select_button_style["bg"] = "blue"  # Blue color for Select Video button

# Generate Graph button style
generate_graph_button_style = button_style.copy()
generate_graph_button_style["bg"] = "blue"  # Blue color for Generate Graph button

# Exit App button style
exit_button_style = button_style.copy()
exit_button_style["bg"] = "red"  # Red color for Exit App button



# Generate Graph button style
generate_graph_button_style = button_style.copy()
generate_graph_button_style["bg"] = "blue"  # Purple color for Generate Graph button



# # Generate Graph button
# generate_graph_button = tk.Button(bottom_frame, text="Generate Graph", command=generate_bar_graph, **button_style)
# generate_graph_button.pack(side="left", padx=10, pady=5)



# Generate Graph button
generate_graph_button = tk.Button(bottom_frame, text="Generate Graph", command=generate_bar_graph, **generate_graph_button_style)
generate_graph_button.pack(side="left", padx=10, pady=5)



# Exit App button
exit_button = tk.Button(bottom_frame, text="Exit App", command=exit_app, **exit_button_style)
exit_button.pack(side="right", padx=10, pady=5)


# Run the Tkinter event loop
root.mainloop()

