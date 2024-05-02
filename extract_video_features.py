import os
import sys
import numpy as np
import h5py
import torch
import torchaudio
import random
from moviepy.editor import VideoFileClip

from transformers import VideoMAEImageProcessor, VideoMAEModel, ASTFeatureExtractor, ASTModel

# Set seed for numpy
seed = 42  # You can use any integer value
np.random.seed(seed)

# Set seed for PyTorch
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

def read_video_moviepy(video_clip, indices):
    '''
    Decode the video with MoviePy.
    Args:
        file_path (str): Path to the video file.
        indices (List[int]): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    # Load the video clip

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


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

video_input_folder = "KidsGuard/kidsguard-dataset/video_split/"
audio_input_folder = "KidsGuard/kidsguard-dataset/audio_split/"
labels_folder = "KidsGuard/kidsguard-dataset/labels"

image_processor = VideoMAEImageProcessor()
video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(device)

audio_processor = ASTFeatureExtractor()
audio_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)

videos = sorted(os.listdir(video_input_folder))

with h5py.File('KidsGuard/kidsguard-dataset/kidsguard_dummy.h5', 'a') as hdf5_file:       
    for video in videos:
        if(video in hdf5_file):
            continue
        print("Processing Video:", video)
        video_folder = os.path.join(video_input_folder, video)
        audio_folder = os.path.join(audio_input_folder, video)

        video_segments = os.listdir(video_folder)

        # Extract the numerical part from the file names and convert to integers
        segment_numbers = [int(file.split('_')[1].split('-')[0]) for file in video_segments]

        # Sort the files based on the numerical part
        sorted_segments = [file for _, file in sorted(zip(segment_numbers, video_segments))]

        try:
            label_file = "vid-" + video + ".txt"
            label_file_path = os.path.join(labels_folder, label_file)
            labels = []

            labels_dict = {"none":0, "sexual":1, "violent":2, "both":3}
            # Open the label file for reading
            with open(label_file_path, 'r') as file:
                # Read each line (label) from the file
                for line in file:
                    label = line.split()[-1]  # Remove leading/trailing whitespace (e.g., newline characters)
                    labels.append(label)
            # print(labels)
        except FileNotFoundError as error:
            print("Can't find file:", error)
            error_file_path = 'KidsGuard/kidsguard-dataset/error_videos_1.txt'
            with open(error_file_path, 'a') as file:
                # Write or append content to the file
                file.write(video)
                file.write("\n")
        else:
            # sys.exit()
            embedding_dim = 768

            # Create datasets within the HDF5 file to store embeddings
            video_embeddings_dataset = hdf5_file.create_dataset(f'{video}/video_embeddings', shape=(len(labels), embedding_dim), dtype='f')
            audio_embeddings_dataset = hdf5_file.create_dataset(f'{video}/audio_embeddings', shape=(len(labels), embedding_dim), dtype='f')
            labels_dataset = hdf5_file.create_dataset(f'{video}/labels', shape=(len(labels),), dtype='i')
            
            for i, segment in enumerate(sorted_segments):
                if i >= len(labels):
                    break
                print("i:", i)
                try:
                    labels_dataset[i] = labels_dict[labels[i]]
                except:
                    labels_dataset[i] = -1
                print('Label saved!')

                video_segment_path = os.path.join(video_folder, segment)
                video_clip = VideoFileClip(video_segment_path)

                # sample 16 frames
                indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=int(video_clip.duration * video_clip.fps))
                video_frames = read_video_moviepy(video_clip, indices)

                # prepare video for the model
                inputs = image_processor(list(video_frames), return_tensors="pt").to(device)
                # print(inputs.pixel_values.shape)
                # sys.exit()

                # forward pass
                with torch.no_grad():
                    outputs = video_model(**inputs)
                last_hidden_states = outputs.last_hidden_state
                print(last_hidden_states.shape)
                last_hidden_states = last_hidden_states.mean(dim=1).squeeze()
                video_embeddings_dataset[i] = last_hidden_states.cpu().numpy()
                # print('Video embedding saved!')
                # sys.exit()

                audio_segment_path = os.path.join(audio_folder, segment[:-4] + ".wav")
                # Load stereo audio
                stereo_audio, sample_rate = torchaudio.load(audio_segment_path)
                # Convert stereo to mono
                mono_audio = stereo_audio.mean(dim=0)

                # Resample to 16 kHz
                resample = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                mono_audio_16khz = resample(mono_audio)
                inputs = audio_processor(mono_audio_16khz, sampling_rate=16000, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = audio_model(**inputs)
                last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze()
                # print(type(last_hidden_states))
                audio_embeddings_dataset[i] = last_hidden_states.cpu().numpy()
                print('Audio embedding saved!')

                print(f"{video} {segment} done\n")
            print(f"{video} done!\n")   

print('Done')