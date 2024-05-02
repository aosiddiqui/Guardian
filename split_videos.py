from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import os

def split_video_into_segments(input_video_path, video_output_folder, audio_output_folder, segment_duration=1):
    # Load the video
    video_clip = VideoFileClip(input_video_path)
    
    # Create the output folder if it doesn't exist
    os.makedirs(video_output_folder, exist_ok=True)
    os.makedirs(audio_output_folder, exist_ok=True)
    
    # Calculate the total duration of the video in seconds
    total_duration = video_clip.duration
    
    # Split the video into one-second segments
    for start_time in range(0, int(total_duration), segment_duration):
        end_time = min(start_time + segment_duration, total_duration)
        output_segment_path = os.path.join(video_output_folder, f"segment_{start_time}-{end_time}.mp4")
        output_audio_path = os.path.join(audio_output_folder, f"segment_{start_time}-{end_time}.wav")

        if(os.path.exists(output_segment_path) and os.path.exists(output_audio_path)):
            continue
        
        # Use FFmpeg to extract the subclip preserving video and audio
        ffmpeg_extract_subclip(input_video_path, start_time, end_time, targetname=output_segment_path)

        # Save the audio separately as an wav file
        video_clip.subclip(start_time, end_time).audio.write_audiofile(output_audio_path, codec="pcm_s32le")
    
    video = video_output_folder.split("/")[-1]
    print(f"Video {video} split into {int(total_duration/segment_duration)} segments")
    video_clip.close()

if __name__ == "__main__":
    videos_path = "KidsGuard/kidsguard-dataset/videos"
    video_output_folder = "KidsGuard/kidsguard-dataset/video_split/"
    audio_output_folder = "KidsGuard/kidsguard-dataset/audio_split/"
    error_file_path = 'KidsGuard/kidsguard-dataset/error_videos.txt'
    # file1_path = 'KidsGuard/kidsguard-dataset/error_videos_1.txt'
    videos = sorted(os.listdir(videos_path))
    for video in videos:
        video_path = os.path.join(videos_path, video)
        video_output_path = os.path.join(video_output_folder, video[:-4])
        audio_output_path = os.path.join(audio_output_folder, video[:-4])
        # if(os.path.isdir(video_output_path) and os.path.isdir(audio_output_path)):
        #     continue
        try:
            split_video_into_segments(video_path, video_output_path, audio_output_path)
        except:
            with open(error_file_path, 'a') as file:
                # Write data to the file using the write() method
                file.write(video)
                file.write("\n")
    # with open(file1_path, 'r') as file:
    #     for line in file:
    #         # Remove leading and trailing whitespace (e.g., newline characters)
    #         video = line.strip()
    #         for video in videos:
    #             video_path = os.path.join(videos_path, video)
    #             output_path = os.path.join(output_folder, video[:-4])
    #             if(os.path.isdir(output_path)):
    #                 continue
    #             try:
    #                 split_video_into_segments(video_path, output_path)
    #             except:
    #                 with open(file1_path, 'a') as file1:
    #                     # Write data to the file using the write() method
    #                     file1.write(video)

    #                     file1.write("\n")
    print("Done!")