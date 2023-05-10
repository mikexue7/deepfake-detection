import numpy as np
import cv2
import os
import json
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/michaelxue/anaconda3/lib/python3.9/site-packages/ffmpeg/"
# from moviepy.editor import AudioFileClip

TRAIN_DATA_DIRECTORY = "./train_sample_videos/"

def load_frames(filepath):
    # Open the video file
    cap = cv2.VideoCapture(filepath)
    frames = []

    # Loop through the video frames
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If there are no more frames, break out of the loop
        if not ret:
            break

        frames.append(np.expand_dims(frame, axis=0))

    # Release the video file
    cap.release()

    return np.concatenate(frames, axis=0)

def load_data_and_labels(directory):
    data = []
    labels = []
    for i, filename in enumerate(os.listdir(directory)):
        if i > 3: # save memory locally
            break
        if filename.endswith('.mp4'): 
            filepath = os.path.join(directory, filename)
            frames = load_frames(filepath)
            data.append(frames) # we will need to pad these later before passing into models
    # load metadata info
    metadata_path = os.path.join(directory, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        for d in metadata.values():
            label = d["label"]
            labels.append(1 if label == 'FAKE' else 0)
    
    return data, np.asarray(labels)

# def load_audio(filepath):
#     # Load the audio file
#     audio_clip = AudioFileClip(filepath)

#     # Extract the audio data
#     audio_data = audio_clip.to_soundarray()

#     # Print the shape of the audio data array
#     print(audio_data.shape)

#     # Release the audio file
#     audio_clip.close()

# WE MAY NEED TO DOWNSAMPLE VIDEOS, RIGHT NOW THEY ARE HD AND TAKE UP A LOT OF SPACE
# AND/OR WE CAN TAKE A SUBSET OF FRAMES PER VIDEO
if __name__ == '__main__':
    train_data, labels = load_data_and_labels(TRAIN_DATA_DIRECTORY)
    print(len(train_data), labels.shape)
