import tkinter as tk
from tkinter import filedialog
from video_processor import process_video
import os

def main():
    # Set up the root Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Open a dialog to ask for the input video file
    video_path = filedialog.askopenfilename(
        title="Select a Pit Stop Video",
        filetypes=(("Video Files", "*.mp4 *.avi *.mov"), ("All files", "*.*"))
    )

    # If the user selected a file, proceed
    if video_path:
        # Suggest a default output filename
        input_dir, input_filename = os.path.split(video_path)
        output_filename = os.path.splitext(input_filename)[0] + "_analyzed.mp4"
        
        # Open a dialog to ask where to save the output video
        output_path = filedialog.asksaveasfilename(
            title="Save Analyzed Video As...",
            initialdir=input_dir,
            initialfile=output_filename,
            defaultextension=".mp4",
            filetypes=(("MP4 files", "*.mp4"),)
        )
        
        # If the user confirmed the save location, start processing
        if output_path:
            print(f"Input video: {video_path}")
            print(f"Output will be saved to: {output_path}")
            print("Processing, please wait...")
            process_video(video_path, output_path)
        else:
            print("Save operation cancelled.")
    else:
        print("No video selected.")

if __name__ == '__main__':
    main()
