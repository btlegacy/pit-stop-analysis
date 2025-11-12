import argparse
from video_processor import process_video

def main():
    parser = argparse.ArgumentParser(description='Analyze a pit stop video.')
    parser.add_argument('video_path', type=str, help='Path to the input video file.')
    parser.add_argument('--output_path', type=str, default='output.mp4', help='Path to save the output video file.')
    args = parser.parse_args()

    process_video(args.video_path, args.output_path)

if __name__ == '__main__':
    main()