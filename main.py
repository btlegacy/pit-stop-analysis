import streamlit as st
from video_processor import process_video
import tempfile
import os

def main():
    st.title("🏎️ Pit Stop Analysis")

    st.write("""
    Upload a video of a pit stop to analyze the total stop time, 
    tire changing time, and refueling time.
    """)

    # File uploader widget
    uploaded_file = st.file_uploader(
        "Choose a video...", 
        type=["mp4", "mov", "avi"]
    )

    if uploaded_file is not None:
        # Create a temporary directory to store files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Get the path for the uploaded and output files
            input_path = os.path.join(temp_dir, uploaded_file.name)
            output_path = os.path.join(temp_dir, "analyzed_video.mp4")

            # Write the uploaded file to the temp directory
            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display the uploaded video
            st.video(input_path)

            # A button to start the analysis
            if st.button("Analyze Video"):
                # A placeholder and progress bar for user feedback
                with st.spinner("Processing... This may take a moment."):
                    process_video(input_path, output_path)

                st.success("Analysis Complete!")
                
                # Display the processed video
                st.video(output_path)

                # Provide a download button for the analyzed video
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Analyzed Video",
                        data=file,
                        file_name="analyzed_video.mp4",
                        mime="video/mp4"
                    )

if __name__ == '__main__':
    main()
