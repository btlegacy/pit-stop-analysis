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

    uploaded_file = st.file_uploader(
        "Choose a video...", 
        type=["mp4", "mov", "avi"]
    )

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, uploaded_file.name)
            output_path = os.path.join(temp_dir, "analyzed_video.mp4")

            with open(input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.video(input_path)

            if st.button("Analyze Video"):
                progress_bar = st.progress(0, text="Starting analysis...")

                def update_progress(percentage):
                    progress_bar.progress(int(percentage * 100), text=f"Processing... {int(percentage * 100)}%")

                total_stopped_time, tire_change_time, refuel_time = process_video(input_path, output_path, update_progress)
                
                progress_bar.progress(100, text="Analysis Complete!")
                st.success("Analysis Complete!")

                # Display metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Stop Time", f"{total_stopped_time:.2f}s")
                col2.metric("Tire Change Time", f"{tire_change_time:.2f}s")
                col3.metric("Refueling Time", f"{refuel_time:.2f}s")

                st.video(output_path)

                with open(output_path, "rb") as file:
                    st.download_button(
                        label="Download Analyzed Video",
                        data=file,
                        file_name="analyzed_video.mp4",
                        mime="video/mp4"
                    )

if __name__ == '__main__':
    main()
