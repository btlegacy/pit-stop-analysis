import streamlit as st
import tempfile
import os
import io
import csv
from video_processor import process_video

st.set_page_config(page_title="Pit Stop Analyzer", layout="wide")

def save_uploaded_file(uploaded_file):
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path, tmp_dir

def make_csv_bytes(results_dict):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(list(results_dict.keys()))
    writer.writerow([f"{v:.2f}" for v in results_dict.values()])
    return buf.getvalue().encode('utf-8')

st.title("Pit Stop Analyzer")

uploaded = st.file_uploader("Upload video (mp4/mov)", type=["mp4", "mov", "avi", "mkv"])
output_name = st.text_input("Output filename (optional)", value="annotated_output.mp4")

col1, col2 = st.columns([1, 3])
with col1:
    analyze_button = st.button("Analyze Video")
    st.markdown(
        """
        Tips:
        - Provide refs/crew templates and refs/annotations.csv to seed crew recognition.
        - The processor will write an annotated output video and return timings.
        """
    )

with col2:
    progress_bar = st.progress(0)
    progress_text = st.empty()

# progress callback passed into process_video
def update_progress(fraction):
    try:
        percent = int(max(0.0, min(1.0, fraction)) * 100)
        progress_bar.progress(percent)
        progress_text.text(f"Processing... {percent}%")
    except Exception:
        pass

if analyze_button:
    if uploaded is None:
        st.warning("Please upload a video first.")
    else:
        # Save upload to temp file
        input_path, tmp_dir = save_uploaded_file(uploaded)
        output_path = os.path.join(tmp_dir, output_name)

        st.info("Starting analysis — this can take a few minutes depending on video length and model speed.")
        try:
            result = process_video(input_path, output_path, update_progress)
            # ensure progress shows complete
            progress_bar.progress(100)
            progress_text.text("Processing complete")

            # result may be a tuple of 3 (old) or 5 (new) values
            total_stopped_time = tire_change_time = refuel_time = 0.0
            front_tire_time = rear_tire_time = 0.0

            if isinstance(result, (list, tuple)):
                if len(result) >= 5:
                    total_stopped_time, tire_change_time, refuel_time, front_tire_time, rear_tire_time = result[:5]
                elif len(result) == 3:
                    total_stopped_time, tire_change_time, refuel_time = result
                else:
                    st.error(f"Unexpected result shape from process_video: {result}")
            else:
                st.error("process_video did not return expected result tuple.")

            st.success("Analysis Complete!")

            # Display results
            st.markdown("### Results")
            st.write(f"Car stopped total time: {total_stopped_time:.2f}s")
            st.write(f"Total tire change time (all): {tire_change_time:.2f}s")
            st.write(f"Front tire change time: {front_tire_time:.2f}s")
            st.write(f"Rear tire change time: {rear_tire_time:.2f}s")
            st.write(f"Refuel time: {refuel_time:.2f}s")

            # Provide annotated video for playback / download
            if os.path.exists(output_path):
                st.video(output_path)
                with open(output_path, "rb") as f:
                    btn = st.download_button(
                        label="Download annotated video",
                        data=f,
                        file_name=os.path.basename(output_path),
                        mime="video/mp4"
                    )

            # CSV download of results
            results_dict = {
                "total_stopped_time": total_stopped_time,
                "tire_change_time": tire_change_time,
                "front_tire_time": front_tire_time,
                "rear_tire_time": rear_tire_time,
                "refuel_time": refuel_time,
            }
            csv_bytes = make_csv_bytes(results_dict)
            st.download_button("Download results (CSV)", data=csv_bytes, file_name="pit_stop_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error during processing: {e}")
        finally:
            # cleanup temp files optionally
            try:
                if os.path.exists(input_path):
                    os.remove(input_path)
                if os.path.exists(output_path):
                    # keep output if you want; comment out to remove
                    pass
            except Exception:
                pass
