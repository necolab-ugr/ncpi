import tempfile
import os

# Folder for temporary files > 5 GB
# Set BEFORE any flask imports if possible
tempfile.tempdir = '/home/necolab/tmp'
os.environ['TMPDIR'] = '/home/necolab/tmp'

# Temporary folder for uploaded files of forms
temp_uploaded_files = 'temp_uploaded_files'

from flask import Flask, render_template, request, jsonify, url_for, redirect, send_file, after_this_request, flash
from concurrent.futures import ThreadPoolExecutor
import uuid
import time
import io
import pandas as pd
import numpy as np
import compute_utils

# Main app object
app = Flask(__name__)

# Set secret key for sessions (necessary to show alert messages)
app.secret_key = '602e6444-80b2-431c-b26c-b6cda2ac9c09'

# In-memory thread pool
executor = ThreadPoolExecutor(max_workers=5) 

# Dictionary to store job progress/results (job_id: status_dict)
# NOTE: This dictionary is volatile and will reset if the server restarts.
job_status = {}


# Main dashboard page loading
@app.route("/")
def dashboard():
    return render_template("0.dashboard.html")

# Simulation configuration page
@app.route("/simulation")
def simulation():
    return render_template("1.simulation.html")

@app.route("/upload_sim")
def upload_sim():
    return render_template("1.1.upload_sim.html")

@app.route("/new_sim")
def new_sim():
    return render_template("1.2.0.new_sim.html")

@app.route("/new_sim_brunel")
def new_sim_brunel():
    return render_template("1.2.1.new_sim_brunel.html")

@app.route("/new_sim_custom")
def new_sim_custom():
    return render_template("1.2.2.new_sim_custom.html")

# Field potential configuration page
@app.route("/field_potential")
def field_potential():
    return render_template("2.field_potential.html")

# Features configuration page
@app.route("/features", methods=["GET", "POST"])
def features():
    return render_template("3.features.html")

# Inference configuration page
@app.route("/inference")
def inference():
    return render_template("4.inference.html")

# Analysis configuration page
@app.route("/analysis")
def analysis():
    return render_template("5.analysis.html")


@app.route("/start_computation_redirect/<computation_type>", methods=["POST"])
def start_computation_redirect(computation_type):
    """Starts the background job and redirects to the status page."""
    # Allowed function names to redirect to
    allowed_functions = {'features', 'inference', 'analysis'}

    if computation_type not in allowed_functions:
        return f"Type of computation is not valid", 400

    # Build the name of the function to compute depending on the page form this function was called from
    func_name_string = f"{computation_type}_computation"
    func = getattr(compute_utils, func_name_string) # filtered function name for security reasons

    # Get all files from the request
    files = request.files
    
    # Check how many files were uploaded with actual filenames
    uploaded_files = [f for f in files.values() if f.filename]

    # First check if ANY files were uploaded. If at least one file was uploaded
    if len(uploaded_files) == 0:
        # Flash an error message
        flash('No files uploaded, please try again.', 'error')
        return redirect(request.referrer)

    # File filter and checks for every computation type
    if computation_type == 'features':
        # Expect exactly 1 file
        if len(uploaded_files) != 1:
            flash('This computation requires exactly 1 file.', 'error')
            return redirect(request.referrer or url_for('features'))
        estimated_time_remaining = time.time() + 68 # 68 seconds of estimated time remaining

    if computation_type == 'inference':
        # Expect 5 files
        if len(uploaded_files) != 5 and len(uploaded_files) != 1:
            flash('This computation requires you to upload all 5 files or only the features prediction file.', 'error')
            return redirect(request.referrer or url_for('inference'))
        estimated_time_remaining = time.time() + 130 # 130 seconds of estimated time remaining

    if computation_type == 'analysis':
        estimated_time_remaining = time.time() + 10 # 15 seconds of estimated time remaining

    # Unique id for job
    job_id = str(uuid.uuid4())

    # If everything is OK, save the file(s)
    file_paths = {}
    for i, file_key in enumerate(request.files):
        file = request.files[file_key]
        unique_filename = f"{computation_type}_{file_key}_{i}_{job_id}_{file.filename}" # E.g. features_ data_file_ 0_ 444961cc-5b72-43fc-b87e-3f4c8304ecdd_ df_inputIn_features_lfp.pkl
        file_path = os.path.join(temp_uploaded_files, unique_filename)
        # Ensure directory exists
        os.makedirs(temp_uploaded_files, exist_ok=True)
        file.save(file_path)
        # Save dictionary with file_key: file_path
        file_paths[file_key] = file_path

    data = request.form.to_dict() # Get parameters from form POST
    # Add file information to the data dictionary
    data['file_paths'] = file_paths

    # Store initial status
    job_status[job_id] = {
        "status": "in_progress",
        "progress": 0,
        "start_time": time.time(),
        "estimated_time_remaining": estimated_time_remaining,
        "results": None,
        "error": False
    }

    # Submit the long-running task according to the computation type
    executor.submit(func, job_id, job_status, data, temp_uploaded_files)

    # Redirect immediately to the loading page (PRG pattern)
    return redirect(url_for('job_status_page', job_id=job_id, computation_type=computation_type))

@app.route("/job_status/<job_id>")
def job_status_page(job_id):
    """Renders the loading page that begins polling."""
    # Get computation_type from the query parameters
    computation_type = request.args.get('computation_type') 
    # Pass the job_id to the template for use in Alpine.js
    return render_template("loading_page.html", job_id=job_id, computation_type=computation_type)

@app.route("/status/<job_id>")
def get_status(job_id):
    """AJAX endpoint for the client to poll for status updates."""
    status = job_status.get(job_id)
    if not status:
        return jsonify({
            "status": "failed", 
            "error": "Job not found"
        }), 404

    # Calculate progress based on time elapsed
    elapsed = time.time() - status["start_time"]
    total_estimated = status["estimated_time_remaining"] - status["start_time"]

    # Progress as percentage (0-100), capped at 99 until finished
    if status["status"] == "in_progress":
        progress = min(99, int((elapsed / total_estimated) * 100))
    elif status["status"] == "finished":
        progress = 100
    else:
        progress = status.get("progress", 0)

    # Update the progress in job_status
    status["progress"] = progress
    
    return jsonify({
        "status": status["status"],
        "progress": status["progress"],
        "elapsed_time": int(time.time() - status["start_time"]),
        "estimated_time_remaining": status["estimated_time_remaining"],
        "error": False
    })


@app.route("/download_results/<job_id>")
def download_results(job_id):
    """Handles the download of the final Pandas DataFrame."""
    status = job_status.get(job_id)

    # Get computation_type from the query parameters
    computation_type = request.args.get('computation_type') 

    if not status or status["status"] != "finished" or status["results"] is None:
        return "Results not available or computation incomplete.", 404

    # Retrieve the stored DataFrame
    output_df_path = status["results"] 

    # Remove file after downloading it
    @after_this_request
    def cleanup(response):
        try:
            if os.path.exists(output_df_path):
                os.remove(output_df_path)
                app.logger.info(f"Cleaned up {output_df_path}")
        except Exception as e:
            app.logger.error(f"Error removing file {output_df_path}: {e}")
        return response

    if computation_type != 'analysis':
        output_df = compute_utils.read_df_file(output_df_path)

        # Create an in-memory byte stream (io.BytesIO)
        output = io.BytesIO()
        # Save the DataFrame to Pickle in the in-memory stream
        output_df.to_pickle(output)
        output.seek(0)

        # Use send_file to trigger the download
        return send_file(
            output,
            mimetype='application/python-pickle',
            as_attachment=True,
            download_name=f'{computation_type}_results_{job_id}_output.pkl'
        )

    else: # computation_type == 'analysis'
        return send_file(
            f'{temp_uploaded_files}/LFP_predictions.png',
            mimetype='image/png',
            as_attachment=True,
            download_name='LFP_predictions.png'
        )