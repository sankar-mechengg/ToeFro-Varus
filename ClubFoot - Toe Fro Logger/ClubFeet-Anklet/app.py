import os
import sys
import socket
import csv
import time
import webbrowser
from datetime import datetime
from threading import Timer
from multiprocessing import Process, Value, freeze_support
import tkinter as tk
from tkinter import filedialog
from flask import Flask, render_template, request, jsonify

# --- Part 1: PyInstaller Path Helper ---
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Part 2: Folder Selection (moved inside main block) ---
def select_initial_folder():
    """Opens a native dialog to select the initial data folder."""
    print("Opening folder selection dialog...")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True) # Ensure the dialog appears on top
    folder_selected = filedialog.askdirectory(
        initialdir=os.getcwd(),
        title="Please select the folder to save data files"
    )
    root.destroy()
    if not folder_selected:
        print("No folder selected. Exiting.")
        sys.exit()
    print(f"Folder selected: {folder_selected}")
    return folder_selected

# --- Flask App Initialization ---
template_dir = resource_path('templates')
app = Flask(__name__, template_folder=template_dir)

# Define a global variable for the folder path
DEFAULT_FOLDER_PATH = ""

# --- Global variables for multiprocessing ---
listener_process = None
is_running = Value('b', False)
last_packet_time = Value('d', 0.0)

# --- UDP Configuration ---
LISTEN_IP = ''
LISTEN_PORT = 2390
CSV_HEADER = [
    "DateTime", "Timestamp_ms", "Gyro_X", "Gyro_Y", "Gyro_Z",
    "LinearAccel_X", "LinearAccel_Y", "LinearAccel_Z",
    "Magnetometer_X", "Magnetometer_Y", "Magnetometer_Z",
    "RawAccel_X", "RawAccel_Y", "RawAccel_Z", "Quat_W", "Quat_X", "Quat_Y", "Quat_Z",
    "Cal_System", "Cal_Gyro", "Cal_Accel", "Cal_Mag"
]

def parse_bno_data(data_string):
    parsed_data = {}
    sections = data_string.split('\t')
    for section in sections:
        parts = section.strip().split(' ')
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                try: parsed_data[key.strip()] = float(value.strip())
                except ValueError:
                    try: parsed_data[key.strip()] = int(value.strip())
                    except ValueError: parsed_data[key.strip()] = value.strip()
    return parsed_data

def get_row_from_parsed_data(parsed_data):
    row = []
    ordered_keys = [
        "ms", "avgx", "avgy", "avgz", "lax", "lay", "laz", "mx", "my", "mz",
        "rax", "ray", "raz", "qw", "qx", "qy", "qz", "calsys", "calgyro", "calaccel", "calmag"
    ]
    for key in ordered_keys: row.append(parsed_data.get(key, 'N/A'))
    return row

def udp_listener_and_logger(is_running_flag, last_packet_time_val, folder_path, participant_name, attribute):
    """
    This function runs in a separate process to listen for UDP packets and log them.
    """
    filename = f"{participant_name}_{attribute}.csv"
    output_filename = os.path.join(folder_path, filename)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    try:
        sock.bind((LISTEN_IP, LISTEN_PORT))
        print(f"SUCCESS: UDP listener bound to {LISTEN_IP or '0.0.0.0'}:{LISTEN_PORT}. Waiting for data...")
        
        with open(output_filename, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(CSV_HEADER)
            print(f"Logging data to: {output_filename}")
            
            while is_running_flag.value:
                try:
                    sock.settimeout(1.0) # Use a timeout to remain non-blocking
                    data, addr = sock.recvfrom(1024)
                    
                    last_packet_time_val.value = time.time()
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    decoded_data = data.decode('utf-8')
                    
                    parsed_sensor_data = parse_bno_data(decoded_data)
                    sensor_data_row = get_row_from_parsed_data(parsed_sensor_data)
                    csv_row = [current_time] + sensor_data_row
                    
                    csv_writer.writerow(csv_row)
                    csv_file.flush() # Ensure data is written immediately
                except socket.timeout:
                    # This is expected. It allows the loop to check is_running_flag.
                    continue
                except Exception as e:
                    print(f"An error occurred in the listener loop: {e}")
    except socket.error as e:
        print(f"FATAL: Socket error in listener process: {e}")
        print("This might be due to the port being in use or a firewall issue.")
    finally:
        sock.close()
        print("UDP listener stopped and socket closed.")

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', default_folder=DEFAULT_FOLDER_PATH)

@app.route('/start', methods=['POST'])
def start_listener():
    global listener_process, is_running, last_packet_time
    if listener_process and listener_process.is_alive():
        return jsonify({'status': 'already_running'})
        
    data = request.get_json()
    folder_path = data.get('folderPath')
    participant_name = data.get('participantName')
    attribute = data.get('attribute')

    if not folder_path or not participant_name or not attribute:
        return jsonify({'status': 'error', 'message': 'Folder path, participant name, and attribute are required.'})

    # Robustness: Ensure the directory exists before starting the process
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as e:
        return jsonify({'status': 'error', 'message': f'Could not create directory: {e}'})

    is_running.value = True
    last_packet_time.value = 0.0
    
    listener_process = Process(target=udp_listener_and_logger, args=(is_running, last_packet_time, folder_path, participant_name, attribute))
    listener_process.start()
    
    print(f"Started listener process (PID: {listener_process.pid}) for {participant_name}_{attribute}.csv")
    return jsonify({'status': 'started', 'filename': f"{participant_name}_{attribute}.csv"})

@app.route('/stop', methods=['POST'])
def stop_listener():
    global listener_process, is_running
    if listener_process and listener_process.is_alive():
        print("Stopping listener process...")
        is_running.value = False
        listener_process.join(timeout=3) # Wait for 3 seconds for graceful shutdown
        if listener_process.is_alive():
            print("Process did not join gracefully, terminating.")
            listener_process.terminate() # Forcefully terminate if stuck
        listener_process = None
        print("Listener process stopped.")
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not_running'})

@app.route('/check_status', methods=['GET'])
def check_status():
    if not (listener_process and listener_process.is_alive()):
        return jsonify({'status': 'off'})
    if last_packet_time.value == 0.0:
        return jsonify({'status': 'waiting'}) # Actively listening but no data yet
    if (time.time() - last_packet_time.value) > 3.0:
        return jsonify({'status': 'disconnected'}) # Was streaming, but signal lost
    return jsonify({'status': 'streaming'})

@app.route('/get_folder_contents', methods=['POST'])
def get_folder_contents():
    folder_path = request.get_json().get('folderPath')
    if not folder_path or not os.path.isdir(folder_path):
        return jsonify({'error': 'Invalid or missing folder path.'}), 400
    try:
        contents = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        return jsonify({'contents': sorted(contents)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Part 3: Main Execution Block ---
def open_browser():
    """Opens the default web browser to the application's URL."""
    webbrowser.open_new("http://127.0.0.1:5000")

if __name__ == '__main__':
    # This MUST be the first line for PyInstaller on Windows
    freeze_support()
    
    # Get the folder path from the user before starting the server
    DEFAULT_FOLDER_PATH = select_initial_folder()
    
    # Open the browser shortly after the app starts
    Timer(1.5, open_browser).start()
    
    print("Starting Flask web server...")
    # Use debug=False for the final executable
    app.run(host='127.0.0.1', port=5000, debug=False)