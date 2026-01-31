# TOE-FRO Data Logger

This project is a Flask-based web application for logging sensor data via UDP and saving it as CSV files. It features a simple web interface for starting/stopping data collection and viewing saved files.

## Features
- UDP listener for real-time sensor data
- Save data to CSV files with customizable filenames
- Web interface for folder selection, starting/stopping logging, and viewing CSV files
- Multiprocessing for robust background data collection
- Cross-platform support (Windows, PyInstaller compatible)

## Requirements
- Python 3.8+
- Flask
- tkinter

## Getting Started
1. Clone the repository:
   ```
   git clone https://github.com/sankar-mechengg/ClubFeet-Anklet.git
   cd ClubFeet-Anklet
   ```
2. (Optional) Create and activate a virtual environment:
   ```
   python -m venv env
   env\Scripts\activate  # On Windows
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
   ```
   python app.py
   ```
5. Select a folder for saving data when prompted.
6. Access the web interface at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## Folder Structure
- `app.py` - Main application file
- `templates/` - HTML templates for Flask
- `build/` - PyInstaller build output
- `env/` - (Optional) Python virtual environment

## Packaging
To create a standalone executable (Windows):
```
pyinstaller --onefile --add-data "templates;templates" app.py
```

## License
This project is for research and educational purposes.
