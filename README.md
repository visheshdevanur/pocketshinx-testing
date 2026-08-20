# PocketSphinx STT Evaluation Harness

A standalone, offline Speech-to-Text testing and evaluation application built with Python and Streamlit. It uses PocketSphinx for completely offline transcription and allows you to test Word Error Rate (WER) and accuracy in real-time.

## Prerequisites
To run this application on any device, you must have **Python** installed (Python 3.8 to 3.11 is recommended). 

## Setup Instructions

1. **Extract the files:**
   Unzip the folder you transferred onto the new device.

2. **Open a terminal or command prompt:**
   Navigate into the unzipped folder. For example:
   ```bash
   cd path/to/pocket
   ```

3. **Install the dependencies:**
   Run the following command to install all the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Depending on your Python installation, you might need to use `pip3` instead of `pip`).*

4. **Run the application:**
   Once everything is installed, start the Streamlit web server by running:
   ```bash
   python -m streamlit run app.py
   ```

5. **Open the app:**
   The application should automatically open in your default web browser (usually at `http://localhost:8501`). Everything is run locally on your machine, requiring no internet connection for the speech recognition!
