#!/bin/bash

# Activate virtual environment if needed
# source venv/bin/activate

# Install required packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run dashboard.py

# Deactivate virtual environment when done
# deactivate
