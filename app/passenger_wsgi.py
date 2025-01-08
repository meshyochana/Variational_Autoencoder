import sys
import os

# Set the path to the application directory
# Replace '/path/to/your/application' with the actual path
sys.path.insert(0, os.path.dirname(__file__))

# Import the Flask app
from app import app as application

# Ensure the environment variables are set
os.environ['FLASK_ENV'] = 'production' # or 'development'
