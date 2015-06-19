# Simple setup file to install the virtual environment
# N.B.: Does not perform any checks to see if it is already there!

virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt