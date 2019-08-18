#! /venv/bin/python3.5

import logging
import sys
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0, '/home/kefentse/InstaAPi/')
from app import app as application
application.secret_key = 'random_key'
