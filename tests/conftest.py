import os
import sys

# The pipeline scripts import their helpers from the scripts directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AutoDeep", "scripts"))
