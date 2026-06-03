"""
Primary Streamlit entrypoint for CellQuantX.
Run with: streamlit run app.py
"""

# Current UI remains in QuantX.py; this module gives a stable app entrypoint
# while core logic is split into dedicated modules.
from QuantX import *  # noqa: F401,F403
