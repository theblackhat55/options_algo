"""
dashboard/unified_app.py  — P3 FIX #14: canonical file is dashboard/app.py

This file is kept as a compatibility shim so that any bookmarked
`streamlit run dashboard/unified_app.py` invocations continue to work.

Always launch via:
    streamlit run dashboard/app.py
"""
# Re-export everything from the canonical app so that Streamlit can run
# either file interchangeably.
from dashboard.app import *  # noqa: F401, F403
