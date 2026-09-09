"""Shared test configuration: add src/ to sys.path so tests can import modules."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
