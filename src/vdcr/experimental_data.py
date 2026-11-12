"""Locate the current experimental assessment and its archived predecessors."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]

def assessment_directory():
    index=ROOT/'inputs/experimental_assessment.json'
    if index.exists():
        return ROOT/json.loads(index.read_text())['results_directory']
    return ROOT/'results/reference'

def assessment_file(name):
    return assessment_directory()/name
