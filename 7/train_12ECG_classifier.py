#!/usr/bin/env python
import subprocess

def train_12ECG_classifier(input_directory, output_directory):
  subprocess.run(["python", "main.py", "--input_dir", input_directory, "--output_dir", output_directory])