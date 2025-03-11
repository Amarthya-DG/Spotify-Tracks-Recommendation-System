#!/usr/bin/env python
"""
Run all steps of the Spotify Recommendation System project.
This script will:
1. Process the data
2. Run exploratory data analysis
3. Evaluate recommendation approaches
4. Start the web interface
"""

import os
import subprocess
import sys
import time


def print_section(title):
    """Print a section title with formatting"""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def run_command(command, description, critical=False):
    """Run a command and print its output"""
    print_section(description)
    print(f"Running: {command}\n")

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # Stream output in real-time
        for line in process.stdout:
            print(line, end="")

        process.wait()

        if process.returncode != 0:
            print(f"\nCommand failed with exit code {process.returncode}")
            if critical:
                print("This is a critical step. Exiting.")
                sys.exit(1)
            return False
        return True
    except Exception as e:
        print(f"Error running command: {e}")
        if critical:
            print("This is a critical step. Exiting.")
            sys.exit(1)
        return False


def check_dependencies():
    """Check if all dependencies are installed"""
    print_section("CHECKING DEPENDENCIES")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from datasets import load_dataset
        from flask import Flask
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler

        print("All required dependencies are installed!")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install all dependencies with: pip install -r requirements.txt")
        return False


def main():
    # Check dependencies
    if not check_dependencies():
        choice = (
            input("Do you want to install dependencies now? (y/n): ").strip().lower()
        )
        if choice == "y":
            run_command(
                "pip install -r requirements.txt",
                "INSTALLING DEPENDENCIES",
                critical=True,
            )
        else:
            print("Please install dependencies and try again.")
            return

    # Create required directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Step 1: Process the data
    if not run_command(
        "python src/data_processing.py", "STEP 1: PROCESSING DATA", critical=True
    ):
        print("Data processing failed. Exiting.")
        return

    print("Waiting 2 seconds for file system to catch up...")
    time.sleep(2)

    # Step 2: Run EDA
    if not run_command(
        "python notebooks/Spotify_Tracks_EDA.py",
        "STEP 2: RUNNING EXPLORATORY DATA ANALYSIS",
    ):
        print("EDA failed. You can still continue, but visualizations may be missing.")
        choice = input("Do you want to continue? (y/n): ").strip().lower()
        if choice != "y":
            return

    # Step 3: Evaluate recommendation approaches
    if not run_command(
        "python src/evaluation.py", "STEP 3: EVALUATING RECOMMENDATION APPROACHES"
    ):
        print(
            "Evaluation failed. You may not have information about which recommendation approach is best."
        )
        choice = (
            input("Do you want to continue to the web interface? (y/n): ")
            .strip()
            .lower()
        )
        if choice != "y":
            return

    # Step 4: Start the web interface
    print_section("STEP 4: STARTING WEB INTERFACE")
    print("Starting the web interface at http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")

    try:
        os.system("python src/web_app.py")
    except KeyboardInterrupt:
        print("\nWeb interface stopped by user")

    print_section("ALL STEPS COMPLETED")
    print("The recommendation system has been set up and is ready to use.")
    print("You can now use the following commands:")
    print("- python src/recommend.py - Interactive terminal")
    print("- python src/cli.py --help - CLI interface")
    print("- python src/web_app.py - Web interface")


if __name__ == "__main__":
    main()
