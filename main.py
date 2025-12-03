"""
Omni-Qwen: Process VoiceAssistant-400K with Dia2-1B

Main entry point - redirects to the processing script.
"""

import subprocess
import sys


def main():
    """Run the dataset processing script."""
    subprocess.run([sys.executable, "process_dataset.py"] + sys.argv[1:])


if __name__ == "__main__":
    main()
