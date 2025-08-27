import subprocess
import pkg_resources
import sys
import os

# Path to your requirements.txt
requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

def install_missing_packages(requirements_file):
    # Read required packages
    with open(requirements_file) as f:
        required = f.read().splitlines()

    # Get installed packages
    installed = {pkg.key for pkg in pkg_resources.working_set}

    # Find missing packages
    missing = [pkg for pkg in required if pkg.split("==")[0].lower() not in installed]

    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("All packages already installed.")

if __name__ == "__main__":
    install_missing_packages(requirements_path)
