import subprocess
import sys
import os
import importlib.metadata

requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")

def get_installed_packages():
    return {dist.metadata['Name'].lower() for dist in importlib.metadata.distributions()}

def install_missing_packages(requirements_file):
    with open(requirements_file) as f:
        required = f.read().splitlines()

    installed = get_installed_packages()
    missing = [pkg for pkg in required if pkg.split("==")[0].lower() not in installed]

    if missing:
        print(f"Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("All packages already installed.")

if __name__ == "__main__":
    install_missing_packages(requirements_path)
