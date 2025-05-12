import subprocess
import sys
import importlib

def install_package_from_git(package_name: str, git_url: str):
    """
    Attempts to install a Python package from a Git URL using pip.

    Args:
        package_name (str): The name of the package (for re-import attempt).
        git_url (str): The Git URL for pip installation (e.g., git+https://github.com/user/repo.git).

    Returns:
        bool: True if installation was successful or package already importable, False otherwise.
    """
    try:
        importlib.import_module(package_name)
        print(f"{package_name} package is already available.")
        return True
    except ImportError:
        print(f"{package_name} package not found. Attempting to install from {git_url}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", git_url])
            print(f"{package_name} installation successful!")
            # Verify by trying to import again
            importlib.import_module(package_name)
            print(f"{package_name} is now available.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to install {package_name} from Git. Pip process error: {e}")
        except ImportError:
            print(f"ERROR: Installed {package_name}, but still cannot import it.")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during {package_name} installation: {e}")
        
        print(f"Please try installing {package_name} manually, e.g., by running:")
        print(f"  {sys.executable} -m pip install {git_url}")
        return False

def ensure_oequant_installed(github_url: str = "git+https://github.com/oequant/oequant.git"): # Corrected typo in org name
    """
    Ensures that the 'oequant' package is installed, attempting to install it from GitHub if not found.

    Args:
        github_url (str): The Git URL for oequant.
    
    Returns:
        bool: True if oequant is available/installed, False otherwise.
    """
    return install_package_from_git("oequant", github_url)

if __name__ == '__main__':
    # Example usage (for testing this script directly)
    print("Testing ensure_oequant_installed function...")
    # You might want to point to a test repo or a non-existent package for a full test
    # For now, it will likely say oequant is already available if run from within the project
    if ensure_oequant_installed():
        import oequant as oq
        print("Successfully imported oequant after check.")
        print(f"oequant version (if available): {getattr(oq, '__version__', 'N/A')}")
    else:
        print("oequant could not be made available.") 