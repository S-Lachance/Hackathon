"""
Helper script to download pretrained models from Google Drive.
"""

import os
import sys

def download_with_gdown():
    """Try downloading with gdown package."""
    try:
        import gdown
        print("✓ Using gdown package to download models...")
        
        # Google Drive file IDs (these would need to be extracted from the shared folder)
        # Since the link is to a folder, we'll need direct file links
        
        folder_url = "https://drive.google.com/drive/folders/13r3tEXg5SfyesCLX0w76K5fJ7MHmbJes"
        
        print("\n⚠ The provided link is to a Google Drive folder.")
        print("Please visit the following URL and download the files manually:")
        print(folder_url)
        print("\nRequired files:")
        print("  - random_forest_synthetic.joblib")
        print("  - random_forest_original.joblib")
        
        return False
        
    except ImportError:
        print("✗ gdown not installed. Install with: pip install gdown")
        return False


def check_existing_models():
    """Check if models already exist."""
    synthetic_exists = os.path.exists("random_forest_synthetic.joblib")
    original_exists = os.path.exists("random_forest_original.joblib")
    
    if synthetic_exists and original_exists:
        print("✓ Both models already exist!")
        return True
    elif synthetic_exists:
        print("⚠ Only random_forest_synthetic.joblib exists")
        print("  Missing: random_forest_original.joblib")
        return False
    elif original_exists:
        print("⚠ Only random_forest_original.joblib exists")
        print("  Missing: random_forest_synthetic.joblib")
        return False
    else:
        print("✗ No models found")
        return False


def main():
    """Main download function."""
    print("\n" + "="*80)
    print("PRETRAINED MODEL DOWNLOADER")
    print("="*80)
    
    # Check if models exist
    if check_existing_models():
        return
    
    print("\nAttempting to download models...")
    
    # Try gdown
    if not download_with_gdown():
        print("\n" + "="*80)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*80)
        print("\nPlease follow these steps:")
        print("1. Visit: https://drive.google.com/drive/folders/13r3tEXg5SfyesCLX0w76K5fJ7MHmbJes")
        print("2. Download both files:")
        print("   - random_forest_synthetic.joblib")
        print("   - random_forest_original.joblib")
        print("3. Place them in the current directory: " + os.getcwd())
        print("4. Run evaluate_baselines.py")
        print("="*80)


if __name__ == "__main__":
    main()

