#!/usr/bin/env python3
"""
WebP to JPEG Converter
Converts all WebP images in the data folders to JPEG format
"""

import os
from pathlib import Path

try:
    from PIL import Image
    print("✓ Pillow library found")
except ImportError:
    print("✗ Pillow library not found!")
    print("Installing Pillow...")
    import subprocess
    subprocess.check_call(['python', '-m', 'pip', 'install', 'Pillow'])
    from PIL import Image
    print("✓ Pillow installed successfully")

def convert_webp_to_jpeg(data_path):
    """Convert all WebP files in data folders to JPEG"""
    categories = ['Belts', 'Keyboard', 'Shoes', 'Watch']
    
    total_converted = 0
    total_failed = 0
    
    for category in categories:
        category_path = Path(data_path) / category
        if not category_path.exists():
            print(f"! Folder not found: {category_path}")
            continue
        
        print(f"\n📁 Processing {category} folder...")
        converted = 0
        
        # Find all WebP files
        webp_files = list(category_path.glob("*.webp"))
        
        for webp_file in webp_files:
            jpeg_file = webp_file.with_suffix('.jpg')
            
            # Skip if JPEG already exists
            if jpeg_file.exists():
                continue
            
            try:
                # Open and convert WebP to RGB (removes alpha channel if present)
                img = Image.open(webp_file)
                img = img.convert('RGB')
                
                # Save as JPEG
                img.save(jpeg_file, 'JPEG', quality=95)
                
                print(f"  ✓ Converted: {webp_file.name}")
                converted += 1
                total_converted += 1
                
            except Exception as e:
                print(f"  ✗ Failed to convert {webp_file.name}: {e}")
                total_failed += 1
        
        print(f"  Converted {converted} files in {category}")
    
    print("\n" + "="*50)
    print("Conversion Complete!")
    print("="*50)
    print(f"✓ Total converted: {total_converted}")
    print(f"✗ Total failed: {total_failed}")
    print("\nNow rerun your CNN program to use all images!")

if __name__ == "__main__":
    data_path = Path(__file__).parent / "data"
    print("="*50)
    print("WebP to JPEG Converter")
    print("="*50)
    print(f"Data path: {data_path}")
    
    convert_webp_to_jpeg(data_path)
