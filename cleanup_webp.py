#!/usr/bin/env python3
"""
WebP Cleanup and Converter
1. Converts any remaining WebP files to JPEG
2. Removes all WebP files after conversion
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

def cleanup_webp_files(data_path):
    """Convert all WebP files to JPEG and remove WebP files"""
    categories = ['Belts', 'Keyboard', 'Shoes', 'Watch']
    
    total_converted = 0
    total_removed = 0
    
    for category in categories:
        category_path = Path(data_path) / category
        if not category_path.exists():
            print(f"! Folder not found: {category_path}")
            continue
        
        print(f"\n📁 Processing {category} folder...")
        converted = 0
        removed = 0
        
        # Find all WebP files
        webp_files = list(category_path.glob("*.webp"))
        
        for webp_file in webp_files:
            jpeg_file = webp_file.with_suffix('.jpg')
            
            # Convert if JPEG doesn't exist
            if not jpeg_file.exists():
                try:
                    img = Image.open(webp_file)
                    img = img.convert('RGB')
                    img.save(jpeg_file, 'JPEG', quality=95)
                    print(f"  ✓ Converted: {webp_file.name} → {jpeg_file.name}")
                    converted += 1
                    total_converted += 1
                except Exception as e:
                    print(f"  ✗ Failed to convert {webp_file.name}: {e}")
                    continue
            
            # Remove WebP file
            try:
                webp_file.unlink()
                print(f"  🗑️  Removed: {webp_file.name}")
                removed += 1
                total_removed += 1
            except Exception as e:
                print(f"  ✗ Failed to remove {webp_file.name}: {e}")
        
        print(f"  {category}: Converted {converted}, Removed {removed} WebP files")
    
    print("\n" + "="*50)
    print("Cleanup Complete!")
    print("="*50)
    print(f"✓ Total converted: {total_converted}")
    print(f"🗑️  Total removed: {total_removed}")
    
    # Summary of remaining images
    print("\n" + "="*50)
    print("Image Count Summary:")
    print("="*50)
    for category in categories:
        category_path = Path(data_path) / category
        if category_path.exists():
            jpg_count = len(list(category_path.glob("*.jpg"))) + len(list(category_path.glob("*.jpeg")))
            png_count = len(list(category_path.glob("*.png")))
            webp_count = len(list(category_path.glob("*.webp")))
            total = jpg_count + png_count + webp_count
            print(f"{category:12} - Total: {total:3} (JPG: {jpg_count:3}, PNG: {png_count:3}, WebP: {webp_count:3})")
    
    print("\n✅ All WebP files removed! Now only JPG/PNG images remain.")

if __name__ == "__main__":
    data_path = Path(__file__).parent / "data"
    print("="*50)
    print("WebP Cleanup and Converter")
    print("="*50)
    print(f"Data path: {data_path}\n")
    
    cleanup_webp_files(data_path)
