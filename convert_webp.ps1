# WebP to JPEG Converter Script
# This script converts WebP images to JPEG format using ImageMagick or falls back to Python with Pillow

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "WebP to JPEG Converter" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$dataPath = "C:\Users\jefin\Desktop\visualSearchCNN\data"
$categories = @("Belts", "Keyboard", "Shoes", "Watch")

# Check if ImageMagick is available
$hasImageMagick = $false
try {
    $magickVersion = magick --version 2>&1 | Select-Object -First 1
    if ($LASTEXITCODE -eq 0) {
        $hasImageMagick = $true
        Write-Host "✓ Found ImageMagick" -ForegroundColor Green
    }
} catch {
    Write-Host "! ImageMagick not found, will try Python method" -ForegroundColor Yellow
}

# Check if Python is available
$hasPython = $false
if (-not $hasImageMagick) {
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -eq 0) {
            $hasPython = $true
            Write-Host "✓ Found Python" -ForegroundColor Green
            
            # Check if Pillow is installed
            $pillowCheck = python -c "import PIL" 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Installing Pillow library..." -ForegroundColor Yellow
                python -m pip install Pillow
            }
        }
    } catch {
        Write-Host "✗ Neither ImageMagick nor Python found!" -ForegroundColor Red
        Write-Host "Please install one of the following:" -ForegroundColor Yellow
        Write-Host "  1. ImageMagick: https://imagemagick.org/script/download.php" -ForegroundColor White
        Write-Host "  2. Python with Pillow: pip install Pillow" -ForegroundColor White
        exit 1
    }
}

Write-Host ""

$totalConverted = 0
$totalFailed = 0

foreach ($category in $categories) {
    $categoryPath = Join-Path $dataPath $category
    Write-Host "Processing $category folder..." -ForegroundColor Yellow
    
    $webpFiles = Get-ChildItem -Path $categoryPath -Filter "*.webp" -File
    $converted = 0
    
    foreach ($file in $webpFiles) {
        $outputPath = $file.FullName -replace '\.webp$', '.jpg'
        
        # Skip if JPEG already exists
        if (Test-Path $outputPath) {
            continue
        }
        
        try {
            if ($hasImageMagick) {
                # Use ImageMagick
                magick convert $file.FullName $outputPath 2>&1 | Out-Null
            } else {
                # Use Python with Pillow
                $pythonScript = @"
from PIL import Image
img = Image.open('$($file.FullName)')
img = img.convert('RGB')
img.save('$outputPath', 'JPEG', quality=95)
"@
                $pythonScript | python 2>&1 | Out-Null
            }
            
            if (Test-Path $outputPath) {
                $converted++
                $totalConverted++
                Write-Host "  ✓ Converted: $($file.Name)" -ForegroundColor Green
            } else {
                $totalFailed++
                Write-Host "  ✗ Failed: $($file.Name)" -ForegroundColor Red
            }
        } catch {
            $totalFailed++
            Write-Host "  ✗ Error converting: $($file.Name)" -ForegroundColor Red
        }
    }
    
    Write-Host "  Converted $converted files in $category" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Conversion Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total converted: $totalConverted" -ForegroundColor Green
Write-Host "Total failed: $totalFailed" -ForegroundColor $(if ($totalFailed -eq 0) { "Green" } else { "Red" })
Write-Host ""
Write-Host "Now rerun your CNN program to use the converted images!" -ForegroundColor Yellow
