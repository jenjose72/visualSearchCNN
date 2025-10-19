# Visual Search CNN - Windows Compilation Script
# Run this from the project root directory

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Visual Search CNN - Compilation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if g++ is available
try {
    $gccVersion = g++ --version 2>&1 | Select-Object -First 1
    Write-Host "✓ Found g++: $gccVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ g++ not found! Please install MinGW-w64 or MSYS2" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Compiling Sequential Version..." -ForegroundColor Yellow
Push-Location Sequential
try {
    g++ -O3 -std=c++11 Main.cpp -o cnn_sequential.exe 2>&1 | Write-Host
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Sequential version compiled successfully!" -ForegroundColor Green
    } else {
        Write-Host "✗ Sequential compilation failed!" -ForegroundColor Red
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Compiling OpenMP Version..." -ForegroundColor Yellow
Push-Location Openmp
try {
    g++ -O3 -std=c++11 -fopenmp Main.cpp -o cnn_openmp.exe 2>&1 | Write-Host
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ OpenMP version compiled successfully!" -ForegroundColor Green
    } else {
        Write-Host "✗ OpenMP compilation failed!" -ForegroundColor Red
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Compilation Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the programs:" -ForegroundColor Yellow
Write-Host "  Sequential: .\Sequential\cnn_sequential.exe" -ForegroundColor White
Write-Host "  OpenMP:     .\Openmp\cnn_openmp.exe -t 4" -ForegroundColor White
Write-Host ""
