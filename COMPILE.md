# Compilation Guide for Visual Search CNN

## Prerequisites

### Windows (MinGW-w64 or MSYS2)
- Install MinGW-w64 or MSYS2
- Make sure `g++` is in your PATH

### Linux/WSL
- Install g++ and OpenMP support:
  ```bash
  sudo apt-get install g++ libomp-dev
  ```

## Compilation Commands

### Sequential Version

Navigate to the Sequential folder:
```bash
cd Sequential
```

#### Windows (PowerShell)
```powershell
g++ -O3 -std=c++11 Main.cpp -o cnn_sequential.exe
```

#### Linux/WSL
```bash
g++ -O3 -std=c++11 Main.cpp -o cnn_sequential
```

### OpenMP Version

Navigate to the OpenMP folder:
```bash
cd Openmp
```

#### Windows (PowerShell)
```powershell
g++ -O3 -std=c++11 -fopenmp Main.cpp -o cnn_openmp.exe
```

#### Linux/WSL
```bash
g++ -O3 -std=c++11 -fopenmp Main.cpp -o cnn_openmp
```

## Running the Programs

### Sequential Version
```bash
./cnn_sequential
```

### OpenMP Version

With default threads:
```bash
./cnn_openmp
```

With specific number of threads (e.g., 4):
```bash
./cnn_openmp -t 4
```

Or:
```bash
./cnn_openmp --threads 8
```

## Expected Output

```
Loaded 209 images from Belts
Loaded 269 images from Keyboard
Loaded 263 images from Shoes
Loaded 197 images from Watch
Total images loaded: 938
Train set: 750 images, Test set: 188 images
Visual Search Using CNN
 2023BCS0017 - Jen Jose Jeeson
 2023BCS0053 - Jefin Francis
Learning
error: X.XXXe-XX, time_on_cpu: X.XXXXXX
...
Error Rate: XX.XX%
```

## Troubleshooting

### Issue: "stb_image.h not found"
**Solution:** Make sure you're compiling from the correct directory (Sequential/ or Openmp/) where the stb header files are located.

### Issue: "Cannot open directory data/Belts"
**Solution:** Make sure you run the executable from the project root directory where the `data` folder is located.

### Issue: OpenMP not working
**Solution:** 
- On Windows with MinGW: Use the `-fopenmp` flag
- Verify OpenMP support: `g++ --version` and check for OpenMP support
- Try installing a newer version of g++ that includes OpenMP

### Issue: Images not loading
**Solution:**
- Verify images are in the correct folders: `data/Belts/`, `data/Keyboard/`, `data/Shoes/`, `data/Watch/`
- Check that images have supported extensions (.jpg, .jpeg, .png, .webp)
- Ensure the executable is run from the project root directory

## Performance Notes

- The Sequential version will use a single CPU core
- The OpenMP version will parallelize computations across multiple cores
- Use `-t` flag to control the number of threads in OpenMP version
- Optimal thread count usually matches your CPU core count
