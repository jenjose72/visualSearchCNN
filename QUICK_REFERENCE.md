# Quick Reference - Visual Search CNN

## 📁 Project Structure
```
visualSearchCNN/
├── data/
│   ├── Belts/       (209 images)
│   ├── Keyboard/    (269 images)
│   ├── Shoes/       (263 images)
│   └── Watch/       (197 images)
├── Sequential/
│   ├── Main.cpp
│   ├── layer.h
│   ├── image_loader.h
│   ├── stb_image.h
│   └── stb_image_resize2.h
├── Openmp/
│   ├── Main.cpp
│   ├── layer.h
│   ├── image_loader.h
│   ├── stb_image.h
│   └── stb_image_resize2.h
├── compile.ps1      (Windows compilation script)
├── compile.sh       (Linux compilation script)
├── README.md
├── COMPILE.md
└── SUMMARY.md
```

## 🚀 Quick Start

### Windows
```powershell
# Compile both versions
.\compile.ps1

# Or compile manually
cd Sequential
g++ -O3 -std=c++11 Main.cpp -o cnn_sequential.exe
cd ..\Openmp
g++ -O3 -std=c++11 -fopenmp Main.cpp -o cnn_openmp.exe
cd ..

# Run
.\Sequential\cnn_sequential.exe
.\Openmp\cnn_openmp.exe -t 4
```

### Linux/WSL
```bash
# Make script executable and run
chmod +x compile.sh
./compile.sh

# Or compile manually
cd Sequential && g++ -O3 -std=c++11 Main.cpp -o cnn_sequential && cd ..
cd Openmp && g++ -O3 -std=c++11 -fopenmp Main.cpp -o cnn_openmp && cd ..

# Run
./Sequential/cnn_sequential
./Openmp/cnn_openmp -t 4
```

## 🎯 What Changed

| Aspect | Before (MNIST) | After (Custom) |
|--------|---------------|----------------|
| **Dataset** | MNIST ubyte files | Image files (JPEG, PNG, WebP) |
| **Classes** | 10 (digits 0-9) | 4 (Belts, Keyboard, Shoes, Watch) |
| **Input** | 28x28 grayscale | 28x28 grayscale (auto-resized) |
| **Loader** | mnist.h | image_loader.h + STB libraries |
| **Data Split** | Pre-split | Auto 80/20 split |

## 📊 Dataset Info

- **Total Images:** 938
- **Training Set:** ~750 images (80%)
- **Test Set:** ~188 images (20%)
- **Labels:** 
  - 0 = Belts
  - 1 = Keyboard
  - 2 = Shoes
  - 3 = Watch

## 🔧 Key Functions Modified

### In Main.cpp:
- `loaddata()` - Uses image_loader instead of MNIST
- `classify()` - Changed from 10 to 4 classes
- Layer definition - `l_f(6*6*6, 4, 4)` instead of `(6*6*6, 10, 10)`

### In layer.h:
- `fp_preact_f()` - Now accepts `num_outputs` parameter
- `fp_bias_f()` - Now accepts `num_outputs` parameter
- `bp_weight_f()` - Now accepts `num_outputs` parameter
- `bp_bias_f()` - Now accepts `num_outputs` parameter
- `bp_output_s1()` - Now accepts `num_outputs` parameter

## 💡 Tips

1. **Always run from project root** (where `data/` folder is)
2. **OpenMP threads:** Use `-t` flag to specify (e.g., `-t 4` for 4 threads)
3. **Add more categories:** Just create new folder in `data/` and update code
4. **Supported formats:** .jpg, .jpeg, .png, .webp
5. **Image preprocessing:** Automatic (resize + grayscale + normalize)

## 📚 Documentation Files

- **README.md** - Project overview and changes
- **COMPILE.md** - Detailed compilation instructions
- **SUMMARY.md** - Complete summary of all changes
- **QUICK_REFERENCE.md** - This file

## ✅ Checklist Before Running

- [ ] Images are in `data/Belts/`, `data/Keyboard/`, `data/Shoes/`, `data/Watch/`
- [ ] g++ compiler is installed
- [ ] For OpenMP: OpenMP support is enabled (-fopenmp flag)
- [ ] Running from project root directory
- [ ] Compiled successfully without errors

## 🐛 Common Issues

| Problem | Solution |
|---------|----------|
| "stb_image.h not found" | Compile from Sequential/ or Openmp/ directory |
| "Cannot open directory" | Run executable from project root |
| OpenMP not working | Add `-fopenmp` flag, check g++ version |
| Low accuracy | Normal for small dataset, train longer or add more images |

## 📈 Expected Performance

- **Training:** 1 iteration through dataset
- **Accuracy:** Depends on dataset quality and training
- **Speed:** OpenMP version should be faster with multiple threads
- **Memory:** ~938 images × 28×28 = ~750KB for images

---

**Ready to go! Just compile and run! 🎉**
