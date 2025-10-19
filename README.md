# Visual Search CNN - Custom Dataset Support

## Changes Made

### Overview
The project has been updated to work with a custom image dataset (Belts, Keyboard, Shoes, Watch) instead of the MNIST ubyte files.

### Key Changes

1. **New Image Loader (`image_loader.h`)**
   - Replaces the MNIST loader
   - Uses STB image libraries for loading various image formats (JPEG, PNG, WebP)
   - Automatically loads images from 4 categories:
     - Belts (label 0)
     - Keyboard (label 1)
     - Shoes (label 2)
     - Watch (label 3)
   - Resizes all images to 28x28 grayscale
   - Normalizes pixel values to [0, 1]
   - Automatically splits dataset into 80% training, 20% testing

2. **Network Architecture Updates**
   - Changed output layer from 10 classes (MNIST digits) to 4 classes (product categories)
   - Updated fully connected layer: `Layer l_f(6*6*6, 4, 4);`
   - Modified all function calls to support variable output sizes

3. **Layer Functions Updated**
   - `fp_preact_f()`, `fp_bias_f()`, `bp_weight_f()`, `bp_bias_f()`, `bp_output_s1()` now accept `num_outputs` parameter
   - Functions are now flexible and can work with any number of output classes

4. **Both Implementations Updated**
   - **Sequential** (`Sequential/Main.cpp`, `Sequential/layer.h`)
   - **OpenMP** (`Openmp/Main.cpp`, `Openmp/layer.h`)

### Dependencies Added
- `stb_image.h` - For loading various image formats
- `stb_image_resize2.h` - For resizing images to 28x28

### Dataset Structure
```
data/
├── Belts/       (images: .jpg, .jpeg, .png, .webp)
├── Keyboard/    (images: .jpg, .jpeg, .png, .webp)
├── Shoes/       (images: .jpg, .jpeg, .png, .webp)
└── Watch/       (images: .jpg, .jpeg, .png, .webp)
```

### How to Compile and Run

#### Sequential Version
```bash
cd Sequential
g++ -O3 -std=c++11 Main.cpp -o cnn_sequential
./cnn_sequential
```

#### OpenMP Version
```bash
cd Openmp
g++ -O3 -std=c++11 -fopenmp Main.cpp -o cnn_openmp
./cnn_openmp -t 4    # Use 4 threads
```

### Output
- Loads all images from the 4 categories
- Displays count of images loaded from each category
- Shows total train/test split
- Performs CNN training on the custom dataset
- Reports error rate on test set

### Notes
- Images are automatically converted to grayscale
- All images are resized to 28x28 to match CNN input layer
- The dataset is shuffled before splitting into train/test
- Training uses the same parameters as original MNIST implementation
