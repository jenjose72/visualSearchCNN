# Visual Search CNN (Sequential + OpenMP)

End-to-end image classifier for 3 product classes using a tiny CNN, with both sequential and OpenMP-accelerated implementations.

- Classes: Belts (0), Shoes (1), Watch (2)
- Input: 28×28 grayscale (images are auto-loaded and resized)
- Dataset: Loaded from `data/` and split 80/20 (train/test)
- Extras: Learning-rate decay, data shuffling, optional data augmentation, save/load model, single-image classification

## Directory layout

```
Sequential/   # Single-threaded implementation
Openmp/       # OpenMP-accelerated implementation
data/         # Dataset folders (see below)
```

## Dataset

Place your images under `data/` like this. Supported formats: jpg, jpeg, png, webp. Images are converted to grayscale and resized to 28×28 automatically.

```
data/
├── Belts/
├── Shoes/
└── Watch/
```

The loader will print how many images it found per class and the final train/test split.

## Build (Linux, g++)

You only need a C++11 compiler. STB headers (`stb_image.h`, `stb_image_resize2.h`) are already in the repo. No OpenCV required.

### Sequential
```bash
cd Sequential
g++ -O3 -std=c++11 Main.cpp -o cnn_sequential -lm
```

### OpenMP
```bash
cd Openmp
g++ -O3 -std=c++11 -fopenmp Main.cpp -o cnn_openmp -lm
```

## Train

Training uses 80 epochs by default with learning rate decay and random shuffling each epoch.

### Sequential (train and evaluate)
```bash
./Sequential/cnn_sequential
```

### OpenMP (train with N threads and evaluate)
```bash
./Openmp/cnn_openmp -t 8   # use 8 threads (or omit -t to use the default)
```

During training, you’ll see epoch progress with error and current learning rate. At the end, a confusion matrix and accuracy are printed.

## Save / Load model

Both builds can save weights after training and reload them later to avoid retraining.

- Default model file names:
   - Sequential: `cnn_model.bin`
   - OpenMP: `cnn_model_omp.bin` (you can load the sequential model too)

### Train and save (default file)
```bash
./Sequential/cnn_sequential
./Openmp/cnn_openmp -t 8
```

### Load saved model (skip training)
```bash
./Sequential/cnn_sequential --load
./Openmp/cnn_openmp --load
```

### Use a custom model file
```bash
./Sequential/cnn_sequential --model my_model.bin
./Openmp/cnn_openmp --load --model cnn_model.bin
```

## Classify a single custom image (no dataset required)

You can classify any single image without loading the dataset.

### Sequential
```bash
# Fast: load model, classify one image, skip dataset and test set
./Sequential/cnn_sequential --load -i ~/Downloads/my_image.jpg --no-test
```

### OpenMP
```bash
# Fast: if --load and -i are provided together, dataset is skipped
./Openmp/cnn_openmp --load -i ~/Downloads/my_image.jpg

# Optional: also specify threads
./Openmp/cnn_openmp --load -i ~/Downloads/my_image.jpg -t 8
```

The output shows the predicted class and confidence scores for Belts/Shoes/Watch.

## Full CLI reference

### Sequential
- `--load, -l`        Load a saved model (skip training)
- `--model, -m <f>`   Use custom model file (default: `cnn_model.bin`)
- `--test-image, -i`  Classify a single image (auto-resize to 28×28)
- `--no-test`         Skip validation testing (useful with `--load -i`)
- `--help, -h`        Show usage

### OpenMP
- `-t, --threads <N>` Set number of threads
- `--load, -l`        Load a saved model (skip training)
- `--model, -m <f>`   Use custom model file (default: `cnn_model_omp.bin`)
- `--test-image, -i`  Classify a single image (auto-resize to 28×28)
- `--help, -h`        Show usage and examples

Notes:
- For OpenMP, if you pass `--load` and `-i` together, the program skips loading the dataset and immediately classifies your image.
- OpenMP will print the number of threads it’s using. You can also set `OMP_NUM_THREADS` in the environment.

## Tips & troubleshooting

- If accuracy is low:
   - Ensure classes are balanced and images are reasonably clear.
   - Let it train for all 80 epochs at least once to produce a good `cnn_model.bin`.
   - The code performs shuffling every epoch and applies light augmentation after epoch 10.
- If the executable is missing:
   - Double-check you compiled to `cnn_sequential` / `cnn_openmp` in the corresponding folders.
- If you only want to classify one image quickly:
   - Sequential: `--load -i <img> --no-test`
   - OpenMP: `--load -i <img>` (dataset is skipped automatically)

## Example session

```bash
# Train sequential and save model
./Sequential/cnn_sequential

# Classify one image quickly using the saved model
./Sequential/cnn_sequential --load -i ~/Downloads/belts1.jpg --no-test

# Train OpenMP with 12 threads, then classify one image
./Openmp/cnn_openmp -t 12
./Openmp/cnn_openmp --load -i ~/Downloads/belts1.jpg
```

That’s it—compile, train, and classify with either build. Have fun experimenting!
