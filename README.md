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
### Sequential (train and evaluate)
```bash
./Sequential/cnn_sequential
```

### OpenMP (train with N threads and evaluate)
```bash
./Openmp/cnn_openmp -t 8   # use 8 threads (or omit -t to use the default)
```

During training, you’ll see epoch progress with error and current learning rate. At the end, a confusion matrix and accuracy are printed.

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

