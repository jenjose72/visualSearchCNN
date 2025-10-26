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
./Openmp/cnn_openmp -t 8
```


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
./Sequential/cnn_sequential --load -i ~/Downloads/my_image.jpg --no-test
```

### OpenMP
```bash
./Openmp/cnn_openmp --load -i ~/Downloads/my_image.jpg

./Openmp/cnn_openmp --load -i ~/Downloads/my_image.jpg -t 8
```

T

