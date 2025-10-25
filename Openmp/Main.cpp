#include "image_loader.h" 
#include "layer.h"
#include <cstdio>
#include <ctime>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <omp.h>
    
static image_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN (3 output classes: Belts, Shoes, Watch)
static Layer l_input(0, 0, 28*28);
static Layer l_c1(5*5, 6, 24*24*6);
static Layer l_s1(4*4, 1, 6*6*6);
static Layer l_f(6*6*6, 3, 3);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();
static void save_model(const char* filename);
static bool load_model(const char* filename);
static void test_single_image(double data[28][28]);

float vectorNorm(float* vec, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Simple data augmentation: add random noise
void augment_image(double original[28][28], double augmented[28][28], float noise_level = 0.05f) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            // Add small random noise
            float noise = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * noise_level;
            augmented[i][j] = original[i][j] + noise;
            
            // Clamp to [0, 1]
            if (augmented[i][j] < 0.0) augmented[i][j] = 0.0;
            if (augmented[i][j] > 1.0) augmented[i][j] = 1.0;
        }
    }
}

// Horizontal flip augmentation
void flip_horizontal(double original[28][28], double flipped[28][28]) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            flipped[i][27 - j] = original[i][j];
        }
    }
}

static inline void loaddata()
{
	image_data *all_data;
	unsigned int total_count;
	
	// Load all images from the four categories
	if (load_custom_dataset(&all_data, &total_count, "data") != 0) {
		fprintf(stderr, "Failed to load dataset\n");
		exit(1);
	}
	
	// Split into train and test sets (80/20 split)
	split_dataset(all_data, total_count, &train_set, &train_cnt, &test_set, &test_cnt);
	
	free(all_data);
}

int main(int argc, const char **argv) {
    int num_threads = 0;
    const char* model_file = "cnn_model_omp.bin";
    bool skip_training = false;
    const char* test_image_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                num_threads = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "--load") == 0 || strcmp(argv[i], "-l") == 0) {
            skip_training = true;
        } else if (strcmp(argv[i], "--model") == 0 || strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) {
                model_file = argv[++i];
            }
        } else if (strcmp(argv[i], "--test-image") == 0 || strcmp(argv[i], "-i") == 0) {
            if (i + 1 < argc) {
                test_image_path = argv[++i];
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stdout, "Usage: %s [OPTIONS]\n", argv[0]);
            fprintf(stdout, "Options:\n");
            fprintf(stdout, "  -t, --threads <N>           Set number of OpenMP threads\n");
            fprintf(stdout, "  --load, -l                  Load pre-trained model instead of training\n");
            fprintf(stdout, "  --model, -m <file>          Specify model file (default: cnn_model_omp.bin)\n");
            fprintf(stdout, "  --test-image, -i <file>     Test a custom image and show prediction\n");
            fprintf(stdout, "  --help, -h                  Show this help message\n");
            fprintf(stdout, "\nExamples:\n");
            fprintf(stdout, "  %s -t 8                                 # Train with 8 threads\n", argv[0]);
            fprintf(stdout, "  %s --load -t 4                          # Load model, test with 4 threads\n", argv[0]);
            fprintf(stdout, "  %s --load --test-image shoe.jpg         # Test custom image\n", argv[0]);
            fprintf(stdout, "  %s -t 8 --test-image data/Shoes/s1.jpg  # Train and test custom image\n", argv[0]);
            return 0;
        } else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
            /* allow a bare numeric positional argument */
            num_threads = atoi(argv[i]);
        }
    }

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
        fprintf(stdout, "OpenMP: set number of threads to %d\n", num_threads);
    } else {
        fprintf(stdout, "OpenMP: using default number of threads (%d)\n", omp_get_max_threads());
    }

    srand(time(NULL));
    
    // Check if we only need to test a custom image with a loaded model
    bool only_test_image = skip_training && test_image_path != nullptr;
    
    // If just testing custom image, skip dataset loading
    if (only_test_image) {
        if (load_model(model_file)) {
            fprintf(stdout, "Model loaded from %s\n\n", model_file);
            double custom_data[28][28];
            if (load_single_image(test_image_path, custom_data) == 0) {
                fprintf(stdout, "=== Testing Custom Image ===\n");
                fprintf(stdout, "Image: %s\n", test_image_path);
                test_single_image(custom_data);
                return 0;
            } else {
                fprintf(stderr, "Failed to load custom image: %s\n", test_image_path);
                return 1;
            }
        } else {
            fprintf(stderr, "Failed to load model from %s\n", model_file);
            return 1;
        }
    }
    
    // Load dataset for training or full testing
    loaddata();
    
    // Try to load existing model or train new one
    if (skip_training && load_model(model_file)) {
        fprintf(stdout, "Using pre-trained model from %s\n\n", model_file);
    } else {
        if (skip_training) {
            fprintf(stdout, "Could not load model, training new model...\n\n");
        }
        learn();
        save_model(model_file);
    }
    
    // Test custom image if provided
    if (test_image_path) {
        double custom_data[28][28];
        if (load_single_image(test_image_path, custom_data) == 0) {
            fprintf(stdout, "\n=== Testing Custom Image ===\n");
            fprintf(stdout, "Image: %s\n", test_image_path);
            test_single_image(custom_data);
        } else {
            fprintf(stderr, "Failed to load custom image: %s\n", test_image_path);
        }
    }
    
    test();

    return 0;
}

static double forward_pass(double data[28][28]) {
  float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();
    double start_1 = omp_get_wtime();
	

	l_input.setOutput((float *)input);
	 // forward pass Convolution Layer
    fp_c1((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight,l_c1.bias);
    apply_step_function(l_c1.preact, l_c1.output, l_c1.O);
    
    fp_s1((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight,l_s1.bias);
    apply_step_function(l_s1.preact, l_s1.output, l_s1.O);
    

 // forward pass Fully Connected Layer
   
    fp_preact_f((float (*)[6][6])l_s1.output, l_f.preact, l_f.weight, l_f.N);
    fp_bias_f(l_f.preact, l_f.bias, l_f.N);
    apply_step_function(l_f.preact, l_f.output, l_f.O);
    
    double end_1 = omp_get_wtime();
    return end_1 - start_1;
}

static double back_pass() {
    double start_1 = omp_get_wtime();
   
    bp_weight_f(l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output, l_f.N);
    bp_bias_f(l_f.bias, l_f.d_preact, l_f.N);
 
    bp_output_s1((float (*)[6][6])l_s1.d_output, l_f.weight, l_f.d_preact, l_f.N);
    bp_preact_s1((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
    bp_weight_s1((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
    bp_bias_s1(l_s1.bias, (float (*)[6][6])l_s1.d_preact);
     
    bp_output_c1((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
    bp_preact_c1((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
    bp_weight_c1((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
    bp_bias_c1(l_c1.bias, (float (*)[24][24])l_c1.d_preact);

	apply_grad(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
	apply_grad(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
	apply_grad(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
    
    double end_1 = omp_get_wtime();
    return end_1 - start_1;
}

static void learn() {
    float err;
	int total_epochs = 80;  // Total epochs for high accuracy
	int iter = total_epochs;
	int current_epoch = 0;
	
	double time_taken = 0.0;
    fprintf(stdout ,"Visual Search Using CNN\n 2023BCS0017 - Jen Jose Jeeson\n 2023BCS0053 - Jefin Francis\n");
	fprintf(stdout ,"Learning with %d epochs and adaptive learning rate (OpenMP)\n", total_epochs);

	while (iter < 0 || iter-- > 0) {
		current_epoch++;
		
		// Update learning rate with decay
		update_learning_rate(current_epoch, total_epochs);
		
		err = 0.0f;

		// Shuffle training indices for randomization
		std::vector<int> indices(train_cnt);
		for(int i = 0; i < train_cnt; ++i) indices[i] = i;
		std::shuffle(indices.begin(), indices.end(), std::default_random_engine(time(NULL) + current_epoch));

		for (int idx : indices) {
			float tmp_err;
			
			// Randomly augment data (50% chance)
			double augmented_data[28][28];
			if (rand() % 2 == 0 && current_epoch > 10) {  // Start augmentation after 10 epochs
				if (rand() % 2 == 0) {
					augment_image(train_set[idx].data, augmented_data, 0.05f);
				} else {
					flip_horizontal(train_set[idx].data, augmented_data);
				}
				time_taken += forward_pass(augmented_data);
			} else {
				time_taken += forward_pass(train_set[idx].data);
			}

			l_f.bp_clear();
			l_s1.bp_clear();
			l_c1.bp_clear();

            // Euclid distance of train_set[idx]
    makeError(l_f.d_preact, l_f.output, train_set[idx].label, 3);
            tmp_err = vectorNorm(l_f.d_preact, 3);
            err += tmp_err;
           time_taken += back_pass();
        }

        err /= train_cnt;
		
		// Print progress every 10 epochs or if error is very low
		if (current_epoch % 10 == 0 || current_epoch == 1 || err < 0.15) {
			extern float dt;  // Access current learning rate
			fprintf(stdout, "Epoch %3d/%d - error: %.6f, lr: %.6f, time: %.2lf s\n", 
					current_epoch, total_epochs, err, dt, time_taken);
		}

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}

	}
	
	fprintf(stdout, "\n Time - %lf\n", time_taken);
}

static unsigned int classify(double data[28][28]) {
    float res[3];
    forward_pass(data);
    unsigned int max = 0;
   for (int i = 0; i < 3; i++) {
        res[i] = l_f.output[i];
    }
    for (int i = 1; i < 3; ++i) {
        if (res[max] < res[i]) {
            max = i;
        }
    }

    return max;
}

static void test()
{
	int error = 0;
	const char* class_names[] = {"Belts", "Shoes", "Watch"};
	int confusion_matrix[3][3] = {0}; // [actual][predicted]

	for (int i = 0; i < test_cnt; ++i) {
		unsigned int predicted = classify(test_set[i].data);
		unsigned int actual = test_set[i].label;
		
		confusion_matrix[actual][predicted]++;
		
		if (predicted != actual) {
			++error;
		}
	}

	fprintf(stdout, "\n=== Test Results ===\n");
	fprintf(stdout, "Total Test Samples: %d\n", test_cnt);
	fprintf(stdout, "Correct: %d\n", test_cnt - error);
	fprintf(stdout, "Incorrect: %d\n", error);
	fprintf(stdout, "Accuracy: %.2lf%%\n", 
		(1.0 - double(error) / double(test_cnt)) * 100.0);
	fprintf(stdout, "Error Rate: %.2lf%%\n\n",
		double(error) / double(test_cnt) * 100.0);
	
	fprintf(stdout, "Confusion Matrix:\n");
	fprintf(stdout, "Actual\\Predicted   Belts  Shoes  Watch\n");
	fprintf(stdout, "----------------------------------------\n");
	for (int i = 0; i < 3; i++) {
		fprintf(stdout, "%-15s", class_names[i]);
		for (int j = 0; j < 3; j++) {
			fprintf(stdout, "%7d", confusion_matrix[i][j]);
		}
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "===================\n");
}

// Save model weights to file
static void save_model(const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Failed to open file for writing: %s\n", filename);
        return;
    }
    
    // Save convolution layer weights and biases
    fwrite(l_c1.weight, sizeof(float), l_c1.M * l_c1.N, file);
    fwrite(l_c1.bias, sizeof(float), l_c1.N, file);
    
    // Save pooling layer weights and biases
    fwrite(l_s1.weight, sizeof(float), l_s1.M * l_s1.N, file);
    fwrite(l_s1.bias, sizeof(float), l_s1.N, file);
    
    // Save fully connected layer weights and biases
    fwrite(l_f.weight, sizeof(float), l_f.M * l_f.N, file);
    fwrite(l_f.bias, sizeof(float), l_f.N, file);
    
    fclose(file);
    fprintf(stdout, "\nModel saved to %s\n", filename);
}

// Load model weights from file
static bool load_model(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        return false;
    }
    
    // Load convolution layer weights and biases
    size_t read_count;
    read_count = fread(l_c1.weight, sizeof(float), l_c1.M * l_c1.N, file);
    if (read_count != l_c1.M * l_c1.N) {
        fclose(file);
        return false;
    }
    
    read_count = fread(l_c1.bias, sizeof(float), l_c1.N, file);
    if (read_count != l_c1.N) {
        fclose(file);
        return false;
    }
    
    // Load pooling layer weights and biases
    read_count = fread(l_s1.weight, sizeof(float), l_s1.M * l_s1.N, file);
    if (read_count != l_s1.M * l_s1.N) {
        fclose(file);
        return false;
    }
    
    read_count = fread(l_s1.bias, sizeof(float), l_s1.N, file);
    if (read_count != l_s1.N) {
        fclose(file);
        return false;
    }
    
    // Load fully connected layer weights and biases
    read_count = fread(l_f.weight, sizeof(float), l_f.M * l_f.N, file);
    if (read_count != l_f.M * l_f.N) {
        fclose(file);
        return false;
    }
    
    read_count = fread(l_f.bias, sizeof(float), l_f.N, file);
    if (read_count != l_f.N) {
        fclose(file);
        return false;
    }
    
    fclose(file);
    return true;
}

// Test a single custom image
static void test_single_image(double data[28][28]) {
    unsigned int prediction = classify(data);
    
    const char* class_names[] = {"Belts", "Shoes", "Watch"};
    
    fprintf(stdout, "\n=== Prediction Results ===\n");
    fprintf(stdout, "Predicted class: %s (label %d)\n", class_names[prediction], prediction);
    fprintf(stdout, "\nConfidence scores:\n");
    for (int i = 0; i < 3; i++) {
        fprintf(stdout, "  %s: %.4f\n", class_names[i], l_f.output[i]);
    }
    fprintf(stdout, "========================\n\n");
}