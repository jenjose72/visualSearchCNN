#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h" 
#include "layer.h"
#include <cstdio>
#include <ctime>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include <cstring>
#include <cstdlib>
#include <omp.h>
    
static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

// Define layers of CNN
static Layer l_input(0, 0, 28*28);
static Layer l_c1(5*5, 6, 24*24*6);
static Layer l_s1(4*4, 1, 6*6*6);
static Layer l_f(6*6*6, 10, 10);

static void learn();
static unsigned int classify(double data[28][28]);
static void test();
static double forward_pass(double data[28][28]);
static double back_pass();

float vectorNorm(float* vec, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

static inline void loaddata()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const char **argv) {
    int num_threads = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                num_threads = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            fprintf(stdout, "Usage: %s [-t N|--threads N]\n", argv[0]);
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
    loaddata();
    learn();
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
   
    fp_preact_f((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
    fp_bias_f(l_f.preact, l_f.bias);
    apply_step_function(l_f.preact, l_f.output, l_f.O);
    
    double end_1 = omp_get_wtime();
    return end_1 - start_1;
}

static double back_pass() {
    double start_1 = omp_get_wtime();
   
    bp_weight_f((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
    bp_bias_f(l_f.bias, l_f.d_preact);
 
    bp_output_s1((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
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
	int iter = 1;
	
	double time_taken = 0.0;

    fprintf(stdout ,"Visual Search Using CNN\n 2023BCS0017 - Jen Jose Jeeson\n 2023BCS0053 - Jefin Francis\n");
	fprintf(stdout ,"Learning\n");

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;

		for (int i = 0; i < train_cnt; ++i) {
			float tmp_err;

			time_taken += forward_pass(train_set[i].data);

			l_f.bp_clear();
			l_s1.bp_clear();
			l_c1.bp_clear();

            // Euclid distance of train_set[i]
    makeError(l_f.d_preact, l_f.output, train_set[i].label, 10);
            tmp_err = vectorNorm(l_f.d_preact, 10);
            err += tmp_err;
           time_taken += back_pass();
        }

    err /= train_cnt;
    double avg_time = (train_cnt > 0) ? (time_taken / train_cnt) : 0.0;
    fprintf(stdout, "error: %e, total_time(s): %lf, avg_time_per_sample(s): %lf\n", err, time_taken, avg_time);

		if (err < threshold) {
			fprintf(stdout, "Training complete, error less than threshold\n\n");
			break;
		}

	}
	
    fprintf(stdout, "\n Total training time (s): %lf\n", time_taken);
}

static unsigned int classify(double data[28][28]) {
    float res[10];
    forward_pass(data);
    unsigned int max = 0;
   for (int i = 0; i < 10; i++) {
        res[i] = l_f.output[i];
    }
    for (int i = 1; i < 10; ++i) {
        if (res[max] < res[i]) {
            max = i;
        }
    }

    return max;
}

static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}