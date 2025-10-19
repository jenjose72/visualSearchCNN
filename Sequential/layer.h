#include <cstdlib>
#include <vector>
#include <memory>
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifndef LAYER_H
#define LAYER_H
#endif

static float dt = 5.0E-02f;  // Initial learning rate (will decay over time)
const static float threshold = 1.0E-02f;

// Function to update learning rate with decay
void update_learning_rate(int epoch, int total_epochs) {
    // Learning rate decay: reduce by factor over time
    float initial_lr = 5.0E-02f;
    float decay_factor = 0.95f;  // Decay by 5% each epoch
    dt = initial_lr * pow(decay_factor, epoch);
    
    // Ensure learning rate doesn't get too small
    if (dt < 1.0E-04f) dt = 1.0E-04f;
}

class Layer {
	public:
	int M, N, O;

	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int M, int N, int O);

	~Layer();

	void setOutput(float *data);
	void clear();
	void bp_clear();
};

// Constructor
Layer::Layer(int M, int N, int O) : M(M), N(N), O(O) {
    output = new float[O]();
    preact = new float[O]();
    bias = new float[N]();
    weight = new float[M * N]();
    d_output = new float[O]();
    d_preact = new float[O]();
    d_weight = new float[M * N]();

    for (int i = 0; i < N; ++i) {
        bias[i] = 0.5f - static_cast<float>(rand()) / RAND_MAX;

        for (int j = 0; j < M; ++j) {
            weight[i * M + j] = 0.5f - static_cast<float>(rand()) / RAND_MAX;
        }
    }
}

// Destructor
Layer::~Layer() {
    delete[] output;
    delete[] preact;
    delete[] bias;
    delete[] weight;
    delete[] d_output;
    delete[] d_preact;
    delete[] d_weight;
}

void Layer::setOutput(float *data) {
    memcpy(output, data, sizeof(float) * O);
}

void Layer::clear() {
    memset(output, 0, sizeof(float) * O);
    memset(preact, 0, sizeof(float) * O);
}

void Layer::bp_clear() {
    memset(d_weight, 0, sizeof(float) * M * N);
}

float step_function(float v) {
    return 1 / (1 + exp(-v));
}

void apply_step_function(float *input, float *output, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] = step_function(input[i]);
    }
}

void makeError(float *err, float *output, unsigned int Y, int N) {
    for (int i = 0; i < N; ++i) {
        err[i] = (i == Y) ? 1.0f - output[i] : -output[i];
    }
}

void apply_grad(float *output, float *grad, int N) {
    for (int i = 0; i < N; ++i) {
        output[i] += dt * grad[i];
    }
}



void fp_c1(const float input[28][28], float preact[6][24][24], const float weight[6][5][5], const float bias[6]) {
    // Initialize preact with zeros
  
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
            for (int k = 0; k < 24; ++k) {
                preact[i][j][k] = 0;
            }
        }
    }

    // Compute preact values
   
    for (int m = 0; m < 6; ++m) {
        for (int x = 0; x < 24; ++x) {
            for (int y = 0; y < 24; ++y) {
                float sum = 0.0f;  // sum needs to be private to each thread
                for (int i = 0; i < 5; ++i) {
                    for (int j = 0; j < 5; ++j) {
                        sum += input[x + i][y + j] * weight[m][i][j];
                    }
                }
                preact[m][x][y] += sum;
            }
        }
    }

 
    for (int i = 0; i < 6; ++i) {
        for (int x = 0; x < 24; ++x) {
            for (int y = 0; y < 24; ++y) {
                preact[i][x][y] += bias[i];
            }
        }
    }
}


void fp_s1(const float input[6][24][24], float preact[6][6][6], const float weight[1][4][4], const float bias[1]) {
    // Initialize preact to zero before accumulation
    for (int m = 0; m < 6; ++m) {
        for (int x = 0; x < 6; ++x) {
            for (int y = 0; y < 6; ++y) {
                preact[m][x][y] = 0;
            }
        }
    }

    // Nested loops to simulate the behavior of the CUDA kernel
    for (int m = 0; m < 6; ++m) {
        // for each output feature map
        for (int x = 0; x < 6; ++x) {
            // output dimensions are reduced by factor of 4
            for (int y = 0; y < 6; ++y) {
                float sum = 0.0f;
                for (int i = 0; i < 4; ++i) {
                    // kernel width
                    for (int j = 0; j < 4; ++j) {
                        // kernel height
                        // Applying weights on input and summing up to form the pooled output
                        sum += weight[0][i][j] * input[m][x * 4 + i][y * 4 + j];
                    }
                }
                preact[m][x][y] += sum; // Pooling operation with weighted sum
            }
        }
    }

    // Add bias to preact
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                preact[i][j][k] += bias[0];
            }
        }
    }
}


void fp_preact_f(const float input[6][6][6], float *preact, const float *weight, int num_outputs) {
    // Initialize the output preactivation array to zero
    
    for (int i = 0; i < num_outputs; ++i) {
        preact[i] = 0;
    }

    // Compute the dot product of the input with weights for each output unit
    
    for (int i = 0; i < num_outputs; ++i) { // output dimension
        for (int j = 0; j < 6; ++j) { // first dimension of input
            for (int k = 0; k < 6; ++k) { // second dimension of input
                for (int l = 0; l < 6; ++l) { // third dimension of input
                    int weight_idx = i * 6 * 6 * 6 + j * 6 * 6 + k * 6 + l;
                    preact[i] += weight[weight_idx] * input[j][k][l];
                }
            }
        }
    }
}


void fp_bias_f(float *preact, const float *bias, int num_outputs) {
    // Iterate through each element of the preact array and add the corresponding bias
    for (int i = 0; i < num_outputs; ++i) {
        preact[i] += bias[i];
    }
}


void bp_weight_f(float *d_weight, const float *d_preact, const float p_output[6][6][6], int num_outputs) {
    // Iterate over all indices for weight updates
    
    for (int i = 0; i < num_outputs; ++i) { // over output dimension
        for (int j = 0; j < 6; ++j) { // first dimension of input
            for (int k = 0; k < 6; ++k) { // second dimension of input
                for (int l = 0; l < 6; ++l) { // third dimension of input
                    // Calculate the gradient for each weight
                    int weight_idx = i * 6 * 6 * 6 + j * 6 * 6 + k * 6 + l;
                    d_weight[weight_idx] = d_preact[i] * p_output[j][k][l];
                }
            }
        }
    }
}

void bp_bias_f(float *bias, const float *d_preact, int num_outputs) {
    // Iterate over each bias and update it based on the gradient
    for (int i = 0; i < num_outputs; ++i) {
        bias[i] += dt * d_preact[i];
    }
}


void bp_output_s1(float d_output[6][6][6], const float *n_weight, const float *nd_preact, int num_outputs) {
    // Initialize d_output to zero before accumulation
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                d_output[i][j][k] = 0;
            }
        }
    }

    // Compute the gradient contribution from each neuron's weight and pre-activation gradient
    for (int i1 = 0; i1 < num_outputs; ++i1) { // over each output neuron
        for (int i2 = 0; i2 < 6; ++i2) { // first dimension of output
            for (int i3 = 0; i3 < 6; ++i3) { // second dimension of output
                for (int i4 = 0; i4 < 6; ++i4) { // third dimension of output
                    int weight_idx = i1 * 6 * 6 * 6 + i2 * 6 * 6 + i3 * 6 + i4;
                    d_output[i2][i3][i4] += n_weight[weight_idx] * nd_preact[i1];
                }
            }
        }
    }
}


void bp_preact_s1(float d_preact[6][6][6], const float d_output[6][6][6], const float preact[6][6][6]) {
    // Iterate through each element to calculate gradient of preactivation
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                float o = step_function(preact[i][j][k]);
                d_preact[i][j][k] = d_output[i][j][k] * o * (1 - o);
            }
        }
    }
}

void bp_weight_s1(float d_weight[1][4][4], const float d_preact[6][6][6], const float p_output[6][24][24]) {
    // Initialize d_weight to zero before accumulation
    
    for (int i = 0; i < 1; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                d_weight[i][j][k] = 0;
            }
        }
    }

    // Compute the gradient for each weight
    
    for (int i1 = 0; i1 < 1; ++i1) { // single weight map
        for (int i2 = 0; i2 < 4; ++i2) { // kernel width
            for (int i3 = 0; i3 < 4; ++i3) { // kernel height
                for (int i4 = 0; i4 < 6; ++i4) { // over each output feature map dimension
                    for (int i5 = 0; i5 < 6; ++i5) { // first dimension of output
                        for (int i6 = 0; i6 < 6; ++i6) { // second dimension of output
                            // Calculate the corresponding output location and accumulate the gradient
                            
                            d_weight[i1][i2][i3] += d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3];
                        }
                    }
                }
            }
        }
    }
}

void bp_bias_s1(float bias[1], const float d_preact[6][6][6]) {
    float sum = 0.0f;
    int total_elements = 6 * 6 * 6; // Total elements in the d_preact array

    // Sum all gradient contributions
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            for (int k = 0; k < 6; ++k) {
                sum += d_preact[i][j][k];
            }
        }
    }

    // Update the bias term by averaging the gradients
    bias[0] += dt * sum / total_elements;
}

void bp_output_c1(float d_output[6][24][24], const float n_weight[1][4][4], const float nd_preact[6][6][6]) {
    // Initialize d_output to zero before accumulation
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
            for (int k = 0; k < 24; ++k) {
                d_output[i][j][k] = 0;
            }
        }
    }

    // Calculate the contribution of each neuron's error
    for (int i1 = 0; i1 < 1; ++i1) { // only one set of weights
        for (int i2 = 0; i2 < 4; ++i2) { // kernel width
            for (int i3 = 0; i3 < 4; ++i3) { // kernel height
                for (int i4 = 0; i4 < 6; ++i4) { // over each output feature map dimension
                    for (int i5 = 0; i5 < 6; ++i5) { // reduced dimension due to pooling or stride
                        for (int i6 = 0; i6 < 6; ++i6) { // reduced dimension due to pooling or stride
                            // Map the small dimension back to the original large dimension and accumulate the error
                            int x = i5 * 4 + i2;
                            int y = i6 * 4 + i3;
                            d_output[i4][x][y] += n_weight[i1][i2][i3] * nd_preact[i4][i5][i6];
                        }
                    }
                }
            }
        }
    }
}

void bp_preact_c1(float d_preact[6][24][24], const float d_output[6][24][24], const float preact[6][24][24]) {
    // Assume step_function is a sigmoid activation function
    auto sigmoid = [](float x) {
        return 1.0f / (1.0f + exp(-x));
    };

    // Assume the derivative of the sigmoid function
    auto sigmoid_derivative = [](float x) {
        float s = 1.0f / (1.0f + exp(-x));
        return s * (1 - s);
    };

    // Compute the gradient of pre-activation for each element
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
            for (int k = 0; k < 24; ++k) {
                float o = sigmoid(preact[i][j][k]);
                d_preact[i][j][k] = d_output[i][j][k] * sigmoid_derivative(preact[i][j][k]);
            }
        }
    }
}

void bp_weight_c1(float d_weight[6][5][5], const float d_preact[6][24][24], const float p_output[28][28]) {
    // Initialize d_weight to zero before accumulation
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 5; ++j) {
            for (int k = 0; k < 5; ++k) {
                d_weight[i][j][k] = 0;
            }
        }
    }

    float d = 24.0f * 24.0f;  // Normalization factor

    // Compute the gradient for each weight
    for (int i1 = 0; i1 < 6; ++i1) {
        for (int i2 = 0; i2 < 5; ++i2) {
            for (int i3 = 0; i3 < 5; ++i3) {
                for (int i4 = 0; i4 < 24; ++i4) {
                    for (int i5 = 0; i5 < 24; ++i5) {
                        d_weight[i1][i2][i3] += d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d;
                    }
                }
            }
        }
    }
}


void bp_bias_c1(float bias[6], const float d_preact[6][24][24]) {
    // Initialize accumulators for each bias to zero before accumulation
    float accumulators[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    float d = 24.0f * 24.0f;  // Normalization factor

    // Aggregate gradients for each bias
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 24; ++j) {
            for (int k = 0; k < 24; ++k) {
                accumulators[i] += d_preact[i][j][k];
            }
        }
        // Update each bias by adding the normalized accumulated gradient
        bias[i] += dt * accumulators[i] / d;
    }
}