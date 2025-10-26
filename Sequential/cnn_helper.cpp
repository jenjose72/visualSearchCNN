#include "cnn_helper.h"
#include <cstdlib>

int epoch_delay_ms = 0;

void apply_epoch_delay() {
    if (epoch_delay_ms <= 0) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(epoch_delay_ms));
}

void init_epoch_delay_from_env() {
    const char* env = std::getenv("VISUALSEARCH_DELAY_MS");
    if (!env) return;
    int v = atoi(env);
    if (v > 0) epoch_delay_ms = v;
}
