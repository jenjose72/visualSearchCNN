#pragma once

#ifndef CNN_HELPER_H
#define CNN_HELPER_H

#include <chrono>
#include <thread>

// epoch_delay_ms is set from CLI (in main) or initialized from env
extern int epoch_delay_ms;

// Apply the configured epoch delay (no-op when epoch_delay_ms <= 0)
void apply_epoch_delay();

// Initialize epoch_delay_ms from the VISUALSEARCH_DELAY_MS environment variable
void init_epoch_delay_from_env();

#endif // CNN_HELPER_H
