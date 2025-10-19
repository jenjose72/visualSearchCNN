#ifndef __IMAGE_LOADER_H__
#define __IMAGE_LOADER_H__

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#endif

typedef struct image_data {
    double data[28][28]; /* 28x28 grayscale data for the image */
    unsigned int label;  /* label: 0=Belts, 1=Keyboard, 2=Shoes, 3=Watch */
} image_data;

/* Load images from a directory and assign a label */
static int load_images_from_directory(const char *dir_path, unsigned int label, 
                                       std::vector<image_data> &dataset) {
    int loaded_count = 0;
    
#ifdef _WIN32
    WIN32_FIND_DATAA findData;
    char searchPath[512];
    snprintf(searchPath, sizeof(searchPath), "%s\\*.*", dir_path);
    
    HANDLE hFind = FindFirstFileA(searchPath, &findData);
    if (hFind == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "Error: Cannot open directory %s\n", dir_path);
        return 0;
    }
    
    do {
        if (findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            continue; // Skip directories
        }
        
        const char *filename = findData.cFileName;
        const char *ext = strrchr(filename, '.');
        if (!ext) continue;
        
        // Check if it's an image file
        if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0 || 
            strcmp(ext, ".png") == 0 || strcmp(ext, ".webp") == 0 ||
            strcmp(ext, ".JPG") == 0 || strcmp(ext, ".JPEG") == 0 || 
            strcmp(ext, ".PNG") == 0 || strcmp(ext, ".WEBP") == 0) {
            
            char filepath[512];
            snprintf(filepath, sizeof(filepath), "%s\\%s", dir_path, filename);
            
            // Load image
            int width, height, channels;
            unsigned char *img = stbi_load(filepath, &width, &height, &channels, 1); // Force grayscale
            
            if (img) {
                // Resize to 28x28
                unsigned char resized[28 * 28];
                stbir_resize_uint8_linear(img, width, height, 0,
                                         resized, 28, 28, 0,
                                         STBIR_1CHANNEL);
                
                // Create image_data structure
                image_data img_data;
                img_data.label = label;
                
                // Normalize to [0, 1]
                for (int i = 0; i < 28; i++) {
                    for (int j = 0; j < 28; j++) {
                        img_data.data[i][j] = resized[i * 28 + j] / 255.0;
                    }
                }
                
                dataset.push_back(img_data);
                loaded_count++;
                
                stbi_image_free(img);
            } else {
                fprintf(stderr, "Warning: Failed to load %s\n", filepath);
            }
        }
    } while (FindNextFileA(hFind, &findData));
    
    FindClose(hFind);
#else
    DIR *dir = opendir(dir_path);
    if (!dir) {
        fprintf(stderr, "Error: Cannot open directory %s\n", dir_path);
        return 0;
    }
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) { // Regular file
            const char *filename = entry->d_name;
            const char *ext = strrchr(filename, '.');
            if (!ext) continue;
            
            // Check if it's an image file
            if (strcmp(ext, ".jpg") == 0 || strcmp(ext, ".jpeg") == 0 || 
                strcmp(ext, ".png") == 0 || strcmp(ext, ".webp") == 0) {
                
                char filepath[512];
                snprintf(filepath, sizeof(filepath), "%s/%s", dir_path, filename);
                
                // Load image
                int width, height, channels;
                unsigned char *img = stbi_load(filepath, &width, &height, &channels, 1); // Force grayscale
                
                if (img) {
                    // Resize to 28x28
                    unsigned char resized[28 * 28];
                    stbir_resize_uint8_linear(img, width, height, 0,
                                             resized, 28, 28, 0,
                                             STBIR_1CHANNEL);
                    
                    // Create image_data structure
                    image_data img_data;
                    img_data.label = label;
                    
                    // Normalize to [0, 1]
                    for (int i = 0; i < 28; i++) {
                        for (int j = 0; j < 28; j++) {
                            img_data.data[i][j] = resized[i * 28 + j] / 255.0;
                        }
                    }
                    
                    dataset.push_back(img_data);
                    loaded_count++;
                    
                    stbi_image_free(img);
                }
            }
        }
    }
    closedir(dir);
#endif
    
    return loaded_count;
}

/* Load all images from the four categories */
static int load_custom_dataset(image_data **data, unsigned int *count, 
                               const char *base_path = "data") {
    std::vector<image_data> dataset;
    
    char dir_path[256];
    
    // Load Belts (label 0)
    snprintf(dir_path, sizeof(dir_path), "%s/Belts", base_path);
    int belts_count = load_images_from_directory(dir_path, 0, dataset);
    fprintf(stdout, "Loaded %d images from Belts\n", belts_count);
    
    // Load Keyboard (label 1)
    snprintf(dir_path, sizeof(dir_path), "%s/Keyboard", base_path);
    int keyboard_count = load_images_from_directory(dir_path, 1, dataset);
    fprintf(stdout, "Loaded %d images from Keyboard\n", keyboard_count);
    
    // Load Shoes (label 2)
    snprintf(dir_path, sizeof(dir_path), "%s/Shoes", base_path);
    int shoes_count = load_images_from_directory(dir_path, 2, dataset);
    fprintf(stdout, "Loaded %d images from Shoes\n", shoes_count);
    
    // Load Watch (label 3)
    snprintf(dir_path, sizeof(dir_path), "%s/Watch", base_path);
    int watch_count = load_images_from_directory(dir_path, 3, dataset);
    fprintf(stdout, "Loaded %d images from Watch\n", watch_count);
    
    *count = dataset.size();
    
    if (*count == 0) {
        fprintf(stderr, "Error: No images loaded!\n");
        return -1;
    }
    
    // Allocate memory and copy data
    *data = (image_data *)malloc(sizeof(image_data) * (*count));
    for (unsigned int i = 0; i < *count; i++) {
        (*data)[i] = dataset[i];
    }
    
    fprintf(stdout, "Total images loaded: %d\n", *count);
    return 0;
}

/* Split dataset into train and test sets (80/20 split) */
static void split_dataset(image_data *all_data, unsigned int total_count,
                          image_data **train_set, unsigned int *train_cnt,
                          image_data **test_set, unsigned int *test_cnt) {
    // Shuffle the dataset
    for (unsigned int i = total_count - 1; i > 0; i--) {
        unsigned int j = rand() % (i + 1);
        image_data temp = all_data[i];
        all_data[i] = all_data[j];
        all_data[j] = temp;
    }
    
    // 80% train, 20% test
    *train_cnt = (unsigned int)(total_count * 0.8);
    *test_cnt = total_count - *train_cnt;
    
    *train_set = (image_data *)malloc(sizeof(image_data) * (*train_cnt));
    *test_set = (image_data *)malloc(sizeof(image_data) * (*test_cnt));
    
    memcpy(*train_set, all_data, sizeof(image_data) * (*train_cnt));
    memcpy(*test_set, all_data + *train_cnt, sizeof(image_data) * (*test_cnt));
    
    fprintf(stdout, "Train set: %d images, Test set: %d images\n", *train_cnt, *test_cnt);
}

#endif /* __IMAGE_LOADER_H__ */
