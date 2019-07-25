/*
 * Main file to load a pretrained model and predict (prepared) test samples 
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "hd_encoder.h"
#include "hd_classifier.h"

#define MODEL_FILE "examples/language_classif/data/models/3gramm"

#define TEST_FOLDER "examples/language_classif/data/binary_test_data/"
#define TEST_FOLDER_LEN 48
#define TEST_SAMPLE_NAME "sample_00000"
#define TEST_SAMPLE_NAME_LEN 12
#define TEST_SAMPLE_NAME_IDX TEST_FOLDER_LEN + 7

char current_filename[TEST_FOLDER_LEN + TEST_SAMPLE_NAME_LEN + 1];

feature_t * load_test_sample(int sample_idx, int * n_x, class_t * y)
{
    // prepare filename
    sprintf(current_filename + TEST_SAMPLE_NAME_IDX, "%05d", sample_idx);
    // increment sample_idx

    // try to load the file
    FILE * fp = fopen(current_filename, "rb");
    if (fp == NULL) return NULL;

    int bytes_read = 0; 

    // read class idx and size
    bytes_read += fread(y, sizeof(class_t), 1, fp);
    bytes_read += fread(n_x, sizeof(int), 1, fp);

    // allocate memory of given size
    feature_t * x = malloc(sizeof(feature_t) * (*n_x));

    // read X data
    bytes_read += fread(x, sizeof(feature_t), (*n_x), fp);

    // check if the correct number of bytes were read
    if (bytes_read != 2 + *n_x) {
        printf("Failed to read file: %s!\n", current_filename);
        return NULL;
    }

    return x;
}

int main(void)
{
    // prepare data
    struct hd_encoder_t encoder;
    struct hd_classifier_t classifier;

    // initialize hamming distance
    hamming_distance_init();

    // load
    if (load(&classifier, &encoder, MODEL_FILE) != 0) {
        printf("Could not read model!\n");
        return 1;
    }
    // model is now loaded and ready to do inference!

    // prepare current_filename
    strcpy(current_filename, TEST_FOLDER);
    strcat(current_filename, TEST_SAMPLE_NAME);

    // prepare data
    int idx = 0;

    // loop through every element until file was no longer found
    int n_err = 0;
    int n_tot = 0;
    while(1) {
        // load the sample
        int n_x;
        class_t y;
        feature_t * x = load_test_sample(idx++, &n_x, &y);
        if (x == NULL) break;

        // make prediction
        class_t yhat = hd_classifier_predict(&classifier, &encoder, x, n_x);

        // check if result was the same
        n_tot++;
        if (yhat != y) {
            n_err++;
            printf("Error: True class: %d, Estimation: %d\n", y, yhat);
        }

        // free the sample up again
        free(x);
    }

    // print results
    printf("Accuracy: %f\n", 1.0 - (double)n_err / (double)n_tot);

    // free up all memory
    free(encoder.ngramm_buffer);
    free(encoder.ngramm_sum_buffer);
    free(encoder.item_buffer);
    free(encoder.item_lookup);
    free(classifier.class_vec);
    
    return 0;
}
