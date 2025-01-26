#include <cmath>

extern "C" {
    float add_floats(float a, float b){
        return a + b;
    }

    void square_array(float* array, long size){
        for(size_t i = 0; i < size; i++) {
            array[i] = array[i] * array[i];
        }
    }
}

