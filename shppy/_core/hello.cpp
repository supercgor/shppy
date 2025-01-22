#include <cmath>

#ifdef _WIN32
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT
#endif

extern "C" {
    DLLEXPORT float add_floats(float a, float b){
        return a + b;
    }

    DLLEXPORT void square_array(float* array, long size){
        for(size_t i = 0; i < size; i++) {
            array[i] = array[i] * array[i];
        }
    }
}
