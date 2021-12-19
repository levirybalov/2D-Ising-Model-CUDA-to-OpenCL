#include <CL/cl.h>

#define FLAG_PRINT_SPINS 0
// define FLAG_ENERGY 1 to calculate energy, FLAG_ENERGY 0 to calculate magnetization
#define FLAG_ENERGY 1
#define T_START 3.00
#define T_FACTOR 0.999
#define T_END 2.00
#define GLOBAL_ITERATIONS 100
#define RANDOM_A 1664525
#define RANDOM_B 1013904223

#define BLOCK_SIZE 256

// n = one side of the lattice; N = number of lattice sites
const unsigned int N = 4 * BLOCK_SIZE * BLOCK_SIZE;
const unsigned int n = 2 * BLOCK_SIZE;

void calc(cl_device_id device, cl_int status, cl_context context, cl_command_queue commandQueue, cl_kernel kernel, cl_int err);
void cpu_function(double* E, int* S);

