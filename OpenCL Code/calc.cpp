/****
 *
 * GPU accelerated Monte Carlo simulation of the 2D Ising model
 *
 * Copyright (C) 2008 Tobias Preis (http://www.tobiaspreis.de)
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version
 * 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public
 * License along with this program; if not, see
 * http://www.gnu.org/licenses/.
 *
 * Related publication:
 *
 * T. Preis, P. Virnau, W. Paul, and J. J. Schneider,
 * Journal of Computational Physics 228, 4468-4477 (2009)
 * doi:10.1016/j.jcp.2009.03.018
 *
 */

// note: srand48() and drand48() don't work here, so we must include a package and make some modifications in the code
#include <random>
#include <CL/cl.h>
#include <iostream>
#include "header.hpp"
using namespace std;



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

// other #defines located in header.hpp

void calc(cl_device_id device, cl_int status, cl_context context, cl_command_queue commandQueue, cl_kernel kernel, cl_int err) {

	printf(" ----------------------------------------------------------------------- \n");
	printf(" *\n");
	printf(" *  GPU accelerated Monte Carlo simulation of the 2D Ising model\n");
	printf(" *\n");
	printf(" *  Copyright (C) 2008 Tobias Preis (http://www.tobiaspreis.de)\n");
	printf(" *\n");
	printf(" *  This program is free software; you can redistribute it and/or\n");
	printf(" *  modify it under the terms of the GNU General Public License\n");
	printf(" *  as published by the Free Software Foundation; either version\n");
	printf(" *  3 of the License, or (at your option) any later version.\n");
	printf(" *\n");
	printf(" *  This program is distributed in the hope that it will be useful,\n");
	printf(" *  but WITHOUT ANY WARRANTY; without even the implied warranty of\n");
	printf(" *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the\n");
	printf(" *  GNU General Public License for more details.\n");
	printf(" *\n");
	printf(" *  You should have received a copy of the GNU General Public\n");
	printf(" *  License along with this program; if not, see\n");
	printf(" *  http://www.gnu.org/licenses/\n");
	printf(" *\n");
	printf(" *  Related publication:\n");
	printf(" *\n");
	printf(" *  T. Preis, P. Virnau, W. Paul, and J. J. Schneider,\n");
	printf(" *  Journal of Computational Physics 228, 4468-4477 (2009)\n");
	printf(" *  doi:10.1016/j.jcp.2009.03.018\n");
	printf(" *\n");

	printf(" ----------------------------- Ising model ----------------------------- \n");
	printf(" Number of Spins: %d \n", N);
	printf(" Start Temperature: %f \n", T_START);
	printf(" Decreasing Factor: %f \n", T_FACTOR);
	printf(" Final Temperature: %f \n", T_END);
	printf(" Global Iterations: %d \n", GLOBAL_ITERATIONS);

	//Init
	//Allocate and init host memory for output arrays
	// h_ for host
	int num_entries = 0;
	// the following loop makes num_entries = the number of temperatures at which
	// the monte carlo simulation is run
	for (double t = T_START; t >= T_END; t = t * T_FACTOR) num_entries++;
	// mem_out_size is the size of the output arrays for temperatures and energies,
	// which obviously should be a float
	unsigned int mem_out_size = sizeof(float) * num_entries;
	// h_T is the host (final) output array for the Temperatures
	float* h_T = (float*)malloc(mem_out_size);
	// h_E is the host (final) output array for the Energies
	float* h_E = (float*)malloc(mem_out_size);
	// mem_ref_out_size is an array of size num_entries, each entry is a double
	unsigned int mem_ref_out_size = sizeof(double) * num_entries;
	// h_ref_E is the reference array for the CPU
	double* h_ref_E = (double*)malloc(mem_ref_out_size);
	// the following loop fills h_T with the appropriate temperatures
	num_entries = 0;
	for (double t = T_START; t >= T_END; t = t * T_FACTOR) {
		h_T[num_entries] = t;
		num_entries++;
	}

	//Allocate and init host memory for simulation arrays
	unsigned int mem_size = sizeof(int) * N;
	unsigned int mem_size_random = sizeof(int) * BLOCK_SIZE * BLOCK_SIZE;
	// h_random_data is an integer array of size mem_size_random, meant
	// to hold BLOCK_SIZE random numbers for each of the BLOCK_SIZE blocks
	//int* h_random_data = (int*)malloc(mem_size_random);
	cl_int* h_random_data = (int*)malloc(mem_size_random);
	// h_S is an integer array of size mem_size meant to hold the Spins
	//int* h_S = (int*)malloc(mem_size);
	cl_int* h_S = (int*)malloc(mem_size);
	unsigned int mem_size_out = sizeof(int) * BLOCK_SIZE;
	// h_out is an integer array  of size mem_size_out that holds the REDUCED energies,
	// which is why it has mem_size_out elements, rather than N elements
	//int* h_out = (int*)malloc(mem_size_out);
	cl_int* h_out = (int*)malloc(mem_size_out);
	// the following loop fills h_random_data with consecutive powers of 16807
	// according to eqn. 2 in the paper; the mod operation is done in the GPU kernel
	// during the spin flip determination
	h_random_data[0] = 1;
	for (int i = 1;i < BLOCK_SIZE * BLOCK_SIZE;i++) {
		h_random_data[i] = 16807 * h_random_data[i - 1];
	}
	// the following loop initializes h_S to spin values of +/- 1 with equal
	// probability
	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0, 1.0);
	for (int i = 0;i < N;i++) {
		// rand48() doesn't work, must be replaced; if (drand48() > 0.5) h_S[i] = -1;
		double number = distribution(generator);
		//printf("number = %f; ", number);
		if (number > 0.5) h_S[i] = -1;
		else h_S[i] = 1;
	}

	/******************************* 
	NOTE: in OpenCL, the allocation and copying are done simultaneously, so the following steps
	1) setting up timer; 2) allocating memory; 3) destroying timer, printing message; 
	4) starting timer; 5) transferring memory; 6) destroying timer, printing message
	will be condensed into much less code
	********************************/

	cl_mem d_random_data = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_random, (void*) h_random_data, &err);
	cl_mem d_S = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size, (void*) h_S, &err);
	cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, mem_size_out, (void* ) NULL, &err);

	//Print spins
	if (FLAG_PRINT_SPINS) {
		// copy spins from device to host
		for (int i = 0;i < BLOCK_SIZE * 2;i++) {
			for (int j = 0;j < BLOCK_SIZE * 2;j++) {
				// this prints a row with BLOCK_SIZE*2 columns
				if (h_S[i * BLOCK_SIZE * 2 + j] > 0) printf("+ ");
				else printf("- ");
			}
			printf("\n");
		}
		printf("\n");
		status = clEnqueueReadBuffer(commandQueue, d_S, CL_TRUE, 0, mem_size, h_S, 0, NULL, NULL);
		for (int i = 0;i < BLOCK_SIZE * 2;i++) {
			for (int j = 0;j < BLOCK_SIZE * 2;j++) {
				// this prints a row with BLOCK_SIZE*2 columns
				if (h_S[i * BLOCK_SIZE * 2 + j] > 0) printf("+ ");
				else printf("- ");
			}
			printf("\n");
		}
		printf("\n");
	}

	//Calc energy
	num_entries = 0;
	const size_t numThreads[1] = { BLOCK_SIZE * BLOCK_SIZE };
	const size_t numBlocks[1] = { BLOCK_SIZE };
	for (cl_float t = T_START;t >= T_END;t = t * T_FACTOR) {
		double avg_H = 0;
		for (int global_iteration = 0;global_iteration < GLOBAL_ITERATIONS;global_iteration++) {
			cl_float temperature[1] = { t };
			cl_int trueClause[1] = { 1 };
			cl_int falseClause[1] = { 0 };
			status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)& d_S);
			status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)& d_out);
			status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& d_random_data);
			status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)& temperature);
			status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*) trueClause);
			status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, numThreads, numBlocks, 0, NULL, NULL);

			status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)& d_S);
			status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)& d_out);
			status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)& d_random_data);
			status = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)& temperature);
			status = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*) falseClause);
			status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, numThreads, numBlocks, 0, NULL, NULL);

			status = clEnqueueReadBuffer(commandQueue, d_out, CL_TRUE, 0, mem_size_out, h_out, 0, NULL, NULL);

			cl_int energy_sum = 0;
			for (int i = 0;i < BLOCK_SIZE;i++) {
				energy_sum += h_out[i];
			}
			avg_H += (float)energy_sum / N;
		}
		h_E[num_entries] = avg_H / GLOBAL_ITERATIONS;
		num_entries++;
	}

	cout << "Temperatures: " << endl;
	for (int i = 0; i <= num_entries; i++) {
		cout << h_T[i] << "; ";
	}
	cout << endl;

	if (FLAG_ENERGY) {
		cout << "Energies: " << endl;
	}
	else {
		cout << "Magnetizations: " << endl;
	}
	 
	for (int i = 0; i <= num_entries; i++) {
		cout << h_E[i] << "; ";
	}
	cout << endl;

	//Print spins
	if (FLAG_PRINT_SPINS) {
		status = clEnqueueReadBuffer(commandQueue, d_S, CL_TRUE, 0, mem_size, h_S, 0, NULL, NULL);
		for (int i = 0;i < BLOCK_SIZE * 2;i++) {
			for (int j = 0;j < BLOCK_SIZE * 2;j++) {
				if (h_S[i * BLOCK_SIZE * 2 + j] > 0) printf("+ ");
				else printf("- ");
			}
			printf("\n");
		}
	}

	//Reference solution
 	cpu_function(h_ref_E, h_S);

	//Print spins
	if (FLAG_PRINT_SPINS) {
		printf("\n");
		for (int i = 0;i < BLOCK_SIZE * 2;i++) {
			for (int j = 0;j < BLOCK_SIZE * 2;j++) {
				if (h_S[i * BLOCK_SIZE * 2 + j] > 0) printf("+ ");
				else printf("- ");
			}
			printf("\n");
		}
	}


	//Cleaning memory
	free(h_T);
	free(h_E);
	free(h_ref_E);
	free(h_random_data);
	free(h_S);
	free(h_out);
	status = clReleaseMemObject(d_random_data);	
	status = clReleaseMemObject(d_S);
	status = clReleaseMemObject(d_out);
}


/****
 *
 *  CPU function
 *
 */
void cpu_function(double* E, int* S) {

	int random = 23;
	int num_entries = 0;

	for (double t = T_START;t >= T_END;t = t * T_FACTOR) {
		printf("t = %f   ", t);
		double avg_H = 0;
		double exp_dH_4 = exp(-(4.0) / t);
		double exp_dH_8 = exp(-(8.0) / t);

		for (int global_iteration = 0;global_iteration < GLOBAL_ITERATIONS;++global_iteration) {
			if (FLAG_ENERGY) {
				//Energy
				double H = 0;
				for (int x = 0;x < n;++x) {
					for (int y = 0;y < n;++y) {
						int xr = x + 1, yd = y + 1;
						if (xr == n) xr = 0;
						if (yd == n) yd = 0;
						H += -S[y * n + x] * (S[y * n + xr] + S[yd * n + x]);
					}
				}
				avg_H += H / N;
			}
			else {
				//Magnetisation
				double H = 0;
				for (int x = 0;x < N;++x) {
					H += S[x];
				}
				avg_H += H / N;
			}

			for (int x = 0;x < n;++x) {
				for (int y = 0;y < n;++y) {
					// first part of checkerboard
					if ((y * (n + 1) + x) % 2 == 0) {
						int xl = x - 1, yl = y, xu = x, yu = y - 1, xr = x + 1, yr = y, xd = x, yd = y + 1;
						if (x == 0) {
							xl = n - 1;
						}
						else if (x == n - 1) {
							xr = 0;
						}
						if (y == 0) {
							yu = n - 1;
						}
						else if (y == n - 1) {
							yd = 0;
						}

						//Initial local energy
						int dH = 2 * S[y * n + x] * (
							S[yl * n + xl] +
							S[yr * n + xr] +
							S[yu * n + xu] +
							S[yd * n + xd]
							);

						if (dH == 4) {
							random = RANDOM_A * random + RANDOM_B;
							if (fabs(random * 4.656612e-10) < exp_dH_4) {
								S[y * n + x] = -S[y * n + x];
							}
						}
						else if (dH == 8) {
							random = RANDOM_A * random + RANDOM_B;
							if (fabs(random * 4.656612e-10) < exp_dH_8) {
								S[y * n + x] = -S[y * n + x];
							}
						}
						else {
							S[y * n + x] = -S[y * n + x];
						}
					}
				}
			}

			for (int x = 0;x < n;++x) {
				for (int y = 0;y < n;++y) {
					// second part of checkerboard
					if ((y * (n + 1) + x) % 2 == 1) {
						int xl = x - 1, yl = y, xu = x, yu = y - 1, xr = x + 1, yr = y, xd = x, yd = y + 1;
						if (x == 0) {
							xl = n - 1;
						}
						else if (x == n - 1) {
							xr = 0;
						}
						if (y == 0) {
							yu = n - 1;
						}
						else if (y == n - 1) {
							yd = 0;
						}

						//Initial local energy
						int dH = 2 * S[y * n + x] * (
							S[yl * n + xl] +
							S[yr * n + xr] +
							S[yu * n + xu] +
							S[yd * n + xd]
							);

						if (dH == 4) {
							random = RANDOM_A * random + RANDOM_B;
							if (fabs(random * 4.656612e-10) < exp_dH_4) {
								S[y * n + x] = -S[y * n + x];
							}
						}
						else if (dH == 8) {
							random = RANDOM_A * random + RANDOM_B;
							if (fabs(random * 4.656612e-10) < exp_dH_8) {
								S[y * n + x] = -S[y * n + x];
							}
						}
						else {
							S[y * n + x] = -S[y * n + x];
						}
					}
				}
			}
		}
		E[num_entries] = avg_H / GLOBAL_ITERATIONS;
		num_entries++;
	}
}
