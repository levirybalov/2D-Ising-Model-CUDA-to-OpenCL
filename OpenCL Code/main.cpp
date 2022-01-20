/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AR E DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

// For clarity,error checking has been omitted.

//#include "calc.cpp"
#include "header.hpp"

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>

#define SUCCESS 0
#define FAILURE 1

using namespace std;

/* convert the kernel file into a string */
int convertToString(const char* filename, std::string& s)
{
	size_t size;
	char* str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if (f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size + 1];
		if (!str)
		{
			f.close();
			return 0;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout << "Error: failed to open file\n:" << filename << endl;
	return FAILURE;
}

int main(int argc, char* argv[])
{

	/*Step1: Getting platforms and choose an available one.*/
	cl_uint numPlatforms;	//the NO. of platforms
	cl_platform_id platform = NULL;	//the chosen platform
	cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		cout << "Error: Getting platforms!" << endl;
		return FAILURE;
	}

	/*For clarity, choose the first available platform. */
	if (numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];
		free(platforms);
	}

	/*Step 2:Query the platform and choose the first GPU device if has one.Otherwise use the CPU as device.*/
	cl_uint				numDevices = 0;
	cl_device_id* devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
	if (numDevices == 0)	//no GPU available.
	{
		cout << "No GPU device available." << endl;
		cout << "Choose CPU as default device." << endl;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	}
	else
	{
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
	}


	/*Step 3: Create context.*/
	cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL); 

	/*Step 4: Creating command queue associate with the context.*/
	cl_int err;
	cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, devices[0], NULL, &err);

	/*Step 5: Create program object */
	const char* filename2 = "device_function_main.cl";
	string sourceStr2;
	status = convertToString(filename2, sourceStr2);
	const char* source2 = sourceStr2.c_str();
	size_t sourceSize2[] = { strlen(source2) };
	cl_program program2 = clCreateProgramWithSource(context, 1, &source2, sourceSize2, &err);
	
	/*Step 6: Build program. */
   	status = clBuildProgram(program2, 1, devices, NULL, NULL, NULL);

	if (status == CL_BUILD_PROGRAM_FAILURE) {
		// Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program2, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

		// Allocate memory for the log
		char* log = (char*)malloc(log_size);

		// Get the log
		clGetProgramBuildInfo(program2, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

		// Print the log
		printf("%s\n", log);

		ofstream myfile;
		myfile.open("build_log.txt");
		myfile << log;
		myfile.close();
	}

	/*Step 7: Initial input,output for the host and create memory objects for the kernel
	(not necessary here) */

	/*Step 8: Create kernel object */
	cl_kernel kernel2 = clCreateKernel(program2, "device_function_main", &err);

	printf("\n\n\nNOTE: Not running on OpenCL 2.2; getDeviceAndHostTimer() and work_group_barrier() do not work!!!\n\n\n");

	cl_uint maxComputeUnits;
	status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
	cout << "maxComputeUnits = " << maxComputeUnits << endl;

	cl_uint maxWorkItemDimensions;
	status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDimensions), &maxWorkItemDimensions, NULL);
	cout << "maxWorkItemDimensions = " << maxWorkItemDimensions << endl;


	size_t maxWorkItemSizes[3];
	status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), &maxWorkItemSizes, NULL);
	cout << "maxWorkItemSizes = (" << maxWorkItemSizes[0] << ", " << maxWorkItemSizes[1] << ", " << maxWorkItemSizes[2] << ")" << endl;

	size_t maxWorkGroupSize;
	status = clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
	cout << "maxWorkGroupSize = " << maxWorkGroupSize << endl;

	printf("\n\n\n");

	calc(devices[0], status, context, commandQueue, kernel2, err);


	/*Step 9: Sets Kernel arguments.
	(not necessary here) */
	
	
	/*Step 10: Running the kernel.	
	(not necessary here) */


	/*Step 11: Read the cout put back to host memory.	
	(not necessary here) */

	/*Step 12: Clean the resources.*/
	status = clReleaseKernel(kernel2);				//Release kernel.
	status = clReleaseProgram(program2);				//Release the program object.
	status = clReleaseCommandQueue(commandQueue);			//Release  Command queue.
	status = clReleaseContext(context);				//Release context.

	/*
	if (output != NULL)
	{
		free(output);
		output = NULL;
	}
	*/
	if (devices != NULL)
	{
		free(devices);
		devices = NULL;
	}
	std::cout << "Passed!\n";

	
	return SUCCESS;
}
