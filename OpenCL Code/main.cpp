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
	/*
	const char* filename = "kernel.cl";
	string sourceStr;
	status = convertToString(filename, sourceStr);
	const char* source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };
	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, NULL);
	*/

	const char* filename2 = "device_function_main.cl";
	string sourceStr2;
	status = convertToString(filename2, sourceStr2);
	const char* source2 = sourceStr2.c_str();
	size_t sourceSize2[] = { strlen(source2) };
	cl_program program2 = clCreateProgramWithSource(context, 1, &source2, sourceSize2, &err);
	
	/*
	const char* filename3 = "header.hpp";
	string sourceStr3;
	status = convertToString(filename3, sourceStr3);
	const char* source3 = sourceStr3.c_str();
	size_t sourceSize3[] = { strlen(source3) };

	const char* input_header_names[2] = { "device_function_main.cl", "header.hpp" };
	size_t sourceSizes[2] = { strlen(source2) , strlen(source3) };
	cl_program program2 = clCreateProgramWithSource(context, 2, input_header_names, sourceSizes, &err);
	*/

	/*Step 6: Build program. */
	/*
	status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	*/
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

	/*Step 7: Initial input,output for the host and create memory objects for the kernel*/
	/*
	const char* input = "GdkknVnqkc";
	size_t strlength = strlen(input);
	cout << "input string:" << endl;
	cout << input << endl;
	char* output = (char*)malloc(strlength + 1);

	cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (strlength + 1) * sizeof(char), (void*) input, NULL);
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, (strlength + 1) * sizeof(char), NULL, NULL);
	*/

	/*Step 8: Create kernel object */
	/*
	cl_kernel kernel = clCreateKernel(program, "helloworld", NULL);
	*/
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


	/*Step 9: Sets Kernel arguments.*/
	//status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)& inputBuffer);
	//status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)& outputBuffer);

	/*Step 10: Running the kernel.*/
	//size_t global_work_size[1] = { strlength };
	//status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL);

	/*Step 11: Read the cout put back to host memory.*/
	//status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, strlength * sizeof(char), output, 0, NULL, NULL);

	//output[strlength] = '\0';	//Add the terminal character to the end of output.
	//cout << "\noutput string:" << endl;
	//cout << output << endl;

	/*Step 12: Clean the resources.*/
	//status = clReleaseKernel(kernel);				//Release kernel.
	//status = clReleaseProgram(program);				//Release the program object.
	status = clReleaseKernel(kernel2);				//Release kernel.
	status = clReleaseProgram(program2);//status = clReleaseMemObject(inputBuffer);		//Release mem object.
	//status = clReleaseMemObject(outputBuffer);
	status = clReleaseCommandQueue(commandQueue);	//Release  Command queue.
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

	
	/*
	const char* filename2 = "device_function_main.cl";
	string sourceStr2;
	status = convertToString(filename2, sourceStr2);
	const char* source2 = sourceStr2.c_str();
	size_t sourceSize2[] = { strlen(source2) };
	cl_program program2 = clCreateProgramWithSource(context, 1, &source2, sourceSize2, &err);

	status = clBuildProgram(program2, 1, devices, NULL, NULL, NULL);

	cl_kernel kernel2 = clCreateKernel(program2, "device_function_main", &err);


	calc(devices[0], status, context, commandQueue, kernel2, err);
	*/

	return SUCCESS;
}