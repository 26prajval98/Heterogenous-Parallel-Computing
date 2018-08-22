// 16CO145 Sumukha PK
// 16CO234 Prajval M

#include "wb.h"

#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>

struct gt
{
  __host__ __device__
  bool operator()(int x)
  {
    return x > 127;
  }
};

int main(int argc, char *argv[])
{
	wbArg_t args;
	int inputLength, num_bins;
	unsigned int *hostInput, *hostBins;
	std::vector<unsigned int> ip, itr;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");
	hostInput = (unsigned int *)wbImport(wbArg_getInputFile(args, 3), &inputLength);

	for(int i=0; i < inputLength; i++)
		ip.push_back(hostInput[i]);

	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);

	// Copy the input to the GPU
	wbTime_start(GPU, "Allocating GPU memory");

	thrust::device_vector<int> deviceInput(inputLength);
	deviceInput = ip;
	thrust::sort(deviceInput.begin(), deviceInput.end());

	//@@ Insert code here
	wbTime_stop(GPU, "Allocating GPU memory");

	// Determine the number of bins (num_bins) and create space on the host
	//@@ insert code here
	num_bins = deviceInput.back() + 1;
	hostBins = (unsigned int *)calloc(num_bins, sizeof(unsigned int));
			
	// Allocate a device vector for the appropriate number of bins
	//@@ insert code here

	for(int i=0; i < num_bins; i++)
		itr.push_back(i);

	thrust::device_vector<int> deviceBins(hostBins, hostBins + num_bins);
	// Create a cumulative histogram. Use thrust::counting_iterator and
	// thrust::upper_bound
	//@@ Insert code here
	thrust::device_vector<int> u_b(num_bins + 1);
	thrust::device_vector<int> i(num_bins + 1);
	i = itr;

	thrust::upper_bound(deviceInput.begin(), deviceInput.end(), i.begin(), i.end(), u_b.begin());

	// Use thrust::adjacent_difference to turn the culumative histogram
	// into a histogram.
	//@@ insert code here.

	thrust::adjacent_difference(u_b.begin(), u_b.end(), deviceBins.begin());

	// Copy the histogram to the host
	//@@ insert code here
	
	gt pred;

	// If greater than 127
	thrust::replace_if(deviceBins.begin(), deviceBins.end(), pred, 127);

	thrust::copy(deviceBins.begin(), deviceBins.end(), hostBins);

	// Check the solution is correct
	wbSolution(args, hostBins, num_bins);

	// Free space on the host
	//@@ insert code here
	free(hostBins);

	return 0;
}