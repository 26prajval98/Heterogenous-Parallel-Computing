// 16CO145 - Sumukha PK
// 16CO234 - Prajval M

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>

int main(int argc, char * argv[]){

    std::ifstream input1(argv[4]);
    std::ifstream input2(argv[5]);
    std::ofstream output(argv[2]);

    double readNum;

    /* parse the input arguments */    
    std::vector <double> hostInput1;
    std::vector <double> hostInput2;

    while(input1 >> readNum){
      hostInput1.push_back(readNum);
    }

    while(input2 >> readNum){
      hostInput2.push_back(readNum);
    }

    unsigned int N = hostInput1.size();
    output << N-1 << std::endl;

    // Import host input data
    thrust::host_vector<double> h_A(N);
    h_A = hostInput1;
    thrust::host_vector<double> h_B(N);
    h_B = hostInput2;
    
    // Declare and allocate host output
    thrust::host_vector<double> h_C(N);

                                  // std::generate(h_A.begin(), h_A.end(), rand);
                                  // std::generate(h_B.begin(), h_B.end(), rand);

    // Declare and allocate thrust device input and output vectors
    thrust::device_vector<double> V1(N);                                         
    thrust::device_vector<double> V2(N);
    thrust::device_vector<double> V3(N);

    // Copy to device
    V1 = h_A;
    V2 = h_B;

    // Execute vector addition
    thrust::transform(V1.begin(), V1.end(), V2.begin(), V3.begin(),thrust::plus<double>());
                                  
                                  // thrust::copy(V1.begin(), V1.end(), h_A.begin());
                                  // thrust::copy(V2.begin(), V2.end(), h_B.begin());
    
    
    // Copy data back to host
    thrust::copy(V3.begin(), V3.end(), h_C.begin());


    // Output into file
    for(int i = 1; i < N; i++){
      output << h_C[i] << std::endl;
    }

    hostInput1.clear();
    hostInput1.shrink_to_fit();

    hostInput2.clear();
    hostInput2.shrink_to_fit();    

    return 0;
}