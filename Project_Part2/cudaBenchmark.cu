#include <stdio.h>
#include <time.h>

__global__
void renum(unsigned char* d_x, unsigned long n)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n){
	if (d_x[i] >= 100){
	  d_x[i] = 1;
	}
	else{
	  d_x[i] = 0;
	}
  }
}

int main(void)
{
	float trialCounter[41];
	for(int i = 0; i < 41; i++){
		trialCounter[i] = 0;
	}
	unsigned long sizeCounter[41];
	for(int i = 0; i < 41; i++){
		sizeCounter[i] = 0;
	}
	int indexTrial = -1;
  for (float sizeMult = .1; sizeMult < 4.2; sizeMult += .25){
  	indexTrial += 1;
  	unsigned long N = 1e9 * sizeMult;
  	sizeCounter[indexTrial] = N;
	for(int trial = 1; trial <= 5; trial++){
	  srand(time(NULL));
	  printf("Size of array is %lu\n",N);
	  unsigned char *x, *d_x;
	  x = (unsigned char*)malloc(N*sizeof(unsigned char));

	  cudaMalloc(&d_x, N*sizeof(unsigned char)); 

	  for (int i = 0; i < N; i++) {
		x[i] = (unsigned char) rand()%((255+1)-0) + 0;
	  }

	  printf("Before ");
	  for(int i = 0; i < 20; i++){
		printf("%d ",x[i]);
	  }
	  printf("\n");

	  dim3 blockSize(1024);
	  dim3 gridSize((int)((float)N / 1024.0 + 1.0));
	clock_t tic = clock();
	  cudaMemcpy(d_x, x, N*sizeof(unsigned char), cudaMemcpyHostToDevice);

	  // Perform SAXPY on 1M elements
	  renum<<<gridSize, blockSize>>>(d_x, N);

	  cudaMemcpy(x, d_x, N*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	  clock_t toc = clock();
	  printf("After ");

	  for(int i = 0; i < 20; i++){
		printf("%d ",x[i]);
	  }
	  printf("\n");

	  cudaFree(d_x);
	  free(x);
	  printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
	  trialCounter[indexTrial] += (double)(toc - tic) / CLOCKS_PER_SEC;
  }
  }

  	printf("Here is Trial Size Numbers \n");
  	for(int i = 0; i < 41; i++){
		printf("%lu ",sizeCounter[i]);
	}
	printf("\n");

	printf("Here is Average Times \n");
  	for(int i = 0; i < 41; i++){
		trialCounter[i] = trialCounter[i] / 5;
		printf("%f ",trialCounter[i]);
	}
	printf("\n");
  return 0;
}