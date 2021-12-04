#include <stdlib.h>
#include <stdio.h>

#include "convolution.cuh"


//Question1
//on utilisant 1 seul thread
__global__ void kernel_1thread_init_noyau(
	/* IN */
	int R,
	float coeff,
	/* OUT */
	int *indi,
	int *indj,
	float *C
	)
{
    // A ECRIRE
    int k=0;
	for(int i = -R ; i <= +R ; i++)
	{
		for(int j = -R ; j <= +R ; j++)
		{
		    indi[k] = i;
		    indj[k] = j;
		    C[k] = ((i < R && j < R) ? 1 : 0) * coeff;
		    k++;
		    printf("no Error");
		}
    }
    printf("no Error");
}

//Question1
noyau_t *init_1thread_noyau(noyau_t *h_noyau, int R)
{
    // A ECRIRE
    h_noyau->R = R;
    h_noyau->KMAX = (2*R + 1)*(2*R + 1);

	float coeff = -1/float(h_noyau->KMAX);

	//noyau du device
	noyau_t *d_noyau = (struct noyau_t*) malloc(h_noyau->KMAX * sizeof(struct noyau_t));


	// Alloction sur le host 
    h_noyau->indi =   (int*)malloc(h_noyau->KMAX * sizeof(int));
    h_noyau->indj =   (int*)malloc(h_noyau->KMAX * sizeof(int));
	h_noyau->C    = (float*)malloc(h_noyau->KMAX * sizeof(float));

	//allocation des 3 vecteurs du kernel
	int *indi;
    int *indj;
    float *C;
	
	//  Allocation mémoire dans le kernel
	cudaMalloc(&indi, sizeof(int) * h_noyau->KMAX);
    cudaMalloc(&indj, sizeof(int) * h_noyau->KMAX);
	cudaMalloc(&C, sizeof(float) * h_noyau->KMAX);
	
	// Transefert de donnée du host vers le noyau
	cudaMemcpy(indi, h_noyau->indi, h_noyau->KMAX * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(indj, h_noyau->indj, h_noyau->KMAX * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(C, h_noyau->C, h_noyau->KMAX * sizeof(float), cudaMemcpyHostToDevice);

	// les dimensions
	dim3 gridSize(2*R + 1); //une dimension
	dim3 blockSize(2*R + 1);//une dimension

	//Ecriture dukernel
	kernel_1thread_init_noyau<<<gridSize, blockSize>>>(R,coeff, indi, indj, C);

	//transfere du resultat vers du device le host
	cudaMemcpy(h_noyau->indi, indi, h_noyau->KMAX * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_noyau->indj, indj, h_noyau->KMAX * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_noyau->C, C, h_noyau->KMAX * sizeof(float), cudaMemcpyDeviceToHost);
	//printf("erreur");
	// pour eviter qu'on l'execute plusieurs fois dans le kernel
	//C[R*(2*R+1)+R] = 1;
	d_noyau->indi = indi;
	d_noyau->indj = indj;
	d_noyau->C = C;
	d_noyau->R = R;
    d_noyau->KMAX = (2*R + 1)*(2*R + 1);

	return d_noyau;

}

//Question2
__global__ void kernel_init_noyau(
	/* IN */
	int R,
	float coeff,
	/* OUT */
	int *indi,
	int *indj,
	float *C
	)
{
    // A ECRIRE

    //nombre du thread
    int k = threadIdx.x + blockDim.x * blockIdx.x;

    //les indices
    int ligne = k/(2*R+1) - R;
    int colonne = k%(2*R+1) - R;

    //Initialisation des vecteurs
	if(k < (2*R+1)*(2*R+1))
	{
		indi[k] = ligne;
		indj[k] = colonne;
		C[k] = ((ligne < R && colonne < R) ? 1 : 0) * coeff;
		//printf("i = %d, j= %d %d\n", ligne, colonne);
	}
	//printf("no erreur");
}

//Question2
noyau_t *init_noyau(noyau_t *h_noyau, int R)
{
    // A ECRIRE
    h_noyau->R = R;
    h_noyau->KMAX = (2*R + 1)*(2*R + 1);

	float coeff = -1/float(h_noyau->KMAX);

	//noyau du device
	noyau_t *d_noyau = (struct noyau_t*) malloc(h_noyau->KMAX * sizeof(struct noyau_t));


	// Alloction sur le host 
    h_noyau->indi =   (int*)malloc(h_noyau->KMAX * sizeof(int));
    h_noyau->indj =   (int*)malloc(h_noyau->KMAX * sizeof(int));
	h_noyau->C    = (float*)malloc(h_noyau->KMAX * sizeof(float));

	//allocation des 3 vecteurs du kernel
	int *indi;
    int *indj;
    float *C;
	
	//  Allocation mémoire dans le kernel
	cudaMalloc(&indi, sizeof(int) * h_noyau->KMAX);
    cudaMalloc(&indj, sizeof(int) * h_noyau->KMAX);
	cudaMalloc(&C, sizeof(float) * h_noyau->KMAX);
	
	// Transefert de donnée du host vers le noyau
	cudaMemcpy(indi, h_noyau->indi, h_noyau->KMAX * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(indj, h_noyau->indj, h_noyau->KMAX * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(C, h_noyau->C, h_noyau->KMAX * sizeof(float), cudaMemcpyHostToDevice);

	// les dimensions
	dim3 gridSize(2*R + 1); //une dimension
	dim3 blockSize(2*R + 1);//une dimension

	//Ecriture dukernel
	kernel_init_noyau<<<gridSize, blockSize>>>(R,coeff, indi, indj, C);

	//transfere du resultat vers du device le host
	cudaMemcpy(h_noyau->indi, indi, h_noyau->KMAX * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_noyau->indj, indj, h_noyau->KMAX * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_noyau->C, C, h_noyau->KMAX * sizeof(float), cudaMemcpyDeviceToHost);
	//printf("erreur");
	// pour eviter qu'on l'execute plusieurs fois dans le kernel
	//C[R*(2*R+1)+R] = 1;
	d_noyau->indi = indi;
	d_noyau->indj = indj;
	d_noyau->C = C;
	d_noyau->R = R;
    d_noyau->KMAX = (2*R + 1)*(2*R + 1);

	return d_noyau;

}

void free_noyau(noyau_t *h_noyau, noyau_t *d_noyau)
{
    // A ECRIRE
    //à terminer
    free(h_noyau->indi);
    free(h_noyau->indj);
    free(h_noyau->C);
    cudaFree(d_noyau->indi);
    cudaFree(d_noyau->indj);
    cudaFree(d_noyau->C);
}

__device__ float &elt_ref(void *base_addr, size_t pitch, int i, int j)
{
    float *p_elt = (float*)((char*)base_addr + i*pitch) + j;
    return *p_elt;
}

__global__ void convol_gl(
	noyau_t *d_noyau,
	/* IN */
	float *d_buf_A,
	size_t pitchA,
	int Ni,
	int Nj,
	/* OUT */
	float *d_buf_B,
	size_t pitchB /* IN */
	)
{
    int i = d_noyau->R + blockIdx.x * blockDim.x + threadIdx.x;
    int j = d_noyau->R + blockIdx.y * blockDim.y + threadIdx.y;

    if (i < Ni+d_noyau->R && j < Nj+d_noyau->R)
    {
	float tmp_B = 0;

	for(int k = 0 ; k < d_noyau->KMAX ; k++)
	{
	    const float val_A =
		elt_ref(d_buf_A, pitchA, i+d_noyau->indi[k], j+d_noyau->indj[k]);

	    tmp_B += d_noyau->C[k] * val_A;
	}
	elt_ref(d_buf_B, pitchB, i, j) = tmp_B;
    }
}

//Question3
__global__ void convol_sh(
	noyau_t *d_noyau,
	/* IN */
	float *d_buf_A,
	size_t pitchA,
	int Ni,
	int Nj,
	/* OUT */
	float *d_buf_B,
	size_t pitchB /* IN */
	)
{
    // A ECRIRE
    //allocation d'un nouveau shared noyau
    __shared__ noyau_t *sh_noyau;
    sh_noyau = (struct noyau_t*) malloc(d_noyau->KMAX * sizeof(struct noyau_t));

    //transfere du d_noyau vers sh_noyau
    sh_noyau->indi = d_noyau->indi;
    sh_noyau->indj = d_noyau->indj;
    sh_noyau->C  = d_noyau->C;

    //allocation d'un nouveau shared buffer 
    __shared__ float *sh_buf_A;
    sh_buf_A = (float*)malloc(Ni*Nj*sizeof(float));

    //transfere du d_buf_A vers sh_buf_A
    sh_buf_A = d_buf_A;

    //les indices
    int i = d_noyau->R + blockIdx.x * blockDim.x + threadIdx.x;
    int j = d_noyau->R + blockIdx.y * blockDim.y + threadIdx.y;

    if (i < Ni+sh_noyau->R && j < Nj+sh_noyau->R)
    {
		float tmp_B = 0;

		for(int k = 0 ; k < d_noyau->KMAX ; k++)
		{
			__syncthreads();
		    const float val_A =	elt_ref(sh_buf_A, pitchA, i+sh_noyau->indi[k], j+sh_noyau->indj[k]);

		    tmp_B += sh_noyau->C[k] * val_A;
		}
		elt_ref(d_buf_B, pitchB, i, j) = tmp_B;
    }

}

