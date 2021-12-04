#ifndef _CONVOLUTION_H
#define _CONVOLUTION_H


#define DIM_BX 32
#define DIM_BY 16

#define R_NOYAU 2

struct noyau_t
{
    int R;
    int KMAX;

    int *indi;
    int *indj;

    float *C;
};

noyau_t *init_noyau(noyau_t *h_noyau, int R);
//Question1
noyau_t *init_1thread_noyau(noyau_t *h_noyau, int R);
void free_noyau(noyau_t *h_noyau, noyau_t *d_noyau);

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
	);

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
	);
#endif

