#ifndef _CONVOLUTION_H
#define _CONVOLUTION_H

struct noyau_t
{
    int R;
    int KMAX;

    int *indi;
    int *indj;

    float *C;
};

void init_noyau(noyau_t *noyau, int R);
void free_noyau(noyau_t *noyau);

void convolution(
	noyau_t *noyau,
	/* IN */
	float **A, 
	int Ni, 
	int Nj,
	/* OUT */
	float **B
	);

#endif

