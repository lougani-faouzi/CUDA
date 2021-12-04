# Titre du sujet
Projet Placement mémoire

# Mots clés 
Cuda, NVIDIA, CPU, GPU, cudamalloc, cudafree, performances.

## Description
Opérateur de convolution sur un maillage 2D cartésien


Au cours du projet, il y'a eu des modifications (des ajouts) dans les fichiers suivants :
convolution.cu
convolution.cuh
main.cu

pour executer la question 1 (le cas où l'initialisation du noyau se fait en 1 seul thread), il suffit de modifier dans le main.cu, enlever le commentaire (//) de la ligne 143, et le mettre dans la ligne 147.

## Compilation du projet
Le code se construit simplement avec un Makefile, pour le compiler il suffit de taper "make" dans la ligne du commande.
pour l’exécuter il suffit de taper "./convolution Ni Nj" en spécifiant Ni et Nj dans la ligne du commande.

## Question 4: Quel est l’intérêt d’un tel portage ?
L'allocateur mémoire cudaMallocPitch(), pour des tableaux en 2D, permet de s'assurer que les copies s'effectueront le plus vite possible, avec la fonction appropriée cudaMemcpy2D(). Donc un tel portage permet un gain dans le temps d'exécution de programme. 
