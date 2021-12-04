CXX := g++
CC := gcc
LINK := nvcc
NVCC  := nvcc

# Includes
INCLUDES = -I. -I/home/etudiant/NVIDIA_GPU_Computing_SDK/C/common/inc

# Common flags
#NVCCFLAGS += -deviceemu
COMMONFLAGS += $(INCLUDES) 
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS) -O0

OBJS =  main.o convolution.o

TARGET = convolution
LINKLINE = $(LINK) -o $(TARGET) *.o 

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)

main.o: main.cu
	$(NVCC) -c $< -o $@
	#$(NVCC) $(NVCCFLAGS) -c $< -o $@

convolution.o: convolution.cu
	$(NVCC) -c $< -o $@
	#$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm *.o
	rm $(TARGET)
