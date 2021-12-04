CXX := g++
CC := gcc
LINK := g++
NVCC  := nvcc

# Includes
INCLUDES = -I.

# Common flags
COMMONFLAGS += $(INCLUDES) 
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS) -O0

OBJS =  main.o convolution.o

TARGET = convolution
LINKLINE = $(LINK) -o $(TARGET) *.o

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)

main.o: main.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

convolution.o: convolution.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

clean:
	rm *.o
	rm $(TARGET)
