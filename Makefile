lib/upose.o: src/upose.cpp
	g++ -o lib/upose.o -C -fPIC src/upose.cpp -O3 -std=c++11
