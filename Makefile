lib/upose.o: src/upose.cpp
	g++ -o lib/upose.o -c -fPIC src/upose.cpp -O3 -std=c++11
