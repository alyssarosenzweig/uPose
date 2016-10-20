LIBS=-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_objdetect -lopencv_video -lopencv_contrib -L../lib ../lib/upose.o

webcam: webcam.cpp
	g++ -o webcam webcam.cpp $(LIBS) -I../include -pg
