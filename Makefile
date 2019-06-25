CXXFLAGS=-Wall -Wextra -O2 -std=c++17 -fpic \
		`pkg-config --cflags opencv4 exiv2 lcms2 | sed -E "s/(^| )-I/\1-isystem /g"` \
		`php-config --includes | sed -E "s/(^| )-I/\1-isystem /g"`
LDLIBS=-lphpcpp `pkg-config --libs opencv4 exiv2 lcms2`
LDFLAGS=-shared

OBJECTS=photon-opencv.o

all: photon-opencv.so

photon-opencv.so: $(OBJECTS)
	$(CC) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f photon-opencv.so $(OBJECTS)
