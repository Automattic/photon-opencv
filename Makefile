PHP_CONFIG=php-config
CXXFLAGS=-Wall -Wextra -O3 -std=c++17 -fpic \
		`pkg-config --cflags libheif opencv4 exiv2 lcms2 \
		| sed -E "s/(^| )-I/\1-isystem /g"` \
		`$(PHP_CONFIG) --includes | sed -E "s/(^| )-I/\1-isystem /g"`
LDLIBS=-lphpcpp `pkg-config --libs libheif opencv4 exiv2 lcms2`
LDFLAGS=-shared

OBJECTS=photon-opencv.o

all: photon-opencv.so

photon-opencv.so: $(OBJECTS)
	$(CC) $(CXXFLAGS) $(OBJECTS) -o $@ $(LDFLAGS) $(LDLIBS)

install: all
	install photon-opencv.so `$(PHP_CONFIG) --extension-dir`

clean:
	rm -f photon-opencv.so $(OBJECTS)
