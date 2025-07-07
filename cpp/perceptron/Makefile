.SUFFIXES: .cpp .o

CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -pedantic

# Source files
SRCS = MLP.cpp NeuralNetworks.cpp
# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = nnws

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJS) $(TARGET)
