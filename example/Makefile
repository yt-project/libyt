# Your path
MPI_PATH := /home/cindytsai/software/openmpi/3.1.5-gnu/bin

BIN  := example
FILE := example.cpp
COMPILER := $(MPI_PATH)/mpic++

$(BIN): $(FILE)
	$(COMPILER) -o $(BIN) $(FILE) -I../include -L../src -lyt
	
clean:
	rm -f $(BIN)
	rm -rf log
	rm -rf __pycache__
	rm -f *.png *.gif
