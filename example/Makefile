# Your path
MPI_PATH := $(YOUR_MPI_PATH)

BIN  := example
FILE := example.cpp
COMPILER := $(MPI_PATH)/bin/mpic++

$(BIN): $(FILE)
	$(COMPILER) -o $(BIN) $(FILE) -I../include -L../src -lyt
	
clean:
	rm -f $(BIN)
	rm -rf log
	rm -rf __pycache__
	rm -f *.png *.gif RecordTime_*
