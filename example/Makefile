# Your path
MPI_PATH       := $(YOUR_MPI_PATH)
PYTHON_PATH    := $(YOUR_PYTHON_PATH)
PYTHON_VERSION := $(YOUR_PYTHON_VERSION)
READLINE_PATH  := $(YOUR_READLINE_PATH)

BIN  := example
FILE := example.cpp
COMPILER := $(MPI_PATH)/bin/mpic++

$(BIN): $(FILE)
	$(COMPILER) -o $(BIN) $(FILE) -I../include \
		-L$(READLINE_PATH)/lib -lreadline \
	    -L$(PYTHON_PATH)/lib -lpython$(PYTHON_VERSION) \
		-L../lib -lyt
	
clean:
	rm -f $(BIN)
	rm -rf log
	rm -rf __pycache__
	rm -f *.png *.gif RecordTime_*
