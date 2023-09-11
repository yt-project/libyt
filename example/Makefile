# Your path
####################################################################
MPI_PATH := $(YOUR_MPI_PATH)
LIBYT_PATH := $(YOUR_LIBYT_PATH)

# Compile example
####################################################################
BIN  := example
FILE := example.cpp
COMPILER := $(MPI_PATH)/bin/mpic++

$(BIN): $(FILE)
ifeq ($(LIBYT_PATH),)
	$(COMPILER) -o $(BIN) $(FILE) -I../include -L../lib -lyt -Wl,-rpath,../lib
else
	$(COMPILER) -o $(BIN) $(FILE) -I$(LIBYT_PATH)/include -L$(LIBYT_PATH)/lib -lyt -Wl,-rpath,$(LIBYT_PATH)/lib
endif

clean:
	rm -f $(BIN)
	rm -rf log
	rm -rf __pycache__
	rm -f *.png *.gif RecordTime_*
