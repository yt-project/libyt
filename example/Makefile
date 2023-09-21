# Your path
####################################################################
MPI_PATH   := $(YOUR_MPI_PATH)
GCC_PATH   := $(YOUR_GCC_PATH)
LIBYT_PATH := $(YOUR_LIBYT_PATH)

OPTIONS :=
#OPTIONS += -DSERIAL_MODE

### DO NOT TOUCH ###
# Compile
####################################################################
BIN  := example
FILE := example.cpp

LIB     :=
RPATH   :=
INCLUDE :=

ifneq "$(filter -DSERIAL_MODE, $(OPTIONS))" "-DSERIAL_MODE"
	COMPILER := $(MPI_PATH)/bin/mpic++
else
	ifeq ($(GCC_PATH),)
		COMPILER := g++
	else
		COMPILER := $(GCC_PATH)/bin/g++
		LIB      += -L$(GCC_PATH)/lib -L$(GCC_PATH)/lib64
		RPATH    += -Wl,-rpath,$(GCC_PATH)/lib -Wl,-rpath,$(GCC_PATH)/lib64
	endif
endif

ifeq ($(LIBYT_PATH),)
	LIB     += -L../lib -lyt
	RPATH   += -Wl,-rpath,../lib
	INCLUDE += -I../include
else
	LIB     += -L$(LIBYT_PATH)/lib -lyt
	RPATH   += -Wl,-rpath,$(LIBYT_PATH)/lib
	INCLUDE += -I$(LIBYT_PATH)/include
endif

$(BIN): $(FILE)
	$(COMPILER) $(OPTIONS) $(INCLUDE) -o $(BIN) $(FILE) $(LIB) $(RPATH)

clean:
	rm -f $(BIN)
	rm -rf log
	rm -rf __pycache__
	rm -f *.png *.gif libytTimeProfile_MPI*.json
