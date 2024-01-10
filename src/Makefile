# Dependency Paths: other required dependencies
#######################################################################################################
PYTHON_PATH    := $(YOUR_PYTHON_PATH)
PYTHON_VERSION := $(YOUR_PYTHON_VERSION)
NUMPY_PATH     := $(YOUR_NUMPY_PATH)
MPI_PATH       := $(YOUR_MPI_PATH)
GCC_PATH       := $(YOUR_GCC_PATH)
READLINE_PATH  := $(YOUR_READLINE_PATH)

# Installation Paths: if not set, it will install to current directory
#######################################################################################################
INSTALL_PREFIX := $(YOUR_INSTALL_PREFIX)

# Options
#######################################################################################################
OPTIONS :=

# Serial Mode: compile and run libyt without using MPI.
#OPTIONS += -DSERIAL_MODE

# Interactive Mode: supports reloading inline script, active python prompt and does not halt when
# error occurs. Require readline library, add READLINE_PATH if it is not inside include search path.
#OPTIONS += -DINTERACTIVE_MODE

# Supports Jupyter Notebook UI for interactive mode, must enable INTERACTIVE_MODE
# (Use CMake for this option)
#OPTIONS += -DJUPYTER_KERNEL

# Timer: measure time each libyt substep. Should switch to off in production run.
#OPTIONS += -DSUPPORT_TIMER


#### DO NOT TOUCH ####
# library name and version
#######################################################################################################
LIBYT_MAJOR_VERSION := 0
LIBYT_MINOR_VERSION := 1
LIBYT_MICRO_VERSION := 0
LIB_REALNAME := libyt.so.$(LIBYT_MAJOR_VERSION).$(LIBYT_MINOR_VERSION).$(LIBYT_MICRO_VERSION)
LIB_SONAME   := libyt.so.$(LIBYT_MAJOR_VERSION)


# source files
#######################################################################################################
# libyt
CC_FILE := yt_initialize.cpp  yt_finalize.cpp  yt_set_Parameters.cpp  yt_run.cpp  yt_set_UserParameter.cpp \
           yt_commit.cpp yt_get_GridsPtr.cpp yt_get_FieldsPtr.cpp yt_get_ParticlesPtr.cpp yt_free.cpp      \
           yt_getGridInfo.cpp yt_rma_field.cpp yt_rma_particle.cpp yt_run_InteractiveMode.cpp              \
           yt_run_JupyterKernel.cpp
CC_FILE += logging.cpp  init_python.cpp  init_libyt_module.cpp  add_dict.cpp  allocate_hierarchy.cpp       \
		   check_data.cpp append_grid.cpp get_dtype_property.cpp func_status.cpp                           \
		   func_status_list.cpp define_command.cpp LibytProcessControl.cpp print_data.cpp                  \
		   Timer.cpp TimerControl.cpp libyt_python_shell.cpp


# flags
#######################################################################################################
LIB     :=
RPATH   :=
INCLUDE :=

# GCC/MPI compiler
ifneq "$(filter -DSERIAL_MODE, $(OPTIONS))" "-DSERIAL_MODE"
	CXX := $(MPI_PATH)/bin/mpicxx
else
	ifneq ($(GCC_PATH),)
		CXX     := $(GCC_PATH)/bin/g++
		LIB     += -L$(GCC_PATH)/lib -L$(GCC_PATH)/lib64
		RPATH   += -Wl,-rpath,$(GCC_PATH)/lib -Wl,-rpath,$(GCC_PATH)/lib64
	else
		CXX := g++
	endif
endif

# libyt include
INCLUDE += -I../include

# Python, NumPy
LIB     += -L$(PYTHON_PATH)/lib -lpython$(PYTHON_VERSION)
RPATH   += -Wl,-rpath,$(PYTHON_PATH)/lib
INCLUDE += -I$(PYTHON_PATH)/include/python$(PYTHON_VERSION) \
           -I$(NUMPY_PATH)/core/include

# readline
ifeq "$(filter -DINTERACTIVE_MODE, $(OPTIONS))" "-DINTERACTIVE_MODE"
	ifneq ($(READLINE_PATH),)
		LIB     += -L$(READLINE_PATH)/lib
		RPATH   += -Wl,-rpath,$(READLINE_PATH)/lib
		INCLUDE += -I$(READLINE_PATH)/include
	endif
	LIB += -lreadline
endif

#CXXWARN_FLAG := -w1
CXXWARN_FLAG := -Wall -Wno-write-strings
CXXFLAG := $(CXXWARN_FLAG) $(INCLUDE) $(OPTIONS) -O2 -fPIC -std=c++11

# rules and targets
#######################################################################################################
OBJ_PATH := ./obj
OBJ := $(patsubst %.cpp, $(OBJ_PATH)/%.o, $(CC_FILE))

$(OBJ_PATH)/%.o : %.cpp
	$(CXX) $(CXXFLAG) -o $@ -c $<


# link
#######################################################################################################
UNAME_S := $(shell uname -s)
$(LIB_REALNAME) : $(OBJ)
ifeq ($(UNAME_S),Linux)
	$(CXX) -shared -Wl,-soname,$(LIB_SONAME) -o $@ $^ $(LIB) $(RPATH)
endif
ifeq ($(UNAME_S),Darwin)
	$(CXX) -shared -Wl,-install_name,$(LIB_SONAME) -o $@ $^ $(LIB) $(RPATH)
endif
	ln -sf $(LIB_REALNAME) $(LIB_SONAME)
	ln -sf $(LIB_SONAME)   libyt.so


# install
#######################################################################################################
INSTALL_LIB     := $(LIB_REALNAME) $(LIB_SONAME) libyt.so
INSTALL_INCLUDE := ../include/libyt.h\
                   ../include/yt_type.h\
                   ../include/yt_type_param_libyt.h\
                   ../include/yt_type_param_yt.h\
                   ../include/yt_type_grid.h\
                   ../include/yt_type_field.h\
                   ../include/yt_type_particle.h\
                   ../include/yt_type_array.h\
                   ../include/yt_macro.h

install : $(INSTALL_LIB)
	install -d ../lib
	install -m 644 $(LIB_REALNAME) ../lib
	ln -sf ../lib/$(LIB_REALNAME) ../lib/$(LIB_SONAME)
	ln -sf ../lib/$(LIB_SONAME)   ../lib/libyt.so
ifneq ($(INSTALL_PREFIX),)
	install -d $(INSTALL_PREFIX)/lib
	install -m 644 $(LIB_REALNAME) $(INSTALL_PREFIX)/lib
	ln -sf $(INSTALL_PREFIX)/lib/$(LIB_REALNAME) $(INSTALL_PREFIX)/lib/$(LIB_SONAME)
	ln -sf $(INSTALL_PREFIX)/lib/$(LIB_SONAME)   $(INSTALL_PREFIX)/lib/libyt.so
	install -d $(INSTALL_PREFIX)/include
	install -m 644 $(INSTALL_INCLUDE) $(INSTALL_PREFIX)/include
endif


# clean
#######################################################################################################
clean :
	rm -f $(OBJ)
	rm -f $(LIB_REALNAME) $(LIB_SONAME) libyt.so ../lib/*
