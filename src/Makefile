# Your paths
#######################################################################################################
PYTHON_PATH    := $(YOUR_PYTHON_PATH)
PYTHON_VERSION := $(YOUR_PYTHON_VERSION)
NUMPY_PATH     := $(YOUR_NUMPY_PATH)
MPI_PATH       := $(YOUR_MPI_PATH)
READLINE_PATH  := $(YOUR_READLINE_PATH)

# Options
#######################################################################################################
OPTIONS :=

# Interactive Mode: supports reloading inline script, active python prompt and does not halt when
# error occurs. Require readline library, add READLINE_PATH if it is not inside include search path.
#OPTIONS += -DINTERACTIVE_MODE

# Timer: measure time each libyt substep. Should switch to off in production run.
#OPTIONS += -DSUPPORT_TIMER

# source files
#######################################################################################################
# libyt
CC_FILE := yt_initialize.cpp  yt_finalize.cpp  yt_set_Parameters.cpp  yt_run.cpp  yt_set_UserParameter.cpp \
           yt_commit.cpp yt_get_GridsPtr.cpp yt_get_FieldsPtr.cpp yt_get_ParticlesPtr.cpp yt_free.cpp      \
           yt_getGridInfo.cpp yt_rma_field.cpp yt_rma_particle.cpp
CC_FILE += logging.cpp  init_python.cpp  init_libyt_module.cpp  add_dict.cpp  allocate_hierarchy.cpp       \
		   check_data.cpp append_grid.cpp get_dtype_property.cpp big_mpi.cpp

# for interactive mode
ifeq "$(filter -DINTERACTIVE_MODE, $(OPTIONS))" "-DINTERACTIVE_MODE"
CC_FILE += yt_run_InteractiveMode.cpp func_status.cpp func_status_list.cpp define_command.cpp
endif

# for testing and performance measuring
ifeq "$(filter -DSUPPORT_TIMER, $(OPTIONS))" "-DSUPPORT_TIMER"
CC_FILE += Timer.cpp
endif


# library name
#######################################################################################################
LIBYT_MAJOR_VERSION := 0
LIBYT_MINOR_VERSION := 1.0
LIB_REALNAME := libyt.so.$(LIBYT_MAJOR_VERSION).$(LIBYT_MINOR_VERSION)
LIB_SONAME   := libyt.so.$(LIBYT_MAJOR_VERSION)


# flags
#######################################################################################################
CXX := $(MPI_PATH)/bin/mpicxx

LIB := -L$(PYTHON_PATH)/lib -lpython$(PYTHON_VERSION)

INCLUDE := -I../include -I$(PYTHON_PATH)/include/python$(PYTHON_VERSION) \
           -I$(NUMPY_PATH)/core/include \
           -I$(MPI_PATH)/include

ifeq "$(filter -DINTERACTIVE_MODE, $(OPTIONS))" "-DINTERACTIVE_MODE"
LIB     += -L$(READLINE_PATH)/lib -lreadline
INCLUDE += -I$(READLINE_PATH)/include
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
$(LIB_REALNAME) : $(OBJ)
	 $(CXX) -shared -Wl,-soname,$(LIB_SONAME) -o $@ $^ $(LIB)
	 ln -s $(LIB_REALNAME) $(LIB_SONAME)
	 ln -s $(LIB_SONAME)   libyt.so
	 cp libyt.so* ../lib/.


# miscellaneous
#######################################################################################################
clean :
	 rm -f $(OBJ)
	 rm -f $(LIB_REALNAME) $(LIB_SONAME) libyt.so
