# Your paths
#######################################################################################################
PYTHON_PATH    := $(YOUR_PYTHON_PATH)
PYTHON_VERSION := $(YOUR_PYTHON_VERSION)
NUMPY_PATH     := $(YOUR_NUMPY_PATH)
MPI_PATH       := $(YOUR_MPI_PATH)
READLINE_INCLUDE  := $(READLINE_INCLUDE_PATH)
READLINE_LIB      := $(READLINE_LIB_PATH)

OPTIONS :=
# Options
#######################################################################################################
# interactive mode: reloading inline script and does not halt when having errors in script
OPTIONS += -DINTERACTIVE_MODE

# turn on the timer in each libyt substep.
#OPTIONS += -DSUPPORT_TIMER

# source files
#######################################################################################################
# libyt
CC_FILE := yt_init.cpp  yt_finalize.cpp  yt_set_parameter.cpp  yt_inline.cpp  yt_add_user_parameter.cpp \
           yt_commit_grids.cpp yt_get_gridsPtr.cpp yt_get_fieldsPtr.cpp yt_get_particlesPtr.cpp         \
           yt_free_gridsPtr.cpp yt_getGridInfo.cpp yt_rma_field.cpp yt_rma_particle.cpp
CC_FILE += logging.cpp  init_python.cpp  init_libyt_module.cpp  add_dict.cpp  allocate_hierarchy.cpp    \
		   check_data.cpp append_grid.cpp get_dtype_property.cpp big_mpi.cpp

# for interactive mode
ifeq "$(filter -DINTERACTIVE_MODE, $(OPTIONS))" "-DINTERACTIVE_MODE"
CC_FILE += yt_interactive_mode.cpp func_status.cpp func_status_list.cpp
endif

# for testing and performance measuring
ifeq "$(filter -DSUPPORT_TIMER, $(OPTIONS))" "-DSUPPORT_TIMER"
CC_FILE += Timer.cpp
endif


# library name
#######################################################################################################
LIB_REALNAME := libyt.so.1.0.0
LIB_SONAME   := libyt.so.1


# flags
#######################################################################################################
CXX := $(MPI_PATH)/bin/mpicxx

LIB := -L$(PYTHON_PATH)/lib -lpython$(PYTHON_VERSION)

INCLUDE := -I../include -I$(PYTHON_PATH)/include/python$(PYTHON_VERSION) \
           -I$(NUMPY_PATH)/core/include \
           -I$(MPI_PATH)/include

ifeq "$(filter -DINTERACTIVE_MODE, $(OPTIONS))" "-DINTERACTIVE_MODE"
LIB += -L$(READLINE_LIB) -lreadline
INCLUDE += -I$(READLINE_INCLUDE)
endif

#CXXWARN_FLAG := -w1
CXXWARN_FLAG := -Wall -Wno-write-strings
CXXFLAG := $(CXXWARN_FLAG) $(INCLUDE) $(OPTIONS) -O2 -fPIC


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


# miscellaneous
#######################################################################################################
clean :
	 rm -f $(OBJ)
	 rm -f $(LIB_REALNAME)
