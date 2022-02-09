# Your paths
#######################################################################################################
PYTHON_PATH    := $(YOUR_PYTHON_PATH)
PYTHON_VERSION := $(YOUR_PYTHON_VERSION)
NUMPY_PATH     := $(YOUR_NUMPY_PATH)
MPI_PATH       := $(YOUR_MPI_PATH)


# source files
#######################################################################################################
CC_FILE := yt_init.cpp  yt_finalize.cpp  yt_set_parameter.cpp  yt_inline.cpp  yt_add_user_parameter.cpp \
           yt_commit_grids.cpp yt_get_gridsPtr.cpp yt_get_fieldsPtr.cpp yt_get_particlesPtr.cpp         \
           yt_free_gridsPtr.cpp yt_getGridInfo.cpp yt_rma_field.cpp yt_rma_particle.cpp
CC_FILE += logging.cpp  init_python.cpp  init_libyt_module.cpp  add_dict.cpp  allocate_hierarchy.cpp    \
		   check_data.cpp append_grid.cpp get_dtype_property.cpp big_mpi.cpp


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

#CXXWARN_FLAG := -w1
CXXWARN_FLAG := -Wall -Wno-write-strings
CXXFLAG := $(CXXWARN_FLAG) $(INCLUDE) $(SIMU_OPTION) -O2 -fPIC


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
