
# compilation options 
#######################################################################################################
# double precision
#SIMU_OPTION += -DFLOAT8

# debug mode
#SIMU_OPTION += -DYT_DEBUG


# source files
#######################################################################################################
CC_FILE := init.cpp


# library name
#######################################################################################################
LIB_REALNAME := libyt.so.1.0.0
LIB_SONAME   := libyt.so.1


# rules and targets
#######################################################################################################
CXX := icpc

OBJ_PATH := ./obj

LIB :=

INCLUDE := -I../include

CXXWARN_FLAG := -w1

CXXFLAG  := $(CXXWARN_FLAG) $(INCLUDE) $(SIMU_OPTION) -O2 -fPIC

OBJ := $(patsubst %.cpp, $(OBJ_PATH)/%.o, $(CC_FILE))


# implicit rules
#######################################################################################################
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



