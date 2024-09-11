#ifndef __LIBYTPROCESSCONTROL_H__
#define __LIBYTPROCESSCONTROL_H__

#include "TimerControl.h"
#include "yt_type.h"

//-------------------------------------------------------------------------------------------------------
// Class       :  LibytProcessControl
// Description :  Singleton to deal with libyt internal process. (not thread-safe)
//-------------------------------------------------------------------------------------------------------
class LibytProcessControl {
public:
    // MPI info
    int my_rank;
    int my_size;

#ifdef SUPPORT_TIMER
    // Timer Control
    TimerControl timer_control;
#endif

    // Process control check point and data management for libyt
    bool libyt_initialized;
    bool param_yt_set;
    bool get_fieldsPtr;
    bool get_particlesPtr;
    bool get_gridsPtr;
    bool commit_grids;
    bool free_gridsPtr;

    // Internal libyt data
    yt_field* field_list;
    yt_particle* particle_list;
    yt_grid* grids_local;
    int* num_grids_local_MPI;

#ifdef USE_PYBIND11
    // Hierarchy
    double* grid_left_edge;
    double* grid_right_edge;
    int* grid_dimensions;
    long* grid_parent_id;
    int* grid_levels;
    int* proc_num;
#endif

    // Singleton Methods
    LibytProcessControl(const LibytProcessControl& other) = delete;
    LibytProcessControl& operator=(const LibytProcessControl& other) = delete;
    static LibytProcessControl& Get();

    // Methods
    void Initialize();

private:
    LibytProcessControl();
    static LibytProcessControl s_Instance;
};

#endif  // __LIBYTPROCESSCONTROL_H__
