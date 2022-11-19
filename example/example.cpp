/*
 [Description]
 This example is to show how libyt loads all the data and information into yt,
 then use yt to do analysis on the go. And also, to illustrate the basic
 usage of libyt in steps 0 - 9.

 We have a set of pre-calculated data. We assign each grid to one MPI rank
 randomly to stimulate the actual code of having grid data on different ranks.
 Then we demonstrate how to pass in the grid data and their hierarchies located
 on current MPI rank to libyt. Finally, we use those passed in data to do a
 series of analysis with yt.

 Above description can be concentrated to following steps:
     1. Assign pre-calculated grids to one MPI rank.
     2. Each MPI rank will only pass local data and their hierarchies to libyt
        ---> Here, we calculate all the grids (sim_grids) first,
             then distribute them to grids_local, to simulate the working process.
     3. Do analysis using yt.

 [Compile and Run]
   1. Compile libyt and move libyt.so.* library to ../lib/.
   2. Run this command to set LD_LIBRARY_PATH:
      sh set_ld_path.sh
   3. Update Makefile MPI_PATH.
   4. make clean; make;
   5. mpirun -np 4 --output-filename log ./example
 */

#include <stdlib.h>
#include <math.h>
#include <typeinfo>
#include <mpi.h>
#include <time.h>

// ==========================================
// libyt: 0. include libyt header
// ==========================================
#include "libyt.h"


// single or double precision in the field data
//typedef float real;
typedef double real;

// grid information
#define NGRID_1D  5   // number of root grids along each direction
#define GRID_DIM  8   // grid dimension (this example assumes cubic grids)
#define REFINE_BY 2   // refinement factor between two AMR levels
#define GHOST_CELL 1  // number of ghost cell in each direction

// convenient macros
#define SQR(a)  ( (a)*(a) )
#define CUBE(a) ( (a)*(a)*(a) )


real set_density(const double x, const double y, const double z, const double t, const double v);
void get_randArray(int *array, int length);
void level_derived_func(int list_len, long *gid_list, yt_array *data_array);
void level_derived_func_with_name(int list_len, long *gid_list, char *field, yt_array *data_array);
void par_io_get_attr(int list_len, long *gid_list, char *attribute, yt_array *data_array);
void getPositionByGID(long gid, real (*Pos)[3]);
void getLevelByGID(long gid, int *Level);
yt_grid *gridsPtr;
long num_total_grids;

//-------------------------------------------------------------------------------------------------------
// Function    :  main
// Description :  Main function
//-------------------------------------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    int myrank;
    int nrank;
    int RootRank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);


    // ==========================================
    // libyt: 1. initialize libyt
    // ==========================================
    yt_param_libyt param_libyt;
    param_libyt.verbose = YT_VERBOSE_INFO;                     // libyt log level
    param_libyt.script = "inline_script";                      // inline python script, excluding ".py"
    param_libyt.check_data = false;                            // check passed in data or not

    if (yt_init(argc, argv, &param_libyt) != YT_SUCCESS) {
        fprintf(stderr, "ERROR: yt_init() failed!\n");
        exit(EXIT_FAILURE);
    }


    // ==========================================
    // simulation: simulation settings
    // ==========================================
    const int total_steps = 11;                               // total number of evolution steps
    const double velocity = 1.0;                              // velocity for setting the density field
    const double dt = 0.05;                                   // evolution time-step
    double time = 0.0;                                        // current simulation code time
    const double box_size = 1.0;                              // simulation box size
    const double dh0 = box_size / (NGRID_1D * GRID_DIM);      // cell size at level 0
    const double dh1 = dh0 / REFINE_BY;                       // cell size at level 1
    const int num_fields = 1;                                 // number of fields
    const int num_grids = CUBE(NGRID_1D) + CUBE(REFINE_BY);   // number of grids
    const int num_species = 1;                                // number of particle types

    // simulation data stored in memory, there can be GHOST_CELL at the grid's boundary.
    real (*field_data)[num_fields][CUBE(GRID_DIM+GHOST_CELL*2)] = new real [num_grids][num_fields][CUBE(GRID_DIM+GHOST_CELL*2)];

    int grids_MPI[num_grids];                                 // records what MPI process each grid belongs to
    int num_grids_local;                                      // number of grids at current MPI process

    // assign grids to MPI processes, to simulate simulation code process.
    if (myrank == RootRank) {
        get_randArray(grids_MPI, num_grids);
    }
    MPI_Bcast(grids_MPI, num_grids, MPI_INT, RootRank, MPI_COMM_WORLD);

    // count how many local grids are there on this MPI process.
    num_grids_local = 0;
    for (int i = 0; i < num_grids; i = i + 1) {
        if (grids_MPI[i] == myrank) {
            num_grids_local = num_grids_local + 1;
        }
    }

    num_total_grids = (long) num_grids; // TODO


    // ==========================================
    // simulation: mimic the simulation loop
    // ==========================================
    for (int step = 0; step < total_steps; step++) {


        // ==========================================
        // libyt: 2. provide YT-specific parameters
        // ==========================================
        yt_param_yt param_yt;
        param_yt.frontend = "gamer";                          // simulation frontend that libyt borrows field info from
        param_yt.fig_basename = "FigName";                    // figure base name (default=Fig)
        param_yt.length_unit = 3.0857e21;                     // length unit (cm)
        param_yt.mass_unit = 1.9885e33;                       // mass unit (g)
        param_yt.time_unit = 3.1557e13;                       // time unit (sec)
        param_yt.current_time = time;                         // simulation time in code units
        param_yt.dimensionality = 3;                          // dimensionality, support 3 only
        param_yt.refine_by = REFINE_BY;                       // refinement factor between a grid and its subgrid
        param_yt.num_grids = num_grids;                       // number of grids
        param_yt.num_grids_local = num_grids_local;           // number of local grids
        param_yt.num_fields = num_fields + 1;                 // number of fields
        param_yt.num_species = num_species;                   // number of particle types (or species)

        yt_species species_list[num_species];
        species_list[0].species_name = "io";
        species_list[0].num_attr = 4;
        param_yt.species_list = species_list;                 // define name and number of attributes in each particle

        for (int d = 0; d < 3; d++) {
            param_yt.domain_dimensions[d] = NGRID_1D * GRID_DIM; // domain dimensions in [x][y][z]
            param_yt.domain_left_edge[d] = 0.0;                  // domain left edge in [x][y][z]
            param_yt.domain_right_edge[d] = box_size;            // domain right edge in [x][y][z]
            param_yt.periodicity[d] = 0;                         // periodicity in [x][y][z]
        }

        param_yt.cosmological_simulation = 0;                 // if this is a cosmological simulation or not, 0 for false
        param_yt.current_redshift = 0.5;                      // current redshift
        param_yt.omega_lambda = 0.7;                          // omega lambda
        param_yt.omega_matter = 0.3;                          // omega matter
        param_yt.hubble_constant = 0.7;                       // hubble constant

        if (yt_set_parameter(&param_yt) != YT_SUCCESS) {
            fprintf(stderr, "ERROR: yt_set_parameter() failed!\n");
            exit(EXIT_FAILURE);
        }


        // ==========================================
        // libyt: 3. add code-specific parameters
        // ==========================================
        // specific parameters for GAMER yt frontend
        const int mhd = 0;
        yt_add_user_parameter_int("mhd", 1, &mhd);
        const int srhd = 0;
        yt_add_user_parameter_int("srhd", 1, &srhd);

        // demo of some other parameters we can add
        const int user_int = 1;
        const long user_long = 2;
        const uint user_uint = 3;
        const ulong user_ulong = 4;
        const float user_float = 5.0;
        const double user_double = 6.0;
        const char *user_string = "my_string";
        const int user_int3[3] = {7, 8, 9};
        const double user_double3[3] = {10.0, 11.0, 12.0};

        yt_add_user_parameter_int("user_int", 1, &user_int);
        yt_add_user_parameter_long("user_long", 1, &user_long);
        yt_add_user_parameter_uint("user_uint", 1, &user_uint);
        yt_add_user_parameter_ulong("user_ulong", 1, &user_ulong);
        yt_add_user_parameter_float("user_float", 1, &user_float);
        yt_add_user_parameter_double("user_double", 1, &user_double);
        yt_add_user_parameter_string("user_string", user_string);
        yt_add_user_parameter_int("user_int3", 3, user_int3);
        yt_add_user_parameter_double("user_double3", 3, user_double3);


        // ==========================================
        // libyt: 4. set field information
        // ==========================================
        // get pointer of the array where we should put data to
        yt_field *field_list;
        yt_get_fieldsPtr(&field_list);

        // Density field "Dens"
        field_list[0].field_name = "Dens";
        field_list[0].field_define_type = "cell-centered";
        field_list[0].swap_axes = true;
        field_list[0].field_dtype = (typeid(real) == typeid(float)) ? YT_FLOAT : YT_DOUBLE;
        char *field_name_alias[] = {"Name Alias 1", "Name Alias 2", "Name Alias 3"};
        field_list[0].field_name_alias = field_name_alias;
        field_list[0].num_field_name_alias = 3;
        for (int d = 0; d < 6; d++) {
            field_list[0].field_ghost_cell[d] = GHOST_CELL;
        }

        // Reciprocal of density field "InvDens"
        field_list[1].field_name = "level_derived_func"; // TODO
        field_list[1].field_define_type = "derived_func";
        field_list[1].swap_axes = true;
        field_list[1].field_dtype = (typeid(real) == typeid(float)) ? YT_FLOAT : YT_DOUBLE;
        field_list[1].derived_func = level_derived_func; // TODO
        field_list[1].derived_func_with_name = level_derived_func_with_name; // TODO


        // ==========================================
        // libyt: 5. set particle information
        // ==========================================
        // get pointer of the array where we should put data to
        yt_particle *particle_list;
        yt_get_particlesPtr(&particle_list);

        // Particle type "io"
        particle_list[0].species_name = "io";
        particle_list[0].num_attr = 4;
        char *attr_name[] = {"ParPosX", "ParPosY", "ParPosZ", "Level"};
        char *attr_name_alias[] = {"grid_level"};
        for (int v = 0; v < 4; v++) {
            particle_list[0].attr_list[v].attr_name = attr_name[v];
            if (v == 3) {
                particle_list[0].attr_list[v].attr_dtype = YT_INT;
                particle_list[0].attr_list[v].num_attr_name_alias = 1;
                particle_list[0].attr_list[v].attr_name_alias = attr_name_alias;
                particle_list[0].attr_list[v].attr_display_name = "Level of the Grid";
            } else {
                particle_list[0].attr_list[v].attr_dtype = (typeid(real) == typeid(float)) ? YT_FLOAT : YT_DOUBLE;
            }
        }
        particle_list[0].coor_x = attr_name[0];
        particle_list[0].coor_y = attr_name[1];
        particle_list[0].coor_z = attr_name[2];
        particle_list[0].get_attr = par_io_get_attr;


        // ==================================================
        // simulation: generate simulation grid info and data
        // ==================================================
        yt_grid *sim_grids = new yt_grid[param_yt.num_grids];
        gridsPtr = sim_grids; // TODO

        // set level-0 grids
        int grid_order[3];
        for (grid_order[2] = 0; grid_order[2] < NGRID_1D; grid_order[2]++) {
            for (grid_order[1] = 0; grid_order[1] < NGRID_1D; grid_order[1]++) {
                for (grid_order[0] = 0; grid_order[0] < NGRID_1D; grid_order[0]++) {
                    const int gid = (grid_order[2] * NGRID_1D + grid_order[1]) * NGRID_1D + grid_order[0];
                    for (int d = 0; d < 3; d++) {
                        sim_grids[gid].left_edge[d] = grid_order[d] * GRID_DIM * dh0;
                        sim_grids[gid].right_edge[d] = sim_grids[gid].left_edge[d] + GRID_DIM * dh0;
                        sim_grids[gid].grid_dimensions[d] = GRID_DIM;
                    }
                    sim_grids[gid].id = gid;
                    sim_grids[gid].parent_id = -1;
                    sim_grids[gid].level = 0;

                    for (int k = GHOST_CELL; k < GRID_DIM + GHOST_CELL; k++) {
                        for (int j = GHOST_CELL; j < GRID_DIM + GHOST_CELL; j++) {
                            for (int i = GHOST_CELL; i < GRID_DIM + GHOST_CELL; i++) {
                                const double x = sim_grids[gid].left_edge[0] + ((i - GHOST_CELL) + 0.5) * dh0;
                                const double y = sim_grids[gid].left_edge[1] + ((j - GHOST_CELL) + 0.5) * dh0;
                                const double z = sim_grids[gid].left_edge[2] + ((k - GHOST_CELL) + 0.5) * dh0;
                                for (int v = 0; v < num_fields; v++) {
                                    field_data[gid][v][
                                            (k * (GRID_DIM + GHOST_CELL * 2) + j) * (GRID_DIM + GHOST_CELL * 2) +
                                            i] = set_density(x, y, z, time, velocity);
                                }
                            }
                        }
                    }
                }
            }
        }

        // refine the root grid with the peak density into REFINE_BY^3 subgrids
        const double peak[3] = {0.5 * box_size + velocity * time,
                                0.5 * box_size + velocity * time,
                                0.5 * box_size};
        const double grid_width = GRID_DIM * dh0;
        const int center_idx[3] = {int(peak[0] / grid_width),
                                   int(peak[1] / grid_width),
                                   int(peak[2] / grid_width)};
        const int gid_refine = (center_idx[2] * NGRID_1D + center_idx[1]) * NGRID_1D + center_idx[0];
        const int gid_offset = CUBE(NGRID_1D);

        // set level-1 grids
        for (grid_order[2] = 0; grid_order[2] < param_yt.refine_by; grid_order[2]++) {
            for (grid_order[1] = 0; grid_order[1] < param_yt.refine_by; grid_order[1]++) {
                for (grid_order[0] = 0; grid_order[0] < param_yt.refine_by; grid_order[0]++) {
                    const int gid = (grid_order[2] * param_yt.refine_by + grid_order[1]) * param_yt.refine_by + grid_order[0] + gid_offset;
                    for (int d = 0; d < 3; d++) {
                        sim_grids[gid].left_edge[d] = sim_grids[gid_refine].left_edge[d] + grid_order[d] * GRID_DIM * dh1;
                        sim_grids[gid].right_edge[d] = sim_grids[gid].left_edge[d] + GRID_DIM * dh1;
                        sim_grids[gid].grid_dimensions[d] = GRID_DIM;
                    }
                    sim_grids[gid].id = gid;
                    sim_grids[gid].parent_id = gid_refine;
                    sim_grids[gid].level = 1;

                    for (int k = GHOST_CELL; k < GRID_DIM + GHOST_CELL; k++) {
                        for (int j = GHOST_CELL; j < GRID_DIM + GHOST_CELL; j++) {
                            for (int i = GHOST_CELL; i < GRID_DIM + GHOST_CELL; i++) {
                                const double x = sim_grids[gid].left_edge[0] + ((i - GHOST_CELL) + 0.5) * dh1;
                                const double y = sim_grids[gid].left_edge[1] + ((j - GHOST_CELL) + 0.5) * dh1;
                                const double z = sim_grids[gid].left_edge[2] + ((k - GHOST_CELL) + 0.5) * dh1;
                                for (int v = 0; v < num_fields; v++) {
                                    field_data[gid][v][
                                            (k * (GRID_DIM + GHOST_CELL * 2) + j) * (GRID_DIM + GHOST_CELL * 2) +
                                            i] = set_density(x, y, z, time, velocity);
                                }
                            }
                        }
                    }
                }
            }
        }


        // ==========================================
        // libyt: 6. set grids information
        // ==========================================
        // get pointer of the array where we should put data to
        yt_grid *grids_local;
        yt_get_gridsPtr(&grids_local);

        // Load the local grids information and data to libyt.
        int index_local = 0;
        for (int gid = 0; gid < param_yt.num_grids; gid = gid + 1) {
            if (grids_MPI[gid] == myrank) {
                for (int d = 0; d < 3; d = d + 1) {
                    grids_local[index_local].left_edge[d] = sim_grids[gid].left_edge[d];              // left edge
                    grids_local[index_local].right_edge[d] = sim_grids[gid].right_edge[d];            // right edge
                    grids_local[index_local].grid_dimensions[d] = sim_grids[gid].grid_dimensions[d];  // dimensions
                }
                grids_local[index_local].particle_count_list[0] = 1;             // number of particles in each particle type
                grids_local[index_local].id = sim_grids[gid].id;                 // 0-indexed grid id
                grids_local[index_local].parent_id = sim_grids[gid].parent_id;   // 0-indexed parent id (-1 for root level grids)
                grids_local[index_local].level = sim_grids[gid].level;           // 0-indexed level

                for (int v = 0; v < num_fields; v = v + 1) {
                    grids_local[index_local].field_data[v].data_ptr = field_data[gid][v];   // field data ptr
                }
                index_local = index_local + 1;
            }
        }

        // ==========================================
        // libyt: 7. done loading information
        // ==========================================
        if (yt_commit_grids() != YT_SUCCESS) {
            fprintf(stderr, "ERROR: yt_commit_grids() failed!\n");
            exit(EXIT_FAILURE);
        }

        for (long gid = 0; gid < num_grids; gid++) {
            printf("[GID = %ld]\n", gid);

            double edge[3];
            yt_getGridInfo_LeftEdge(gid, &edge);
            printf("Left Edge: %lf, %lf, %lf\n", edge[0], edge[1], edge[2]);

            yt_getGridInfo_RightEdge(gid, &edge);
            printf("Right Edge: %lf, %lf, %lf\n", edge[0], edge[1], edge[2]);

            long parent_id;
            int l, proc_num;
            yt_getGridInfo_ParentId(gid, &parent_id);
            yt_getGridInfo_Level(gid, &l);
            yt_getGridInfo_ProcNum(gid, &proc_num);

            printf("Parent ID: %ld\n", parent_id);
            printf("Level: %d\n", l);
            printf("yt_getGridInfo_ ProcNum: %d\n", proc_num);
            printf("grids_MPI: %d\n", grids_MPI[gid]);

            long count;
            yt_getGridInfo_ParticleCount(gid, "io", &count);
            printf("Particle Count = %ld\n", count);
        }


//    =============================================================
//    7. perform inline analysis, execute function in python script
//    =============================================================
//    *** libyt API ***
        if (yt_inline_argument("yt_inline_ProjectionPlot", 1, "\'density\'") != YT_SUCCESS) {
            fprintf(stderr, "ERROR: yt_inline_argument() failed!\n");
            exit(EXIT_FAILURE);
        }

        if (yt_inline("yt_inline_ProfilePlot") != YT_SUCCESS) {
            fprintf(stderr, "ERROR: yt_inline() failed!\n");
            exit(EXIT_FAILURE);
        }

        if (yt_inline("yt_inline_ParticlePlot") != YT_SUCCESS) {
            fprintf(stderr, "ERROR: yt_inline() failed!\n");
            exit(EXIT_FAILURE);
        }

        if (yt_inline("yt_derived_field_demo") != YT_SUCCESS) {
            fprintf(stderr, "ERROR: yt_derived_field_demo() failed!\n");
            exit(EXIT_FAILURE);
        }

        if (yt_inline("test_function") != YT_SUCCESS) {
            fprintf(stderr, "ERROR: test_function() failed!\n");
            exit(EXIT_FAILURE);
        }

//    =============================================================================
//    8. end of the inline-analysis at this step, free grid info loaded into python
//    =============================================================================
//    *** libyt API ***
        if (yt_free_gridsPtr() != YT_SUCCESS) {
            fprintf(stderr, "ERROR: yt_free_gridsPtr() failed!\n");
            exit(EXIT_FAILURE);
        }


//    free resources
        delete[] sim_grids;

        time += dt;
    } // for (int step=0; step<total_steps; step++)

// ==========================================
// 9. exit libyt
// ==========================================
// *** libyt API ***
    if (yt_finalize() != YT_SUCCESS) {
        fprintf(stderr, "ERROR: yt_finalize() failed!\n");
        exit(EXIT_FAILURE);
    }


    delete[] field_data;
    delete[] grids_MPI;

    /*
    MPI Finalize
     */
    MPI_Finalize();

    return EXIT_SUCCESS;

} // FUNCTION : main



//-------------------------------------------------------------------------------------------------------
// Function    :  set_density
// Description :  Return density at given coordinates and time
//-------------------------------------------------------------------------------------------------------
real set_density(const double x, const double y, const double z, const double t, const double v) {

    const double center[3] = {0.5 + v * t, 0.5 + v * t, 0.5};   // drift with v along (1,1,0)
    const double sigma = 0.05;
    const double amplitude = 1.0e6;
    const double background = 1.0;

    return amplitude * exp(-0.5 * (SQR(x - center[0]) + SQR(y - center[1]) + SQR(z - center[2])) / SQR(sigma)) +
           background;

} // FUNCTION : set_density

//-------------------------------------------------------------------------------------------------------
// Function    :  get an array of random number in range 0 ~ NRank-1
// Description :  To random distribute grids to MPI rank, so that we can mimic what happen in real world
//                simulation code.
//-------------------------------------------------------------------------------------------------------
void get_randArray(int *array, int length) {

    int NRank;
    MPI_Comm_size(MPI_COMM_WORLD, &NRank);

    srand((unsigned) time(0));

    for (int i = 0; i < length; i = i + 1) {
        array[i] = rand() % NRank;
    }
} // FUNCTION : get_randArray


//-------------------------------------------------------------------------------------------------------
// Function    :  level_derived_func
// Description :  Generate data in field name "level_derived_func" by gid.
//
// Notes       :  1. Prototype must be void func(int, long*, yt_array*) or void func(int, long*, char*, yt_array*).
//                   See Parameter for its usage.
//                2. This function will be concatenated into python C extension, so that yt can get
//                   derived field data when needed.
//                3. In this example, we will write the grid's level to data.
//                4. Write results to double *data in [z][y][x] order, since we set swap_axes = true for
//                   this field.
//
// Parameter   : int       list_len  : number of gid in gid_list.
//               long     *gid_list  : a list of gid field data to prepare.
//               yt_array *data_array: write field data inside this yt_array correspondingly.
//-------------------------------------------------------------------------------------------------------
void level_derived_func(int list_len, long *gid_list, yt_array *data_array) {
    // loop over gid_list, and fill in grid data inside data_array.
    for (int lid = 0; lid < list_len; lid++) {
        // Get grid level by gid.
        int level;
        getLevelByGID(gid_list[lid], &level);

        // Get grid dimension by libyt API.
        int dim[3];
        yt_getGridInfo_Dimensions(gid_list[lid], &dim);

        // Fill in data
        int index;
        for (int k = 0; k < dim[2]; k++) {
            for (int j = 0; j < dim[1]; j++) {
                for (int i = 0; i < dim[0]; i++) {
                    index = k * dim[1] * dim[0] + j * dim[0] + i;
                    // Get the point and store data.
                    ((real *) data_array[lid].data_ptr)[index] = (real) level;
                }
            }
        }
    }
}


//-------------------------------------------------------------------------------------------------------
// Function    :  level_derived_func_with_name
// Description :  Generate data in field name "level_derived_func_with_name" by gid and a given field name.
//
// Notes       :  1. Prototype must be void func(int, long*, yt_array*) or void func(int, long*, char*, yt_array*).
//                   See Parameter for its usage.
//                2. This function will be concatenated into python C extension, so that yt can get
//                   derived field data when needed.
//                3. In this example, we will write the grid's level to data.
//                4. Write results to double *data in [z][y][x] order, since we set swap_axes = true for
//                   this field.
//
// Parameter   : int       list_len  : number of gid in gid_list.
//               long     *gid_list  : a list of gid field data to prepare.
//               char     *field     : target field name
//               yt_array *data_array: write field data inside this yt_array correspondingly.
//-------------------------------------------------------------------------------------------------------
void level_derived_func_with_name(int list_len, long *gid_list, char *field, yt_array *data_array) {
    // loop over gid_list, and fill in grid data inside data_array.
    for (int lid = 0; lid < list_len; lid++) {
        // Get grid level by gid.
        int level;
        getLevelByGID(gid_list[lid], &level);

        // Get grid dimension by libyt API.
        int dim[3];
        yt_getGridInfo_Dimensions(gid_list[lid], &dim);

        // Fill in data
        int index;
        for (int k = 0; k < dim[2]; k++) {
            for (int j = 0; j < dim[1]; j++) {
                for (int i = 0; i < dim[0]; i++) {
                    index = k * dim[1] * dim[0] + j * dim[0] + i;
                    // Get the point and store data.
                    ((real *) data_array[lid].data_ptr)[index] = (real) level;
                }
            }
        }
    }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  par_io_get_attr
// Description :  For particle type "io" to return their attribute.
// 
// Notes       :  1. Prototype must be void func(int, long*, char*, yt_array*).
//                2. This function will be concatenated into python C extension, so that yt can reach
//                   particle attributes when it needs them.
//                3. In this example, we will create particle with position at the center of the grid it
//                   belongs to with Level equals to the level of the grid.
//                4. Write results to yt_array *data_array in series.
// 
// Parameter   : int   list_len      : number of gid in the list gid_list.
//               long *gid_list      : a list of gid to prepare.
//               char *attribute     : get the attribute of the particle inside gid.
//               yt_array *data_array: write the requested particle data to this array correspondingly.
//-------------------------------------------------------------------------------------------------------
void par_io_get_attr(int list_len, long *gid_list, char *attribute, yt_array *data_array) {
    // loop over gid_list, and fill in particle attribute data inside data_array.
    for (int lid = 0; lid < list_len; lid++) {
        // the particle position and level can be generated via getPositionByGID and getLevelByGID in this example.
        int Level;
        real Pos[3];
        getPositionByGID(gid_list[lid], &Pos);
        getLevelByGID(gid_list[lid], &Level);

        // fill in particle data.
        // we can get the length of the array to fill in like this, though this example only has one particle in each grids.
        for (int i = 0; i < data_array[lid].data_length; i++) {
            // fill in particle data according to the attribute.
            if (strcmp(attribute, "ParPosX") == 0) {
                ((real *) data_array[lid].data_ptr)[0] = Pos[0];
            } else if (strcmp(attribute, "ParPosY") == 0) {
                ((real *) data_array[lid].data_ptr)[0] = Pos[1];
            } else if (strcmp(attribute, "ParPosZ") == 0) {
                ((real *) data_array[lid].data_ptr)[0] = Pos[2];
            } else if (strcmp(attribute, "Level") == 0) {
                ((int *) data_array[lid].data_ptr)[0] = Level;
            }
        }
    }

}

//-------------------------------------------------------------------------------------------------------
// Function    :  getPositionByGID
// Description :  Get the center position of the grid id = gid.
//-------------------------------------------------------------------------------------------------------
void getPositionByGID(long gid, real (*Pos)[3]) {
    for (long i = 0; i < num_total_grids; i++) {
        if (gridsPtr[i].id == gid) {
            for (int d = 0; d < 3; d++) {
                (*Pos)[d] = (real) 0.5 * (gridsPtr[i].left_edge[d] + gridsPtr[i].right_edge[d]);
            }
            break;
        }
    }
}

//-------------------------------------------------------------------------------------------------------
// Function    :  getLevelByGID
// Description :  Get the level of the grid id = gid.
//-------------------------------------------------------------------------------------------------------
void getLevelByGID(long gid, int *Level) {
    for (long i = 0; i < num_total_grids; i++) {
        if (gridsPtr[i].id == gid) {
            *Level = gridsPtr[i].level;
            break;
        }
    }
}
