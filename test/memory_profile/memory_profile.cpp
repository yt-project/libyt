#include "libyt.h"

#ifndef SERIAL_MODE
#include <mpi.h>
#endif

#include <valgrind/valgrind.h>

#include <cstdio>
#include <iostream>
#include <vector>

#define CALL_VALGRIND(name_tag, my_rank, time_tag)                                                                     \
    {                                                                                                                  \
        char valgrind[100];                                                                                            \
        snprintf(valgrind, 100, "detailed_snapshot %s_rank%d_time%d.mem_prof\0", name_tag, my_rank, t);                \
        VALGRIND_MONITOR_COMMAND(valgrind);                                                                            \
    }

void DerivedFunc(const int len, const long* list_gid, const char* field_name, yt_array* data_array) {
    for (int l = 0; l < len; l++) {
        for (int idx = 0; idx < (8 * 8 * 8); idx++) {
            ((float*)data_array[l].data_ptr)[idx] = 1.0;
        }
    }
}

// void GetParticleAttr(const int list_len, const long *gid_list, const char *par_type, const char *attribute, yt_array
// *data_array) {
//     for (int lid = 0; lid < list_len; lid++) {
//         for (int i = 0; i < data_array[lid].data_length; i++) {
//             ((int*) data_array[lid].data_ptr)[i] = 100;
//         }
//     }
// }

int main(int argc, char* argv[]) {
    /* initialize testing environment. */
    int my_rank = 0;
    int my_size = 1;
#ifndef SERIAL_MODE
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &my_size);
#endif

    if (argc != 2) {
        printf("Usage: %s <python_function_name>\n", argv[0]);
        return EXIT_FAILURE;
    }

    /* Parameters for testing */
    long iter = 5;
    int grid_size = 8;
    int num_grids = 20000;  // cannot alter arbitrary, domain dim is based on this.
    int num_grids_local = num_grids / my_size;
    if (my_rank == my_size - 1) {
        num_grids_local = num_grids - num_grids_local * (my_size - 1);
    }

    /* Generate data set for testing */
    yt_param_libyt param_libyt;
    param_libyt.verbose = YT_VERBOSE_INFO;
    param_libyt.script = "test_memory_profile";
    param_libyt.check_data = false;
    if (yt_initialize(argc, argv, &param_libyt) != YT_SUCCESS) {
        printf("yt_initialize failed!\n");
    }

    yt_param_yt param_yt;
    param_yt.frontend = "gamer";
    param_yt.fig_basename = "FigName";
    param_yt.length_unit = 3.0857e21;
    param_yt.mass_unit = 1.9885e33;
    param_yt.time_unit = 3.1557e13;
    param_yt.velocity_unit = param_yt.length_unit / param_yt.time_unit;

    param_yt.current_time = 0.0;
    param_yt.dimensionality = 3;
    param_yt.refine_by = 2;
    param_yt.num_grids = num_grids;
    param_yt.domain_dimensions[0] = 100 * grid_size;
    param_yt.domain_dimensions[1] = 20 * grid_size;
    param_yt.domain_dimensions[2] = 10 * grid_size;

    double space[3] = {100.0 / param_yt.domain_dimensions[0], 100.0 / param_yt.domain_dimensions[1],
                       100.0 / param_yt.domain_dimensions[2]};

    for (int d = 0; d < 3; d++) {
        param_yt.domain_left_edge[d] = 0.0;
        param_yt.domain_right_edge[d] = 100.0;
        param_yt.periodicity[d] = 0;
    }

    param_yt.cosmological_simulation = 0;
    param_yt.current_redshift = 0.5;
    param_yt.omega_lambda = 0.7;
    param_yt.omega_matter = 0.3;
    param_yt.hubble_constant = 0.7;
    param_yt.num_grids_local = num_grids_local;

    // for fields and particles
    param_yt.num_fields = 2;  // cell-centered and derived
    // param_yt.num_par_types           = 1;         // io

    // yt_par_type par_type_list[1];
    // par_type_list[0].par_type = "io";
    // par_type_list[0].num_attr = 4;
    // param_yt.par_type_list = par_type_list;

    // generating field (twos field in cell-centered) and particle data (position x/y/z)
    // for testing data wrapping in libyt
    std::vector<double*> field_data;  //, particle_data;

    for (int i = 0; i < num_grids_local; i++) {
        double* temp = new double[grid_size * grid_size * grid_size];
        for (int j = 0; j < grid_size * grid_size * grid_size; j++) {
            temp[j] = 2.0;
        }
        field_data.push_back(temp);
    }

    // if (myrank == 0) {

    //     double shift_x = space[0] / 2;
    //     double shift_y = space[1] / 2;
    //     double shift_z = space[2] / 2;

    //     for (int v = 0; v < 3; v++) {
    //         double *pos_x = new double [grid_size * grid_size * grid_size];
    //         double *pos_y = new double [grid_size * grid_size * grid_size];
    //         double *pos_z = new double [grid_size * grid_size * grid_size];

    //         for (int idz = 0; idz < grid_size; idz++) {
    //             for (int idy = 0; idy < grid_size; idy++) {
    //                 for (int idx = 0; idx < grid_size; idx++) {
    //                     int index = idx + grid_size * idy + grid_size * grid_size * idz;
    //                     pos_x[index] = shift_x + idx * space[0];
    //                     pos_y[index] = shift_y + idy * space[1];
    //                     pos_z[index] = shift_z + idz * space[2];
    //                 }
    //             }
    //         }

    //         particle_data.push_back(pos_x);
    //         particle_data.push_back(pos_y);
    //         particle_data.push_back(pos_z);
    //     }
    // }

    // iteration starts
    for (int t = 0; t < iter; t++) {
        /* libyt API yt_set_Parameters */
        yt_set_Parameters(&param_yt);

        /* libyt API yt_set_UserParameter */
        const int mhd = 0;
        yt_set_UserParameterInt("mhd", 1, &mhd);
        const int srhd = 0;
        yt_set_UserParameterInt("srhd", 1, &srhd);

        /* libyt API yt_get_FieldsPtr */
        yt_field* field_list;
        yt_get_FieldsPtr(&field_list);

        // set derived field
        field_list[0].field_name = "DerivedOnes";
        field_list[0].field_type = "derived_func";
        field_list[0].contiguous_in_x = true;
        field_list[0].field_dtype = YT_FLOAT;
        const char* field_name_alias[] = {"Name Alias 1", "Name Alias 2", "Name Alias 3"};
        field_list[0].field_name_alias = field_name_alias;
        field_list[0].num_field_name_alias = 3;
        field_list[0].derived_func = DerivedFunc;

        // set cell-centered field
        field_list[1].field_name = "CCTwos";
        field_list[1].field_type = "cell-centered";
        field_list[1].contiguous_in_x = true;
        field_list[1].field_dtype = YT_DOUBLE;

        // /* libyt API yt_get_ParticlesPtr */
        // yt_particle *particle_list;
        // yt_get_ParticlesPtr( &particle_list );

        // const char *attr_name[] = {"ParPosX", "ParPosY", "ParPosZ", "Level"};
        // const char *attr_name_alias[] = {"grid_level"};
        // for (int v=0; v < 4; v++) {
        //     particle_list[0].attr_list[v].attr_name = attr_name[v];
        //     if (v == 3) {
        //         particle_list[0].attr_list[v].attr_dtype = YT_INT;
        //         particle_list[0].attr_list[v].num_attr_name_alias = 1;
        //         particle_list[0].attr_list[v].attr_name_alias     = attr_name_alias;
        //         particle_list[0].attr_list[v].attr_display_name   = "Level of the Grid";
        //     }
        //     else {
        //         particle_list[0].attr_list[v].attr_dtype = YT_DOUBLE;
        //     }
        // }
        // particle_list[0].coor_x = attr_name[0];
        // particle_list[0].coor_y = attr_name[1];
        // particle_list[0].coor_z = attr_name[2];
        // particle_list[0].get_par_attr = GetParticleAttr;

        yt_grid* grids_local;
        yt_get_GridsPtr(&grids_local);

        for (long lid = 0; lid < num_grids_local; lid++) {
            // general info
            long gid = lid + my_rank * num_grids_local;
            int grid_idx[3];
            grid_idx[2] = gid / (param_yt.domain_dimensions[0] * param_yt.domain_dimensions[1]);
            grid_idx[1] = (gid - (grid_idx[2] * param_yt.domain_dimensions[0] * param_yt.domain_dimensions[1])) /
                          param_yt.domain_dimensions[0];
            grid_idx[0] = gid - (grid_idx[2] * param_yt.domain_dimensions[0] * param_yt.domain_dimensions[1]) -
                          (grid_idx[1] * param_yt.domain_dimensions[0]);
            for (int d = 0; d < 3; d++) {
                grids_local[lid].left_edge[d] = space[d] * (double)grid_idx[d];
                grids_local[lid].right_edge[d] = grids_local[lid].left_edge[d] + space[d];
                grids_local[lid].grid_dimensions[d] = grid_size;
            }
            grids_local[lid].id = gid;
            grids_local[lid].parent_id = -1;
            grids_local[lid].level = 0;

            // append cell-centered field data
            grids_local[lid].field_data[0].data_ptr = field_data[lid];
            grids_local[lid].field_data[1].data_ptr = field_data[lid];
            for (int d = 0; d < 3; d++) {
                grids_local[lid].field_data[0].data_dimensions[d] = grids_local[lid].grid_dimensions[d];
            }

            // // append particle data on gid = 0 only
            // if (gid == 0) {
            //     grids_local[lid].par_count_list[0] = grid_size * grid_size * grid_size;
            //     grids_local[lid].particle_data[0][0].data_ptr = particle_data[0];
            //     grids_local[lid].particle_data[0][1].data_ptr = particle_data[1];
            //     grids_local[lid].particle_data[0][2].data_ptr = particle_data[2];
            // }
        }

        /* libyt API commit, call Python function, and free*/

        yt_commit();

        yt_run_Function(argv[1]);

        CALL_VALGRIND("BeforeFree", my_rank, t);

        yt_free();

        CALL_VALGRIND("AfterFree", my_rank, t);

        std::cout << "time step = " << t << " ... done" << std::endl;
    }

    // free resources
    for (int i = 0; i < num_grids_local; i++) {
        delete[] field_data[i];
    }

    /* libyt API yt_finalize */
    if (yt_finalize() != YT_SUCCESS) {
        printf("yt_finalize failed!\n");
    }

#ifndef SERIAL_MODE
    MPI_Finalize();
#endif

    return 0;
}
