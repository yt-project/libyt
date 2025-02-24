/* [Compilation]
 *     g++ -o data_gen.out data_gen.cpp
 * [Description]
 *     Generate density data file from step 1 to 10. It creates
 *     data dir to store dumped data file.
 */

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

// grid information
#define NGRID_1D   5  // number of root grids along each direction
#define GRID_DIM   8  // grid dimension (this example assumes cubic grids)
#define REFINE_BY  2  // refinement factor between two AMR levels
#define GHOST_CELL 0  // this must be 0, since we are only generating data here

// convenient macros
#define SQR(a)  ((a) * (a))
#define CUBE(a) ((a) * (a) * (a))

double set_density(const double x, const double y, const double z, const double t,
                   const double v);

enum yt_dtype : int { YT_FLOAT = 0, YT_DOUBLE, YT_INT, YT_LONG, YT_DTYPE_UNKNOWN };

struct yt_data {
  void* data_ptr;
  int data_dimensions[3];
  yt_dtype data_dtype;
};

struct yt_grid {
  double left_edge[3];
  double right_edge[3];
  long* particle_count_list;
  long grid_particle_count;
  long id;
  long parent_id;
  int grid_dimensions[3];
  int level;
  int proc_num;
  yt_data* field_data;
};

//----------------------------------------------------------------------------------------
// Function    :  main
// Description :  Main function
//----------------------------------------------------------------------------------------
int main(int argc, char* argv[]) {
  // create data dir if it does not exit, cause dumped data will store here.
  DIR* dir = opendir("data");
  if (dir) {
    closedir(dir);
  } else if (ENOENT == errno) {
    printf("creating /data dir...\n");
    mkdir("data", 0775);
  } else {
    printf("no /data dir and cannot create one, something else is wrong...\n");
    exit(EXIT_FAILURE);
  }

  // **********************************************
  // following mimic the simulation evolution loop
  // **********************************************
  const int total_steps = 11;   // total number of evolution steps
  const double velocity = 1.0;  // velocity for setting the density field
  const double dt = 0.05;       // evolution time-step
  const double box_size = 1.0;  // simulation box size
  const double dh0 = box_size / (NGRID_1D * GRID_DIM);     // cell size at level 0
  const double dh1 = dh0 / REFINE_BY;                      // cell size at level 1
  const int num_fields = 1;                                // number of fields
  const int num_grids = CUBE(NGRID_1D) + CUBE(REFINE_BY);  // number of grids
  double time = 0.0;
  double(*field_data)[num_fields][CUBE(GRID_DIM + GHOST_CELL * 2)] =
      new double[num_grids][num_fields][CUBE(GRID_DIM + GHOST_CELL * 2)];

  for (int step = 0; step < total_steps; step++) {
    yt_grid* sim_grids = new yt_grid[num_grids];

    //    set level-0 grids
    int grid_order[3];
    for (grid_order[2] = 0; grid_order[2] < NGRID_1D; grid_order[2]++)
      for (grid_order[1] = 0; grid_order[1] < NGRID_1D; grid_order[1]++)
        for (grid_order[0] = 0; grid_order[0] < NGRID_1D; grid_order[0]++) {
          const int gid =
              (grid_order[2] * NGRID_1D + grid_order[1]) * NGRID_1D + grid_order[0];
          for (int d = 0; d < 3; d++) {
            sim_grids[gid].left_edge[d] = grid_order[d] * GRID_DIM * dh0;
            sim_grids[gid].right_edge[d] = sim_grids[gid].left_edge[d] + GRID_DIM * dh0;
            sim_grids[gid].grid_dimensions[d] = GRID_DIM;
          }
          sim_grids[gid].id = gid;
          sim_grids[gid].parent_id = -1;
          sim_grids[gid].level = 0;

          for (int k = GHOST_CELL; k < GRID_DIM + GHOST_CELL; k++)
            for (int j = GHOST_CELL; j < GRID_DIM + GHOST_CELL; j++)
              for (int i = GHOST_CELL; i < GRID_DIM + GHOST_CELL; i++) {
                const double x =
                    sim_grids[gid].left_edge[0] + ((i - GHOST_CELL) + 0.5) * dh0;
                const double y =
                    sim_grids[gid].left_edge[1] + ((j - GHOST_CELL) + 0.5) * dh0;
                const double z =
                    sim_grids[gid].left_edge[2] + ((k - GHOST_CELL) + 0.5) * dh0;

                for (int v = 0; v < num_fields; v++) {
                  field_data[gid][v][(k * (GRID_DIM + GHOST_CELL * 2) + j) *
                                         (GRID_DIM + GHOST_CELL * 2) +
                                     i] = set_density(x, y, z, time, velocity);
                }
              }
        }  // for grid_order[0/1/2]

    // in this example we refine the root grid with the peak density into REFINE_BY^3
    // subgrids
    const double peak[3] = {0.5 * box_size + velocity * time,
                            0.5 * box_size + velocity * time,
                            0.5 * box_size};
    const double grid_width = GRID_DIM * dh0;
    const int center_idx[3] = {
        int(peak[0] / grid_width), int(peak[1] / grid_width), int(peak[2] / grid_width)};
    const int gid_refine =
        (center_idx[2] * NGRID_1D + center_idx[1]) * NGRID_1D + center_idx[0];
    const int gid_offset = CUBE(NGRID_1D);

    for (grid_order[2] = 0; grid_order[2] < REFINE_BY; grid_order[2]++)
      for (grid_order[1] = 0; grid_order[1] < REFINE_BY; grid_order[1]++)
        for (grid_order[0] = 0; grid_order[0] < REFINE_BY; grid_order[0]++) {
          const int gid = (grid_order[2] * REFINE_BY + grid_order[1]) * REFINE_BY +
                          grid_order[0] + gid_offset;
          for (int d = 0; d < 3; d++) {
            sim_grids[gid].left_edge[d] =
                sim_grids[gid_refine].left_edge[d] + grid_order[d] * GRID_DIM * dh1;
            sim_grids[gid].right_edge[d] = sim_grids[gid].left_edge[d] + GRID_DIM * dh1;
            sim_grids[gid].grid_dimensions[d] = GRID_DIM;
          }
          sim_grids[gid].id = gid;
          sim_grids[gid].parent_id = gid_refine;
          sim_grids[gid].level = 1;

          for (int k = GHOST_CELL; k < GRID_DIM + GHOST_CELL; k++)
            for (int j = GHOST_CELL; j < GRID_DIM + GHOST_CELL; j++)
              for (int i = GHOST_CELL; i < GRID_DIM + GHOST_CELL; i++) {
                const double x =
                    sim_grids[gid].left_edge[0] + ((i - GHOST_CELL) + 0.5) * dh1;
                const double y =
                    sim_grids[gid].left_edge[1] + ((j - GHOST_CELL) + 0.5) * dh1;
                const double z =
                    sim_grids[gid].left_edge[2] + ((k - GHOST_CELL) + 0.5) * dh1;

                for (int v = 0; v < num_fields; v++) {
                  field_data[gid][v][(k * (GRID_DIM + GHOST_CELL * 2) + j) *
                                         (GRID_DIM + GHOST_CELL * 2) +
                                     i] = set_density(x, y, z, time, velocity);
                }
              }
        }  // for grid_order[0/1/2]

    // output density data in each step
    for (int gid = 0; gid < num_grids; gid++) {
      FILE* fp;
      char filename[100];

      sprintf(filename, "./data/Dens_grid%d_step%d.txt", gid, step);
      fp = fopen(filename, "w");

      for (int k = GHOST_CELL; k < GRID_DIM + GHOST_CELL; k++)
        for (int j = GHOST_CELL; j < GRID_DIM + GHOST_CELL; j++)
          for (int i = GHOST_CELL; i < GRID_DIM + GHOST_CELL; i++) {
            fprintf(fp,
                    "%.10f\n",
                    field_data[gid][0][(k * (GRID_DIM + GHOST_CELL * 2) + j) *
                                           (GRID_DIM + GHOST_CELL * 2) +
                                       i]);
          }
      fclose(fp);
    }
    printf("Generate density data in step %d\n", step);

    //  free resources
    for (int g = 0; g < num_grids; g++) delete[] sim_grids[g].field_data;
    delete[] sim_grids;

    time += dt;
  }  // for (int step=0; step<total_steps; step++)

  return EXIT_SUCCESS;

}  // FUNCTION : main

//----------------------------------------------------------------------------------------
// Function    :  set_density
// Description :  Return density at given coordinates and time
//----------------------------------------------------------------------------------------
double set_density(const double x, const double y, const double z, const double t,
                   const double v) {
  const double center[3] = {0.5 + v * t, 0.5 + v * t, 0.5};  // drift with v along (1,1,0)
  const double sigma = 0.05;
  const double amplitude = 1.0e6;
  const double background = 1.0;

  return amplitude *
             exp(-0.5 * (SQR(x - center[0]) + SQR(y - center[1]) + SQR(z - center[2])) /
                 SQR(sigma)) +
         background;

}  // FUNCTION : set_density
