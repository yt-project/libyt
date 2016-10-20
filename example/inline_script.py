

print "\nThis is the YT inline analysis script\n"

import libyt

def yt_inline():
   print "\nPerforming YT inline analysis ..."

   print "\nYT-specific parameters:"
   print libyt.param_yt.items()

   print "\nCode-specific parameters:"
   print libyt.param_user.items()

   print "\nKeys in libyt.hierarchy:"
   print libyt.hierarchy.keys()

   for key in libyt.hierarchy.keys():
      print "\nkey \"%s\":"%key
      print libyt.hierarchy[ key ]


   print "\nKeys in libyt.grid_data:"
   print libyt.grid_data.keys()

   last_grid_id = libyt.param_yt['num_grids'] - 1
   print "\nKeys in libyt.grid_data[%d]:" % last_grid_id
   print libyt.grid_data[last_grid_id].keys()

   for field in libyt.grid_data[last_grid_id].keys():
      print "\nlibyt.grid_data[%d][%s]:" % ( last_grid_id, field )
      print libyt.grid_data[last_grid_id][field]

   print ""
