

print "\nThis is the YT inline analysis script\n"

import libyt

def yt_inline():
   print "\nPerforming YT inline analysis ..."

   print ""
   #print libyt.__dict__
   print libyt.param_yt.items()
   print ""
   print libyt.param_user.items()
   print ""
   print libyt.hierarchy.keys()
   print ""

   for key in libyt.hierarchy.keys():
      print "key \"%s\""%key
      print libyt.hierarchy[ key ]
      print ""
