# - Try to find Glib-2.0 (with gobject)
# Once done, this will define

#  Skeltrack_FOUND - system has Glib
#  Skeltrack_INCLUDE_DIRS - the Glib include directories
#  Skeltrack_LIBRARIES - link these to use Glib

include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(Skeltrack_PKGCONF skeltrack-0.1)

# Main include dir
find_path(Skeltrack_INCLUDE_DIR
  NAMES skeltrack.h
  PATHS ${Skeltrack_PKGCONF_INCLUDE_DIRS}
  PATH_SUFFIXES skeltrack-0.1
)

# Glib-related libraries also use a separate config header, which is in lib dir
#find_path(GlibConfig_INCLUDE_DIR
#  NAMES glibconfig.h
#  PATHS ${Glib_PKGCONF_INCLUDE_DIRS} /usr
#  PATH_SUFFIXES lib/glib-2.0/include
#)

# Finally the library itself
find_library(Skeltrack_LIBRARY
  NAMES skeltrack-0.1
  PATHS ${Skeltrack_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(Skeltrack_PROCESS_INCLUDES Skeltrack_INCLUDE_DIR)
set(Skeltrack_PROCESS_LIBS Skeltrack_LIBRARY)
libfind_process(Skeltrack)