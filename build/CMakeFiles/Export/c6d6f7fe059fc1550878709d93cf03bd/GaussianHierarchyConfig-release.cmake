#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GaussianHierarchy" for configuration "Release"
set_property(TARGET GaussianHierarchy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(GaussianHierarchy PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libGaussianHierarchy.a"
  )

list(APPEND _cmake_import_check_targets GaussianHierarchy )
list(APPEND _cmake_import_check_files_for_GaussianHierarchy "${_IMPORT_PREFIX}/lib/libGaussianHierarchy.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
