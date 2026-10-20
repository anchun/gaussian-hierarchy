cmake_minimum_required (VERSION 3.0)
project (GaussianHierarchy LANGUAGES CUDA CXX)

add_library (GaussianHierarchy
 FlatGenerator.h
 FlatGenerator.cpp
 PointbasedKdTreeGenerator.h
 PointbasedKdTreeGenerator.cpp
 ClusterMerger.h
 ClusterMerger.cpp
 appearance_filter.h
 appearance_filter.cpp
 AvgMerger.h
 AvgMerger.cpp
 writer.h
 writer.cpp
 common.h 
 loader.h 
 loader.cpp
 hierarchy_loader.h 
 hierarchy_loader.cpp
 hierarchy_explicit_loader.h
 hierarchy_explicit_loader.cpp
 hierarchy_writer.h
 hierarchy_writer.cpp
 traversal.h
 traversal.cpp
 runtime_maintenance.h
 runtime_maintenance.cu
 runtime_switching.h
 runtime_switching.cu
 rotation_aligner.h
 rotation_aligner.cpp
 merger_policy.h
 merger_policy.cpp
 spatial_metadata.cpp
 half.hpp
 types.h)
target_include_directories(GaussianHierarchy PRIVATE dependencies/eigen)

set_property(TARGET GaussianHierarchy PROPERTY CXX_STANDARD 17)
set_target_properties(GaussianHierarchy PROPERTIES CUDA_ARCHITECTURES "75;86;90;120")
set(CMAKE_CUDA_STANDARD 17)

target_include_directories(GaussianHierarchy
                           PUBLIC
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                            $<INSTALL_INTERFACE:include>
                           )
						   
install(TARGETS
        GaussianHierarchy
    EXPORT GaussianHierarchyTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES runtime_maintenance.h runtime_switching.h hierarchy_loader.h types.h DESTINATION include)
install(EXPORT GaussianHierarchyTargets
  FILE GaussianHierarchyConfig.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

add_executable (GaussianHierarchyCreator
 mainHierarchyCreator.cpp
)

add_executable (GaussianHierarchyMerger
 mainHierarchyMerger.cpp
)

add_executable (GaussianHierarchyPlyGenerator
 mainHierarchyPlyGenerator.cpp
)

add_executable (GaussianPlyConverter
 mainPlyConverter.cpp
)

add_executable (GaussianPlyLODGenerator
mainPlyLODGenerator.cpp
)

target_include_directories(GaussianHierarchyCreator PRIVATE dependencies/eigen)
set_property(TARGET GaussianHierarchyCreator PROPERTY CXX_STANDARD 17)
target_link_libraries(GaussianHierarchyCreator PUBLIC GaussianHierarchy)

target_include_directories(GaussianHierarchyMerger PRIVATE dependencies/eigen)
set_property(TARGET GaussianHierarchyMerger PROPERTY CXX_STANDARD 17)
target_link_libraries(GaussianHierarchyMerger PUBLIC GaussianHierarchy)

target_include_directories(GaussianHierarchyPlyGenerator PRIVATE dependencies/eigen)
set_property(TARGET GaussianHierarchyPlyGenerator PROPERTY CXX_STANDARD 17)
target_link_libraries(GaussianHierarchyPlyGenerator PUBLIC GaussianHierarchy)

target_include_directories(GaussianPlyConverter PRIVATE dependencies/eigen)
set_property(TARGET GaussianPlyConverter PROPERTY CXX_STANDARD 17)
target_link_libraries(GaussianPlyConverter PUBLIC GaussianHierarchy)

target_include_directories(GaussianPlyLODGenerator PRIVATE dependencies/eigen)
set_property(TARGET GaussianPlyLODGenerator PROPERTY CXX_STANDARD 17)
target_link_libraries(GaussianPlyLODGenerator PUBLIC GaussianHierarchy)

include(CTest)
if(BUILD_TESTING)
 add_executable(test_merger_policy tests/test_merger_policy.cpp)
 target_include_directories(test_merger_policy PRIVATE dependencies/eigen)
 set_property(TARGET test_merger_policy PROPERTY CXX_STANDARD 17)
 target_link_libraries(test_merger_policy PRIVATE GaussianHierarchy)
 add_test(NAME merger_policy COMMAND test_merger_policy)
 foreach(spatial_test spatial_policy spatial_hierarchy)
  add_executable(test_${spatial_test} tests/test_${spatial_test}.cpp)
  target_include_directories(test_${spatial_test} PRIVATE dependencies/eigen)
  set_property(TARGET test_${spatial_test} PROPERTY CXX_STANDARD 17)
  target_link_libraries(test_${spatial_test} PRIVATE GaussianHierarchy)
  target_compile_definitions(test_${spatial_test} PRIVATE
   SPATIAL_FIXTURE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/spatial_partition.json")
  add_test(NAME ${spatial_test} COMMAND test_${spatial_test})
 endforeach()
 target_compile_definitions(test_spatial_hierarchy PRIVATE
  SPATIAL_MERGER_PATH="$<TARGET_FILE:GaussianHierarchyMerger>")
 add_dependencies(test_spatial_hierarchy GaussianHierarchyMerger)
endif()
