add_executable(labelpropagation-cpu LabelPropagation.cpp)
add_dependencies(apps labelpropagation-cpu)
target_link_libraries(labelpropagation-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS labelpropagation-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)

add_test_scale(small-sync labelpropagation-cpu "${BASEINPUT}/reference/structured/rome99.gr" -delta 8)
add_test_scale(small-async labelpropagation-cpu "${BASEINPUT}/scalefree/rmat10.gr" -delta 8)

add_executable(labelpropagation-openmp-cpu LabelPropagation-openmp.cpp)
add_dependencies(apps labelpropagation-openmp-cpu)
target_link_libraries(labelpropagation-openmp-cpu PRIVATE Galois::shmem lonestar)
install(TARGETS labelpropagation-openmp-cpu DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT apps EXCLUDE_FROM_ALL)