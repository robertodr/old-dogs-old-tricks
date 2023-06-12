add_executable(run run.cpp)

target_include_directories(run
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
  )

add_subdirectory(mcmurchie_davidson)
add_subdirectory(pade)

target_link_libraries(run
  PRIVATE
    HighFive
  )
