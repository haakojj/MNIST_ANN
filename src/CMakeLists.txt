include_directories(
	${EIGEN_INCLUDE_DIRS}
	${PNGPP_INCLUDE_DIRS}
	../include
)

add_executable(${EXE_NAME}
	MNIST_database.cpp
	MNIST_image.cpp
	MNIST_neuralNetwork.cpp
	Utils.cpp
	test.cpp
)

target_link_libraries(${EXE_NAME}
	${PNG_LIBRARY}
)

set_target_properties(${EXE_NAME}
	PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS ${EXE_NAME} DESTINATION ${INSTALL_DIR})
install(FILES ${DATA_SETS} DESTINATION ${INSTALL_DIR})
