cmake_minimum_required(VERSION 3.14)
project(hearing_aid_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)

find_path(SHERPA_ONNX_INCLUDE_DIR
	NAMES sherpa-onnx/c-api/c-api.h
)
find_library(SHERPA_ONNX_LIBRARY
	NAMES sherpa-onnx-c-api sherpa-onnx
)

add_executable(hearing_aid_realtime hearing_aid_realtime.cpp)

target_include_directories(hearing_aid_realtime PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
target_link_directories(hearing_aid_realtime PRIVATE ${PORTAUDIO_LIBRARY_DIRS})
target_link_libraries(hearing_aid_realtime PRIVATE ${PORTAUDIO_LIBRARIES})

target_compile_options(hearing_aid_realtime PRIVATE ${PORTAUDIO_CFLAGS_OTHER})

if (SHERPA_ONNX_INCLUDE_DIR AND SHERPA_ONNX_LIBRARY)
	target_include_directories(hearing_aid_realtime PRIVATE ${SHERPA_ONNX_INCLUDE_DIR})
	target_link_libraries(hearing_aid_realtime PRIVATE ${SHERPA_ONNX_LIBRARY})
	target_compile_definitions(hearing_aid_realtime PRIVATE HAVE_SHERPA_ONNX_C_API=1)
endif()
