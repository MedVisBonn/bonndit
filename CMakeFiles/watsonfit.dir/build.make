# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xderes/bonndit

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xderes/bonndit

# Include any dependencies generated for this target.
include CMakeFiles/watsonfit.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/watsonfit.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/watsonfit.dir/flags.make

CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.o: CMakeFiles/watsonfit.dir/flags.make
CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.o: src/bonndit/utilc/watsonfit.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xderes/bonndit/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.o -c /home/xderes/bonndit/src/bonndit/utilc/watsonfit.cpp

CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xderes/bonndit/src/bonndit/utilc/watsonfit.cpp > CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.i

CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xderes/bonndit/src/bonndit/utilc/watsonfit.cpp -o CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.s

# Object files for target watsonfit
watsonfit_OBJECTS = \
"CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.o"

# External object files for target watsonfit
watsonfit_EXTERNAL_OBJECTS =

libwatsonfit.so: CMakeFiles/watsonfit.dir/src/bonndit/utilc/watsonfit.cpp.o
libwatsonfit.so: CMakeFiles/watsonfit.dir/build.make
libwatsonfit.so: /usr/local/lib/libceres.a
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libglog.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libspqr.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libcholmod.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libmetis.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libamd.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libcamd.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libccolamd.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libcolamd.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
libwatsonfit.so: /opt/intel/oneapi/tbb/2021.5.1/lib/intel64/gcc4.8/libtbb.so.12
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libcxsparse.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libmkl_intel_lp64.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libmkl_intel_thread.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libmkl_core.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libiomp5.so
libwatsonfit.so: /usr/lib/gcc/x86_64-linux-gnu/8/libgomp.so
libwatsonfit.so: /usr/lib/x86_64-linux-gnu/libpthread.so
libwatsonfit.so: CMakeFiles/watsonfit.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xderes/bonndit/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libwatsonfit.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/watsonfit.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/watsonfit.dir/build: libwatsonfit.so

.PHONY : CMakeFiles/watsonfit.dir/build

CMakeFiles/watsonfit.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/watsonfit.dir/cmake_clean.cmake
.PHONY : CMakeFiles/watsonfit.dir/clean

CMakeFiles/watsonfit.dir/depend:
	cd /home/xderes/bonndit && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xderes/bonndit /home/xderes/bonndit /home/xderes/bonndit /home/xderes/bonndit /home/xderes/bonndit/CMakeFiles/watsonfit.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/watsonfit.dir/depend
