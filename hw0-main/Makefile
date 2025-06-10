# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
COMMON_FLAGS = -Wall -Wextra -Wshadow -Wpedantic -Wconversion -std=c++11 
CFLAGS = -O3 $(COMMON_FLAGS)
DEBUG_FLAGS = -g -O0 $(COMMON_FLAGS)
src/simple_ml_ext.so: src/simple_ml_ext.cpp
	c++ $(CFLAGS) -shared -fPIC $$(python3 -m pybind11 --includes) $^ -o $@

# src/simple_ml_ext_debug.so: src/simple_ml_ext.cpp
# 	c++ $(DEBUG_FLAGS) -shared -fPIC $$(python3 -m pybind11 --includes) $^ -o $@

test: src/simple_ml_ext.so
	python3 -m pytest -k "softmax_regression_epoch_cpp"

# debug: src/simple_ml_ext_debug.so
# 	valgrind --tool=memcheck --leak-check=full --track-origins=yes --log-file=valgrind_output.txt python3 ./test_simple_ml_debug.py
# 	nvim valgrind_output.txt

debug: src/simple_ml_ext.so
	valgrind --tool=memcheck --leak-check=full --track-origins=yes --log-file=valgrind_output.txt python3 tests/test_simple_ml.py
	nvim valgrind_output.txt
