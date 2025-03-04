.PHONY: lib, pybind, clean, format, all

all: lib

lib:
	@if not exist build mkdir build
	@cd build && cmake ..
	@cd build && $(MAKE)

format:
	python -m black .
	clang-format -i src/*.cc src/*.cu

clean:
	if exist build rmdir /s /q build
	del /f /q python/needle/backend_ndarray/ndarray_backend*.so
