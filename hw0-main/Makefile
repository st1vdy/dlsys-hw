# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
# 设置 C++ 编译器
CXX = c++

# 设置编译选项
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC

# 设置 Python 和 pybind11 的包含路径
PYBIND11_INCLUDES = -ID:\Softwares\anaconda\envs\pytorch\include -ID:\Softwares\anaconda\envs\pytorch\lib\site-packages\pybind11\include

# 设置 Python 库路径
PYTHON_LIBS = -LD:\Softwares\anaconda\envs\pytorch\libs -lpython310

# 设置源文件和目标文件
SRC = src/simple_ml_ext.cpp
OUT = src/simple_ml_ext.so

# 默认目标
default:
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(SRC) -o $(OUT)
#default:
#	c++ -O3 -Wall -shared -std=c++17 -fPIC $$(python3 -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so
