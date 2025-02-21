from setuptools import setup, Extension
import pybind11
import sysconfig

# 你可以选择指定 Python 版本
python_include = sysconfig.get_paths()['include']
python_lib = sysconfig.get_paths()['stdlib'] + '/libs'

# 创建 pybind11 模块
module = Extension(
    'simple_ml_ext',
    sources=['simple_ml_ext.cpp'],
    include_dirs=[pybind11.get_include(), python_include],
    library_dirs=[python_lib],
    libraries=['python310'],  # 这里是 Python 版本对应的库名称
)

setup(
    name='simple_ml_ext',
    version='1.0',
    ext_modules=[module],
)