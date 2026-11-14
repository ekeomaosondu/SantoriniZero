"""Builds the `santorini` C++ extension (engine, df-pn solver, MCTS, agent search).

    pip install -e .              # or: python setup.py build_ext --inplace

Set SANTORINI_NATIVE=0 to build without -march=native (portable binary).
"""
import os

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

flags = ["-O3", "-pthread"]
if os.environ.get("SANTORINI_NATIVE", "1") != "0":
    flags.append("-march=native")

setup(
    name="santorini-zero",
    version="0.1.0",
    description="AlphaZero for the Santorini board game: C++ engine, exact solver and self-play MCTS",
    ext_modules=[
        Pybind11Extension(
            "santorini",
            ["cpp/santorini_py.cpp"],
            include_dirs=["cpp"],
            cxx_std=17,
            extra_compile_args=flags,
            extra_link_args=["-pthread"],
        )
    ],
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.10",
    zip_safe=False,
)
