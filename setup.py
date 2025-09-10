import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pso-library",
    version="0.0.1",
    author="William M. Boler",
    author_email="wboler05@gmail.com",
    description="A modular and extensible Particle Swarm Optimization (PSO) library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/pso-library",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pyyaml",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.10',
)
