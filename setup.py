from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='diffusion_filter',
      version='0.1',
      description='Diffusion-based filter for geophysical data',
      long_description= readme(),
      author='Olivier Goux',
      packages=['diffusion_filter'],
      install_requires=['numpy', 'scipy'],
      include_package_data=True)