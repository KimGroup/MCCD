from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='mccd',
   version='0.0.1',
   description='ML decoder for quantum error correction codes',
   license="MIT",
   long_description=long_description,
   author='Yiqing Zhou',
   author_email='zhouyiqingkelly@gmail.com',
#    url="http://www.",
   packages=['mccd'],  
   install_requires=['stim', 'numpy', 'torch'], 
)