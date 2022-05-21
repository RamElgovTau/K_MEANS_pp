from setuptools import setup, Extension

setup(
    name='mykmeanssp',
    version='1.0',
    description='mykmeanssp Module',
    ext_modules=[
          Extension(
                # the qualified name of the extension module to build.
                'mykmeanssp',
                # the files to compile into our module relative to "setup.py".
                sources=['kmeans.c'],
          ),
    ]
)
