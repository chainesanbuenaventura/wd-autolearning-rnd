import setuptools

setuptools.setup(
  name='MakeDatasetTemplate',
  version='v0.0.1',
  install_requires=[
    'tensorflow_transform',
    'apache-beam',
    'tensorflow',
    'Pillow'
  ],
  packages=setuptools.find_packages(),
)
