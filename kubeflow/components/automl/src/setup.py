import setuptools

REQUIRED_PACKAGES = [
  'tensorflow_transform',
  'apache-beam',
  'tensorflow',
  'Pillow'
]


setuptools.setup(
  name='make-dataset',
  version='v0.0.1',
  include_package_data=True,
  install_requires=REQUIRED_PACKAGES,
  packages=setuptools.find_packages(),
)
