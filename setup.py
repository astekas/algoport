from setuptools import setup

setup(
    name='algoport',
    version='0.1.1',
    description='Algorithmic portfolio management framework',
    url='https://github.com/astekas/algoport.git',
    author='Anton Sauchanka',
    author_email='anton.sauchanka@gmail.com',
    license='BSD 2-clause',
    packages=['Algoport'],
    include_package_data=True,
    install_requires=['matplotlib==3.6.1',
                      'mealpy==2.5.1',
                      'numpy==1.23.3',
                      'pandas==1.4.4',
                      'plotly==5.9.0',
                      'pymoo==0.6.0',
                      'qpsolvers',
                      'osqp',
                      'rpy2==3.5.5',
                      'scipy==1.9.3',
                      'statsmodels==0.13.2',
                      'Cython==0.29.32',
                      'pyarrow==10.0.1'
                      ],
)
