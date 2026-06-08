from setuptools import setup, find_packages

HYPEN_DASHES = '-e .'
def get_requirements(file_path):
    with open(file_path, 'r') as file:
        requirements = file.read().splitlines()
    
    if HYPEN_DASHES in requirements:
        requirements.remove(HYPEN_DASHES)

    return requirements


setup(
    name='mlproject',
    version='0.1',
    packages=find_packages(),
    author='Riaz Ahamed',
    author_email='riazlearning@gmail.com',
    description='A machine learning project',
    install_requires=get_requirements('requirements.txt')    
    )