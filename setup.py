from setuptools import setup,find_packages
from typing import List

HYPHEN_E_DOT="-e ."

def get_requirements()->List[str]:
    try:
        with open('requirements.txt','r') as file:
            requirements=file.readlines()
            requirements=[req.replace("\n","") for req in requirements]

            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)

    except Exception as e:
        raise e
    

setup(
    name="Online Payment Fraud Detection",
    version="0.0.1",
    author="Aadrish Sen",
    author_email="aadrishsen003@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)

print(get_requirements())