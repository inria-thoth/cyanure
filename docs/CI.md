# Continuous Integration


The continuous integration is composed of 2 different parts which 
are in 2 different repositories:
- The first part is run on Github through Github Actions and target, 
the test of the development version of Cyanure and to build artifacts 
for conda and PyPi. It is also responsible for the release on PyPi. This 
CI is in the __.github__ folder of the Cyanure project.
- The second part of the CI is in the cyanure-feedstock [repository](https://github.com/conda-forge/cyanure-feedstock) also available on Github. The project is responsible for the distribution of Cyanure on conda-forge.


## Github actions

At the time of the writing there are 5 different pipelines of CI.

2 of them are dedicated to the release of the online documentation.
And the rest of them are linked to compilation and publication.

### Documentation 

One of them is defined directly by github and update the website attached to the repository automatically when the gh-pages branch is updated. This configuration is available in the settings page of the cyanure repository.
And in the settings, you have to go in the Pages repository.

The second one is defined by the file **sphinx.yml**.
This pipeline will first install Cyanure in order to get the API documentation.
After this, it will install the sphinx dependencies and build it.
On top of that, it will build the dash website which only embed a figure containing the curves for different Cyanure configurations.
When every part of the documentation is built, the __gh-pages__ branch is updated which will trigger the previous job.

### Code coverage

One of the job is dedicated to ensure that the code coverage does not decrease. It will build Cyanure with a COVERAGE flag which is a special configuration of Cyanure which will allow to generate file containing the C++ coverage. It runs coverage on the C++ and Python part of the code only based on Python test.
The code coverage threshold is distinct for both language.

The idea is not to block the jobs that's why it is seperated from the building jobs.

**coverage.yml**

### Compilation

There are 2 different worflows which handle the compilation of the library on the different platforms and  verify that each test pass on different architecture.

One of them is for development purpose and the other one is responsible for the publication on PyPi. **develop.yml**

The development pipeline will be triggered on each push. It will verify the code linting, compile and test. **main.yml**

The publication pipeline will be triggered when a push will occur on the master branch. It will verify the linting, build, test, publish on PyPi, create github release and upload artifacts.  

There is no automatic pipeline to update the conda-forge recipe. To update the cyanure recipe you need to go on the conda repository.

Go to the folder recipe, and update the file by:
- incrementing the version (if needed)
- update the hash related to the archive found on Github.
- set build number to 0 if version is incremented otherwise increment the build number

### Extra information

Beside that Github actions file there are several scripts which are linked to the countinuous integration

The first one is in the _build_script_ folder and is responsible to set up the building environment on mac. It will create a conda environment containing the system dependencies depending on the target architecture.

To handle the compilation on different platforms, we use a library called cibuildwheel which handle most of the overhead necessary to compile on 3 different platforms. The configuration of the tool is in the file called **pyproject.toml**

There is a **Makefile** in the __docs/dash__ folder. This file is responsible for the creation of the dash figure. It will run the dash application in the background and download the different element of the website. It also prepare necessary files to the format expected by github.


