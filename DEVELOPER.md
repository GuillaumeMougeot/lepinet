# Dev guide

0. take note of all README in each subfolders to understand the projects and how to code within it.
1. journal everything. Use journal folder to write markdown report about your experiments. Use README files in subfolders journal and dev to register what files in the subfolder are. Use RESULTS.md to report what the trained model folders are. Always consider that you will handout your work to someone else. Take the freedom to write additional documents if needed. Structural doc (README.md, DEV.md, HANDOUT.md etc.) must stay succint while journal or sub-doc can be more verbose.
2. use uv and local venv. (currently broken when uv sync and one day this must be fixed)
3. use config files to start training.
4. use archive folder and its subfolder to store outdated files.
5. the main dataset is data/global but it is 5.6 million images, so use smaller datasets to run tests.
6. for compute, either use local GPU, if free, for tests or short training and ucloud-api for bigger training.

note: this project has a little sister for using model in a PWA: lepinet-app