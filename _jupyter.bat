call _conda.bat
call %conda% activate trilab-tracker
jupyter-notebook
call %conda% deactivate
