call _conda.bat
call %conda% activate trilab-tracker
python viewer.py
call %conda% deactivate
