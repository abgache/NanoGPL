@echo off
setlocal enabledelayedexpansion

set "CONFIG_FILE=config.json"

for /f "usebackq tokens=*" %%A in ("%CONFIG_FILE%") do (
    set "line=%%A"
    echo !line! | findstr /i ""path"" >nul
    if !errorlevel! == 0 (
        for /f "tokens=2 delims=:" %%B in ("!line!") do (
            set "filepath=%%B"
            set "filepath=!filepath: =!"
            set "filepath=!filepath:"=!"
            if exist "!filepath!" (
                echo Deleting !filepath!
                del /f /q "!filepath!"
            )
        )
    )
)

echo Done.