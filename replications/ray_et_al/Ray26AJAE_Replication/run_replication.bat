@echo off
setlocal

for /f "tokens=2*" %%a in (
  'reg query "HKLM\SOFTWARE\R-core\R" /v InstallPath 2^>nul'
) do set RPATH=%%b

if not defined RPATH (
  echo ERROR: R not found
  pause
  exit /b 1
)

set RSCRIPT="%RPATH%\bin\Rscript.exe"

echo ========================================
echo   AJAE Replication Workflow (R)
echo ========================================
echo.
echo IMPORTANT: Before running this file, 
echo open Ray26AJAE_Replication\scripts\0_SetPath_Packages.R
echo set project_root to the folder where the Ray26AJAE_Replication is saved, 
echo run 0_SetPath_Packages.R to install/load all required packages.
echo load packages manually if necessary
echo run run_replication.bat
echo.
pause

echo ========================================
echo Starting Replication Workflow
echo ========================================


REM --- Set working directory to batch file location (replication root) ---
cd /d "%~dp0"

set SCRIPT_DIR=scripts

REM --- Step 0: Set path & install/load required packages ---
echo Running: 0_SetPath_Packages.R ...
call %RSCRIPT% %SCRIPT_DIR%\0_SetPath_Packages.R
if errorlevel 1 (
    echo ERROR in 0_SetPath_Packages.R
    pause
    exit /b 1
)
echo 0_SetPath_Packages.R successful
echo 
echo

REM --- Step 1: Pre-stage (crosswalks + data pull) ---
echo Running: 0a_PRESTAGE.R ...
call %RSCRIPT% %SCRIPT_DIR%\0a_PRESTAGE.R
if errorlevel 1 (
    echo ERROR in 0a_PRESTAGE.R
    pause
    exit /b 1
)
echo 0a_PRESTAGE.R successful


REM --- Step 2: First stage estimation ---
echo Running: 1_FIRST_STEP.R ...
call %RSCRIPT% %SCRIPT_DIR%\1_FIRST_STEP.R
if errorlevel 1 (
    echo ERROR in 1_FIRST_STEP.R
    pause
    exit /b 1
)
echo 1_FIRST_STEP.R successful


REM --- Step 3: Second stage validation, decomposition & identification ---
echo Running: 2_SECOND_STEP.R ...
call %RSCRIPT% %SCRIPT_DIR%\2_SECOND_STEP.R
if errorlevel 1 (
    echo ERROR in 2_SECOND_STEP.R
    pause
    exit /b 1
)
echo 2_SECOND_STEP.R successful

echo.
echo =================================================
echo   All steps completed successfully!
echo   Find Figures and Table in corresponding folders
echo =================================================
pause
