@echo off
rem =====================================================================
rem  AgentEval - double-click demo launcher
rem
rem  Evaluates two versions of a demo agent over the shipped task suite,
rem  scores them, compares them, then opens the trace page in your browser.
rem  Needs no API key, no install and no network. Python 3.11 or newer only.
rem
rem  Run it with the argument  --no-pause  for automated use; it then skips
rem  the browser and the final keypress.
rem =====================================================================
setlocal
cd /d "%~dp0"

if not exist runs mkdir runs
if not exist site mkdir site

set "VLOG=site\demo-validate.txt"
set "LOG1=site\demo-v1-log.txt"
set "LOG2=site\demo-v2-log.txt"
set "RUN1=runs\demo-v1.json"
set "RUN2=runs\demo-v2.json"
set "TRACE=site\trace-demo.html"
set "DIFF=site\demo-compare.txt"
set "INTERACTIVE=1"
if "%~1"=="--no-pause" set "INTERACTIVE="

echo ======================================================================
echo   AgentEval demo - did the new agent version get better, or worse?
echo ======================================================================
echo.
echo   You will get three things:
echo     1. a score for the old version and the new version
echo     2. a pass / fail verdict for the change, like a CI build
echo     3. a browser page showing every tool call the agents made
echo.

rem ---- find Python ----------------------------------------------------
set "PYCMD="
where py >nul 2>nul
if not errorlevel 1 set "PYCMD=py -3"
if defined PYCMD goto have_python
where python >nul 2>nul
if not errorlevel 1 set "PYCMD=python"
if defined PYCMD goto have_python

echo PROBLEM: Python was not found on this PC.
echo Install Python 3.11 or newer from python.org, and on the first setup
echo screen tick the box "Add python.exe to PATH". Then double-click this
echo file again.
goto finish

:have_python
rem ---- refuse politely on Python older than 3.11 -----------------
%PYCMD% -c "import sys;sys.exit(sys.version_info < (3, 11))" 2>nul
if errorlevel 1 goto old_python
echo Using %PYCMD% for everything below.
echo.

rem ---- step 1: check the task files ----------------------------------
echo [1/5] Checking the task suites ...
%PYCMD% -m agenteval.cli validate tasks > "%VLOG%" 2>&1
if errorlevel 1 goto fail
type "%VLOG%"
echo.

rem ---- step 2 and 3: run both versions -------------------------------
echo [2/5] Running the OLD version over 10 tasks, 5 attempts each ...
%PYCMD% -m agenteval.cli run --tasks tasks\basic.json --agent mock:v1 --trials 5 --label "demo old" --out "%RUN1%" > "%LOG1%" 2>&1
if errorlevel 1 goto fail
echo   result for the old version:
findstr /b /c:"success rate" /c:"answer acc" /c:"tool accuracy" /c:"avg steps" "%LOG1%"
echo.

echo [3/5] Running the NEW version, same tasks, same attempts ...
%PYCMD% -m agenteval.cli run --tasks tasks\basic.json --agent mock:v2 --trials 5 --label "demo new" --out "%RUN2%" > "%LOG2%" 2>&1
if errorlevel 1 goto fail
echo   result for the new version:
findstr /b /c:"success rate" /c:"answer acc" /c:"tool accuracy" /c:"avg steps" "%LOG2%"
echo.

rem ---- step 4: the regression gate ----------------------------------
echo [4/5] Comparing them - this is the check that would fail a CI build ...
%PYCMD% -m agenteval.cli compare "%RUN1%" "%RUN2%" --fail-on-regression > "%DIFF%" 2>&1
set "GATE=%ERRORLEVEL%"
findstr /b /c:"success_rate" /c:"fixed" /c:"regressed" /c:"  +" /c:"  -" /c:"gating" /c:"FAIL" "%DIFF%"
echo.
if "%GATE%"=="0" echo   Verdict: the change is safe, nothing regressed - the build stays green.
if not "%GATE%"=="0" echo   Verdict: the gate rejected the build on purpose. The new version is
if not "%GATE%"=="0" echo   better on 9 tasks but broke 1 task, and AgentEval refuses to let a
if not "%GATE%"=="0" echo   single broken task hide behind a better average. That is the point
if not "%GATE%"=="0" echo   of the tool. See site\demo-compare.txt for the full diff.
echo.

rem ---- step 5: the trace page ---------------------------------------
echo [5/5] Building the trace page for the new version ...
%PYCMD% -m agenteval.cli viewer "%RUN2%" --out "%TRACE%" >nul 2>&1
if errorlevel 1 goto fail
echo   written: %TRACE%
echo   In the page: tick "failures only", or pick a failure mode, to see
echo   exactly which tool call went wrong and what the agent did with it.
echo.
if defined INTERACTIVE start "" "%~dp0%TRACE%"

echo ======================================================================
echo  Files this produced
echo ======================================================================
echo   %RUN1% - old version, every attempt scored
echo   %RUN2% - new version, every attempt scored
echo   %TRACE% - the visual trace, opens in your browser
echo   %DIFF% - the full diff and the pass / fail verdict
echo.
echo  Want to test your own agent instead of the two demo versions?
echo  CONTRIBUTING.md shows the three fields you need to implement, and
echo  README section 8 shows the command for a real model with an API key.
goto finish

:old_python
echo PROBLEM: this needs Python 3.11 or newer and yours is older.
echo Install a newer Python from python.org, tick "Add python.exe to PATH"
echo on the first setup screen, then double-click this file again.
goto finish

:fail
echo.
echo PROBLEM: a step above did not run. The messages just above this line
echo say which one. Nothing on your PC was changed except the demo output
echo files. A common cause is running this file from a folder that is not
echo the AgentEval project - double-click the copy that sits next to the
echo "agenteval" folder.
echo.

:finish
if defined INTERACTIVE (
  echo Press any key to close this window.
  pause >nul
)
endlocal
exit /b 0
