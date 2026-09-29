@echo off
REM OlmOCR Windows CLI Wrapper
REM Automatically sets up dependencies and launches the optimized Python pipeline.

echo ====================================================================
echo             OlmOCR OPTIMIZED TAMIL/ENGLISH OCR PIPELINE
echo ====================================================================
echo.

REM 1. Check Python installation
where python >nul 2>&1
if %errorlevel% equ 0 goto python_ok
echo [ERROR] Python was not found on your system PATH.
echo Please install Python 3.10+ and make sure to check "Add Python to PATH" during installation.
pause
exit /b 1

:python_ok

REM 2. Install dependencies automatically if missing
echo [*] Checking dependencies...
python -c "import fitz" >nul 2>&1
if %errorlevel% equ 0 goto deps_ok

echo [!] PyMuPDF is not installed. Installing now...
python -m pip install -r "%~dp0requirements.txt"
if %errorlevel% equ 0 goto deps_installed
echo [ERROR] Failed to install dependencies. Please run 'pip install -r requirements.txt' manually.
pause
exit /b 1

:deps_installed
echo [+] Dependencies successfully installed!
goto deps_done

:deps_ok
echo [+] Dependencies verified (PyMuPDF is installed).

:deps_done

echo.
echo ====================================================================
echo  USAGE EXAMPLES:
echo  1. Process a PDF (all pages):
echo     run_ocr.bat --pdf "samples\document.pdf"
echo.
echo  2. Process specific pages (e.g. page 1, 3, 5 to 7):
echo     run_ocr.bat --pdf "samples\document.pdf" --pages "1,3,5-7"
echo.
echo  3. Process a single screenshot/image:
echo     run_ocr.bat --image "samples\page.png"
echo ====================================================================
echo.

REM 3. Forward arguments to the Python pipeline
if "%~1" == "" goto no_args

python -u "%~dp0run_olm_ocr.py" %*
goto process_ended

:no_args
echo [WARNING] No arguments provided.
echo Please run this batch file with arguments (e.g. run_ocr.bat --pdf "samples\document.pdf").
echo.
goto end

:process_ended
if %errorlevel% equ 0 goto success
echo.
echo [ERROR] The OCR process encountered an error (Code: %errorlevel%).
goto end

:success
echo.
echo [+] OCR process finished successfully.

:end
pause
