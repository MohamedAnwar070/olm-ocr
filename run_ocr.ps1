# OlmOCR PowerShell CLI Wrapper
# Automatically sets up dependencies and launches the optimized Python pipeline.

$ErrorActionPreference = "Stop"

Write-Host "====================================================================" -ForegroundColor Magenta
Write-Host "            OlmOCR OPTIMIZED TAMIL/ENGLISH OCR PIPELINE" -ForegroundColor Magenta
Write-Host "====================================================================" -ForegroundColor Magenta
Write-Host ""

# 1. Check Python installation
$pythonCheck = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCheck) {
    Write-Error "Python was not found on your system PATH. Please install Python 3.10+ and add it to your PATH."
    Exit 1
}

# 2. Verify and install dependencies
Write-Host "[*] Checking dependencies..." -ForegroundColor Cyan
try {
    python -c "import fitz" 2>$null
    Write-Host "[+] Dependencies verified (PyMuPDF is installed)." -ForegroundColor Green
} catch {
    Write-Host "[!] Dependencies missing. Installing now via pip..." -ForegroundColor Yellow
    python -m pip install -r (Join-Path $PSScriptRoot "requirements.txt")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install dependencies. Please run 'pip install -r requirements.txt' manually."
        Exit 1
    }
    Write-Host "[+] Dependencies successfully installed!" -ForegroundColor Green
}

Write-Host ""
Write-Host "====================================================================" -ForegroundColor Gray
Write-Host " USAGE EXAMPLES:" -ForegroundColor Gray
Write-Host ' 1. Process a PDF (all pages):' -ForegroundColor Gray
Write-Host '    .\run_ocr.ps1 --pdf "samples\document.pdf"' -ForegroundColor Gray
Write-Host ' 2. Process specific pages (e.g. pages 1, 3, 5 to 7):' -ForegroundColor Gray
Write-Host '    .\run_ocr.ps1 --pdf "samples\document.pdf" --pages "1,3,5-7"' -ForegroundColor Gray
Write-Host ' 3. Process a single screenshot/image:' -ForegroundColor Gray
Write-Host '    .\run_ocr.ps1 --image "samples\page.png"' -ForegroundColor Gray
Write-Host "====================================================================" -ForegroundColor Gray
Write-Host ""

# 3. Process and Forward Arguments to Python
$scriptPath = Join-Path $PSScriptRoot "run_olm_ocr.py"

if ($args.Count -eq 0) {
    Write-Host "[WARNING] No arguments provided." -ForegroundColor Yellow
    Write-Host 'Example: .\run_ocr.ps1 --pdf "samples\document.pdf"'
} else {
    # Convert PowerShell arguments directly to string arguments for Python
    python -u $scriptPath $args
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[-] OCR process encountered an error (Code: $LASTEXITCODE)." -ForegroundColor Red
} else {
    Write-Host ""
    Write-Host "[+] OCR process finished successfully." -ForegroundColor Green
}
