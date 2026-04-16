# ─── Install Tesseract OCR for Windows ───────────────────
# Run: powershell -ExecutionPolicy Bypass -File scripts/install_tesseract.ps1

Write-Host "Installing Tesseract OCR..." -ForegroundColor Cyan

# Try winget first
try {
    winget install UB-Mannheim.TesseractOCR --accept-package-agreements --accept-source-agreements
    Write-Host "Tesseract installed via winget" -ForegroundColor Green
} catch {
    Write-Host "winget failed. Please install manually from:" -ForegroundColor Yellow
    Write-Host "https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
}

# Verify
$tesseractPath = "C:\Program Files\Tesseract-OCR\tesseract.exe"
if (Test-Path $tesseractPath) {
    Write-Host "Tesseract found at: $tesseractPath" -ForegroundColor Green
    & $tesseractPath --version
} else {
    Write-Host "Tesseract binary not found at expected path." -ForegroundColor Red
    Write-Host "Install from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
}
