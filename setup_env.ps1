# PowerShell script to set up Python virtual environment

Write-Host "Creating Python virtual environment..." -ForegroundColor Green
python -m venv venv

Write-Host "`nActivating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

Write-Host "`nUpgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

Write-Host "`nInstalling requirements..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "`nSetup complete!" -ForegroundColor Green
Write-Host "Virtual environment is now active." -ForegroundColor Green
Write-Host "To activate in the future, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow
