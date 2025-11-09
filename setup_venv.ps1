# Setup script for creating virtual environment and installing dependencies (Windows PowerShell)

Write-Host "Setting up virtual environment..." -ForegroundColor Green

# Create virtual environment
python -m venv .venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
& .venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Green
python -m pip install --upgrade pip

# Install dependencies
Write-Host "Installing dependencies..." -ForegroundColor Green
pip install -r requirements.txt

# Install in development mode (optional)
Write-Host "Installing package in development mode..." -ForegroundColor Green
pip install -e .

Write-Host ""
Write-Host "✓ Virtual environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the virtual environment, run:" -ForegroundColor Yellow
Write-Host "  .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "To deactivate, run:" -ForegroundColor Yellow
Write-Host "  deactivate" -ForegroundColor Cyan

