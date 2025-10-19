# AI Assistant Website Runner
Write-Host "🚀 Starting AI Assistant Website..." -ForegroundColor Green
Write-Host "📁 Current Directory: $(Get-Location)" -ForegroundColor Yellow

# Check if virtual environment exists
if (Test-Path ".venv\Scripts\python.exe") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    
    # Run the application
    Write-Host "🌐 Starting FastAPI server on http://localhost:8002" -ForegroundColor Cyan
    Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
    Write-Host ""
    
    try {
        & ".venv\Scripts\python.exe" "simple_main.py"
    }
    catch {
        Write-Host "❌ Error starting application: $_" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Virtual environment not found. Please run setup first." -ForegroundColor Red
    Write-Host "Run: python -m venv .venv" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
Read-Host