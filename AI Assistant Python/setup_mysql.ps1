# MySQL Setup Script for AI Duty Officer Assistant
# This script sets up the operational database

$mysqlPath = "C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "  MySQL Database Setup for AI Duty Officer Assistant" -ForegroundColor White
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Check if MySQL is accessible
if (-not (Test-Path $mysqlPath)) {
    Write-Host "[ERROR] MySQL not found at: $mysqlPath" -ForegroundColor Red
    Write-Host "Please update the `$mysqlPath variable with your MySQL installation path" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/4] MySQL found: " -NoNewline -ForegroundColor Green
Write-Host "OK" -ForegroundColor Green
Write-Host ""

# Prompt for MySQL root password
Write-Host "Please enter your MySQL root password:" -ForegroundColor Yellow
$password = Read-Host -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($password)
$plainPassword = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

Write-Host ""
Write-Host "[2/4] Testing MySQL connection..." -ForegroundColor Cyan

# Test connection
$testConnection = "SELECT 1;"
$testResult = & $mysqlPath -u root -p"$plainPassword" -e $testConnection 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to connect to MySQL" -ForegroundColor Red
    Write-Host "Error: $testResult" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "- Wrong password" -ForegroundColor Yellow
    Write-Host "- MySQL service not running" -ForegroundColor Yellow
    Write-Host "- Root user not configured" -ForegroundColor Yellow
    exit 1
}

Write-Host "      Connection: " -NoNewline -ForegroundColor Green
Write-Host "OK" -ForegroundColor Green
Write-Host ""

# Create database
Write-Host "[3/4] Creating database 'appdb'..." -ForegroundColor Cyan
$createDb = "CREATE DATABASE IF NOT EXISTS appdb CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;"
& $mysqlPath -u root -p"$plainPassword" -e $createDb 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "      Database created: " -NoNewline -ForegroundColor Green
    Write-Host "OK" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to create database" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Import schema
Write-Host "[4/4] Importing schema and data from db.sql..." -ForegroundColor Cyan

if (-not (Test-Path "db.sql")) {
    Write-Host "[ERROR] db.sql file not found in current directory" -ForegroundColor Red
    Write-Host "Please ensure db.sql is in the same folder as this script" -ForegroundColor Yellow
    exit 1
}

& $mysqlPath -u root -p"$plainPassword" appdb -e "source db.sql" 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "      Schema imported: " -NoNewline -ForegroundColor Green
    Write-Host "OK" -ForegroundColor Green
} else {
    Write-Host "[WARN] Source command failed, trying alternative method..." -ForegroundColor Yellow
    Get-Content "db.sql" | & $mysqlPath -u root -p"$plainPassword" appdb 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "      Schema imported: " -NoNewline -ForegroundColor Green
        Write-Host "OK" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to import schema" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""

# Verify data
Write-Host "Verifying imported data..." -ForegroundColor Cyan
$verifyQuery = @"
SELECT 
    (SELECT COUNT(*) FROM vessel) as vessels,
    (SELECT COUNT(*) FROM container) as containers,
    (SELECT COUNT(*) FROM edi_message) as edi_messages,
    (SELECT COUNT(*) FROM api_event) as api_events;
"@

$result = & $mysqlPath -u root -p"$plainPassword" appdb -e $verifyQuery 2>&1

Write-Host $result -ForegroundColor White
Write-Host ""

# Update .env file
Write-Host "Updating .env file..." -ForegroundColor Cyan

$envPath = ".env"
if (Test-Path $envPath) {
    $envContent = Get-Content $envPath -Raw
    
    # Update OPS_DATABASE_URL
    $newUrl = "OPS_DATABASE_URL=mysql+pymysql://root:$plainPassword@localhost/appdb"
    
    if ($envContent -match "OPS_DATABASE_URL=.*") {
        $envContent = $envContent -replace "OPS_DATABASE_URL=.*", $newUrl
    } else {
        $envContent += "`n$newUrl"
    }
    
    Set-Content $envPath -Value $envContent -NoNewline
    Write-Host "      .env updated: " -NoNewline -ForegroundColor Green
    Write-Host "OK" -ForegroundColor Green
} else {
    Write-Host "[WARN] .env file not found" -ForegroundColor Yellow
    Write-Host "Please manually add to .env:" -ForegroundColor Yellow
    Write-Host "OPS_DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost/appdb" -ForegroundColor White
}

Write-Host ""
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "  SETUP COMPLETE!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host "1. Run: " -NoNewline -ForegroundColor White
Write-Host "python check_database_setup.py" -ForegroundColor Yellow
Write-Host "2. Verify both databases are connected" -ForegroundColor White
Write-Host "3. Start your application: " -NoNewline -ForegroundColor White
Write-Host "python simple_main.py" -ForegroundColor Yellow
Write-Host ""
