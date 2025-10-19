# 🔧 Complete MySQL Installation Guide for AI Duty Officer Assistant

## Step 1: Download MySQL Installer (2 minutes)

1. **Visit**: https://dev.mysql.com/downloads/installer/
2. **Download**: Click "Windows (x86, 32-bit), MSI Installer" (the larger one, ~400MB)
   - File name: `mysql-installer-community-8.0.XX.X.msi`
3. **Click**: "No thanks, just start my download" (you don't need an Oracle account)
4. **Save** the file to your Downloads folder

## Step 2: Install MySQL (5 minutes)

1. **Run** the downloaded MSI installer (double-click it)
2. **Accept** the license agreement
3. **Choose Setup Type**: 
   - Select **"Server only"** (we only need MySQL Server)
   - Click **Next**

4. **Installation**:
   - Click **Execute** to install
   - Wait for installation to complete (green checkmarks)
   - Click **Next**

5. **Product Configuration**:
   - Click **Next** (use default settings)

6. **Type and Networking**:
   - Config Type: **Development Computer**
   - Port: **3306** (default)
   - Leave all other settings as default
   - Click **Next**

7. **Authentication Method**:
   - Choose: **"Use Strong Password Encryption"** (recommended)
   - Click **Next**

8. **Accounts and Roles** (IMPORTANT!):
   - **Root Password**: Enter a simple password you'll remember
     - Suggestion: `root123` or `password` (for development)
     - ⚠️ **WRITE THIS DOWN!** You'll need it later
   - Click **Next**

9. **Windows Service**:
   - Configure MySQL Server as Windows Service: **Checked**
   - Service Name: **MySQL80**
   - Start at System Startup: **Checked**
   - Run as: **Standard System Account**
   - Click **Next**

10. **Server File Permissions**:
    - Use default settings
    - Click **Next**

11. **Apply Configuration**:
    - Click **Execute**
    - Wait for all steps to complete (green checkmarks)
    - Click **Finish**

12. **Installation Complete**:
    - Click **Next**, then **Finish**

## Step 3: Verify MySQL Installation (30 seconds)

Run this in PowerShell:

```powershell
Get-Service -Name MySQL80
```

**Expected output**:
```
Name     Status DisplayName
----     ------ -----------
MySQL80 Running MySQL80
```

If Status shows "Stopped", start it:
```powershell
Start-Service MySQL80
```

## Step 4: Add MySQL to System PATH (1 minute)

This allows you to use `mysql` command from anywhere.

### Option A: Automatic (Recommended)
Run this PowerShell command as **Administrator**:

```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\MySQL\MySQL Server 8.0\bin", [EnvironmentVariableTarget]::Machine)
```

Then **restart PowerShell** (close and reopen).

### Option B: Manual
1. Open **System Properties** → **Environment Variables**
2. Under **System variables**, find **Path**, click **Edit**
3. Click **New**, add: `C:\Program Files\MySQL\MySQL Server 8.0\bin`
4. Click **OK** on all dialogs
5. **Restart PowerShell**

## Step 5: Test MySQL Access (30 seconds)

In a **NEW PowerShell window**:

```powershell
mysql --version
```

**Expected output**: `mysql  Ver 8.0.XX for Win64 on x86_64`

If you get "command not found", restart PowerShell or check PATH setup.

## Step 6: Test MySQL Login (30 seconds)

```powershell
mysql -u root -p
```

When prompted, enter the password you set in Step 2.

**Expected**: You should see `mysql>` prompt

To exit MySQL:
```sql
EXIT;
```

## Step 7: Run Automated Setup (2 minutes)

Now that MySQL is installed, let's set up the database automatically!

### Option 1: Using the Python Setup Script (Recommended)

```powershell
python setup_mysql.py
```

This will:
1. ✅ Test MySQL connection
2. ✅ Create `appdb` database
3. ✅ Import schema from `db.sql` (20 vessels, 21 containers, 20 EDI messages, 20 API events)
4. ✅ Update your `.env` file
5. ✅ Verify all data

### Option 2: Manual Setup (if script fails)

```powershell
# 1. Create database
mysql -u root -p -e "CREATE DATABASE appdb CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;"

# 2. Import schema (this will take 10-20 seconds)
Get-Content db.sql | mysql -u root -p appdb

# 3. Verify import
mysql -u root -p appdb -e "SELECT COUNT(*) as vessels FROM vessel; SELECT COUNT(*) as containers FROM container;"
```

**Expected output**:
```
+---------+
| vessels |
+---------+
|      20 |
+---------+
+------------+
| containers |
+------------+
|         21 |
+------------+
```

## Step 8: Update .env File (30 seconds)

Open `.env` file and update this line:

```properties
OPS_DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost/appdb
```

Replace `YOUR_PASSWORD` with the password you set in Step 2.

**Example** (if your password is `root123`):
```properties
OPS_DATABASE_URL=mysql+pymysql://root:root123@localhost/appdb
```

## Step 9: Verify Complete Setup (30 seconds)

```powershell
python check_database_setup.py
```

**Expected output**:
```
============================================================
1. CHECKING AI DATABASE (SQLite)
============================================================
✅ AI Database: CONNECTED
   - Knowledge Base Entries: 76
   - Training Data Entries: 323
   - RCA Analyses: 0

============================================================
2. CHECKING OPERATIONAL DATABASE (MySQL)
============================================================
✅ Operational Database: CONNECTED
   - Database Type: MySQL
   - Vessels: 20
   - Containers: 21
   - EDI Messages: 20
   - API Events: 20

============================================================
3. CHECKING PYTHON PACKAGES
============================================================
   ✅ All packages installed

============================================================
SUMMARY
============================================================
✅ YOUR PROGRAM CAN RUN!
   
✅ Full features available:
   - Quick Fix: ✅
   - RCA with log analysis: ✅
   - Operational data correlation: ✅
```

## Step 10: Start Your Application! (10 seconds)

```powershell
python simple_main.py
```

Then visit: **http://localhost:8002**

---

## 🎉 You're Done!

Your AI Duty Officer Assistant now has **FULL RCA capabilities** with operational data correlation!

### Test the Full Features:

1. **Quick Fix** - Analyze: "Container GESU1234567 stuck at gate"
2. **Root Cause Analysis** - Try: "Customer seeing duplicate CMAU0000020 containers"
   - The system will detect the actual duplicate in the database!
3. **Database Status** - View both databases connected

---

## 🆘 Troubleshooting

### "MySQL service won't start"
```powershell
# Check service status
Get-Service MySQL80

# Try starting it
Start-Service MySQL80

# If that fails, restart your computer
```

### "Access denied for user 'root'"
- You entered the wrong password
- Try the password you set during installation
- If forgotten, you'll need to reset it (see MySQL documentation)

### "mysql command not found"
- PATH wasn't updated correctly
- **Restart PowerShell completely**
- Or use full path: `"C:\Program Files\MySQL\MySQL Server 8.0\bin\mysql.exe"`

### "db.sql file not found"
- Make sure you're in the correct directory:
  ```powershell
  cd "C:\Users\TanJy\Downloads\Portnet-L2-Automator\AI Assistant Python"
  ```

### "Table doesn't exist" errors
- The import didn't complete
- Try manual import again:
  ```powershell
  Get-Content db.sql | mysql -u root -p appdb
  ```

### Still having issues?
Run this diagnostic:
```powershell
# Check if MySQL is running
Get-Service MySQL80

# Check if database exists
mysql -u root -p -e "SHOW DATABASES;"

# Check if tables exist
mysql -u root -p appdb -e "SHOW TABLES;"
```

---

## 📝 Quick Reference

### Useful MySQL Commands

```powershell
# Check MySQL service
Get-Service MySQL80

# Start MySQL
Start-Service MySQL80

# Stop MySQL
Stop-Service MySQL80

# Login to MySQL
mysql -u root -p

# Show databases
mysql -u root -p -e "SHOW DATABASES;"

# Show tables in appdb
mysql -u root -p appdb -e "SHOW TABLES;"

# Check vessel data
mysql -u root -p appdb -e "SELECT vessel_name, imo_no FROM vessel LIMIT 5;"

# Check container data
mysql -u root -p appdb -e "SELECT cntr_no, status FROM container LIMIT 5;"
```

---

**Time to Complete**: ~10 minutes total
**Difficulty**: Easy (just follow the steps!)
**Result**: Full RCA with operational data correlation 🚀
