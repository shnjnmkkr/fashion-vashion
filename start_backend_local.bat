@echo off
echo 🚀 Starting Fashion-Vashion Backend Locally...
echo.

cd backend

echo 📦 Installing dependencies...
pip install -r requirements.txt

echo.
echo 🔧 Starting backend server...
echo 📍 Backend will be available at: http://localhost:8000
echo 🌐 Frontend can connect to: http://localhost:8000
echo.
echo 💡 Make sure your frontend is deployed and configured to use this backend URL
echo.

python start_backend.py

pause 