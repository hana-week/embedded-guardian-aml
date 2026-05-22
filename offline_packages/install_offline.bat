@echo off
echo ================================================
echo  임베디드 가디언 AML 모델 - 오프라인 패키지 설치
echo ================================================
echo.

:: Python 확인
python --version 2>nul
if errorlevel 1 (
    echo [오류] Python이 설치되지 않았습니다.
    echo Python 3.10 이상을 먼저 설치해주세요.
    pause
    exit /b 1
)

echo [1/2] 일반 패키지 설치 중...
pip install --no-index --find-links=. ^
    numpy ^
    pandas ^
    matplotlib ^
    scikit-learn ^
    networkx ^
    openpyxl ^
    python-docx

echo.
echo [2/2] PyTorch 설치 중 (CPU 버전)...
pip install --no-index --find-links=. torch

echo.
echo ================================================
echo  설치 완료! 아래 명령어로 실행하세요:
echo.
echo    cd ..\model
echo    python main.py --real
echo ================================================
pause
