@echo off
REM Runs the full pipeline in order. Q2 and Q3 depend on Q1's output files.
echo Starting Q1 (Influence)... > run_log.txt
echo %date% %time% >> run_log.txt
python q1_influence.py >> run_log.txt 2>&1
if %errorlevel% neq 0 ( echo Q1 FAILED >> run_log.txt & exit /b 1 )
echo Q1 DONE >> run_log.txt

echo Starting Q2 (Communities)... >> run_log.txt
echo %date% %time% >> run_log.txt
python q2_communities.py >> run_log.txt 2>&1
if %errorlevel% neq 0 ( echo Q2 FAILED >> run_log.txt & exit /b 1 )
echo Q2 DONE >> run_log.txt

echo Starting Q3 (Recommender)... >> run_log.txt
echo %date% %time% >> run_log.txt
python q3_recommender.py >> run_log.txt 2>&1
if %errorlevel% neq 0 ( echo Q3 FAILED >> run_log.txt & exit /b 1 )
echo Q3 DONE >> run_log.txt
echo ALL DONE >> run_log.txt
