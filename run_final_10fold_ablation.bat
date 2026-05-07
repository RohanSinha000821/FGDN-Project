@echo off
cd /d D:\FGDN_Project
call venv\Scripts\activate

REM ============================================================
REM Final Candidate 1: template_k = 10
REM Best 5-fold AUC candidate
REM ============================================================
for %%F in (1 2 3 4 5 6 7 8 9 10) do (
  python -m src.training.train_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 10 --fold %%F --device cuda --hidden-channels 64 --cheb-k 3 --dropout 0.1 --batch-size 16 --epochs 100 --lr 0.0001 --weight-decay 0.0005 --patience 20 --template-k 10 --output-root outputs/ablations/AAL_10fold_final/template_k_10/fold_%%F --log-root outputs/ablations/AAL_10fold_final/template_k_10/fold_%%F/logs
  python -m src.evaluation.evaluate_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 10 --fold %%F --device cuda --checkpoint-type best --output-root outputs/ablations/AAL_10fold_final/template_k_10/fold_%%F
)

REM ============================================================
REM Final Candidate 2: hidden_channels = 128
REM Best 5-fold accuracy candidate
REM ============================================================
for %%F in (1 2 3 4 5 6 7 8 9 10) do (
  python -m src.training.train_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 10 --fold %%F --device cuda --hidden-channels 128 --cheb-k 3 --dropout 0.1 --batch-size 16 --epochs 100 --lr 0.0001 --weight-decay 0.0005 --patience 20 --template-k 20 --output-root outputs/ablations/AAL_10fold_final/hidden_channels_128/fold_%%F --log-root outputs/ablations/AAL_10fold_final/hidden_channels_128/fold_%%F/logs
  python -m src.evaluation.evaluate_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 10 --fold %%F --device cuda --checkpoint-type best --output-root outputs/ablations/AAL_10fold_final/hidden_channels_128/fold_%%F
)

echo Done.
pause