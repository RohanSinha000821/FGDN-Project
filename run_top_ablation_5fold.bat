@echo off
cd /d D:\FGDN_Project
call venv\Scripts\activate

REM ============================================================
REM Candidate A: hidden_channels = 128
REM ============================================================
for %%F in (1 2 3 4 5) do (
  python -m src.training.train_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 5 --fold %%F --device cuda --hidden-channels 128 --cheb-k 3 --dropout 0.1 --batch-size 16 --epochs 100 --lr 0.0001 --weight-decay 0.0005 --patience 20 --template-k 20 --output-root outputs/ablations/AAL_5fold_confirm/hidden_channels_128/fold_%%F --log-root outputs/ablations/AAL_5fold_confirm/hidden_channels_128/fold_%%F/logs
  python -m src.evaluation.evaluate_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 5 --fold %%F --device cuda --checkpoint-type best --output-root outputs/ablations/AAL_5fold_confirm/hidden_channels_128/fold_%%F
)

REM ============================================================
REM Candidate B: dropout = 0.0
REM ============================================================
for %%F in (1 2 3 4 5) do (
  python -m src.training.train_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 5 --fold %%F --device cuda --hidden-channels 64 --cheb-k 3 --dropout 0.0 --batch-size 16 --epochs 100 --lr 0.0001 --weight-decay 0.0005 --patience 20 --template-k 20 --output-root outputs/ablations/AAL_5fold_confirm/dropout_0p0/fold_%%F --log-root outputs/ablations/AAL_5fold_confirm/dropout_0p0/fold_%%F/logs
  python -m src.evaluation.evaluate_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 5 --fold %%F --device cuda --checkpoint-type best --output-root outputs/ablations/AAL_5fold_confirm/dropout_0p0/fold_%%F
)

REM ============================================================
REM Candidate C: cheb_k = 1
REM ============================================================
for %%F in (1 2 3 4 5) do (
  python -m src.training.train_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 5 --fold %%F --device cuda --hidden-channels 64 --cheb-k 1 --dropout 0.1 --batch-size 16 --epochs 100 --lr 0.0001 --weight-decay 0.0005 --patience 20 --template-k 20 --output-root outputs/ablations/AAL_5fold_confirm/cheb_k_1/fold_%%F --log-root outputs/ablations/AAL_5fold_confirm/cheb_k_1/fold_%%F/logs
  python -m src.evaluation.evaluate_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 5 --fold %%F --device cuda --checkpoint-type best --output-root outputs/ablations/AAL_5fold_confirm/cheb_k_1/fold_%%F
)

REM ============================================================
REM Candidate D: template_k = 10
REM ============================================================
for %%F in (1 2 3 4 5) do (
  python -m src.training.train_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 5 --fold %%F --device cuda --hidden-channels 64 --cheb-k 3 --dropout 0.1 --batch-size 16 --epochs 100 --lr 0.0001 --weight-decay 0.0005 --patience 20 --template-k 10 --output-root outputs/ablations/AAL_5fold_confirm/template_k_10/fold_%%F --log-root outputs/ablations/AAL_5fold_confirm/template_k_10/fold_%%F/logs
  python -m src.evaluation.evaluate_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 5 --fold %%F --device cuda --checkpoint-type best --output-root outputs/ablations/AAL_5fold_confirm/template_k_10/fold_%%F
)

REM ============================================================
REM Candidate E: combined exploratory candidate
REM ============================================================
for %%F in (1 2 3 4 5) do (
  python -m src.training.train_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 5 --fold %%F --device cuda --hidden-channels 128 --cheb-k 1 --dropout 0.0 --batch-size 16 --epochs 100 --lr 0.00005 --weight-decay 0.0005 --patience 20 --template-k 20 --output-root outputs/ablations/AAL_5fold_confirm/combined_top/fold_%%F --log-root outputs/ablations/AAL_5fold_confirm/combined_top/fold_%%F/logs
  python -m src.evaluation.evaluate_fgdn --project-root D:/FGDN_Project --atlas AAL --kind tangent --num-folds 5 --fold %%F --device cuda --checkpoint-type best --output-root outputs/ablations/AAL_5fold_confirm/combined_top/fold_%%F
)

python -m src.ablation.summarize_ablations --project-root D:/FGDN_Project

echo Done.
pause