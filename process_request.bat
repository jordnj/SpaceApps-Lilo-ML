

cd /home/NASASpaceApps/PMLDataPipeline-master

python ./download_eopatches.py

pipenv run python ./predict_scene.py --scene scenes/****.json --model models/median_20_both_L1C --method="median" --window=20

