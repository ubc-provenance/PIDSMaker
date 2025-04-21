nohup sh -c "\
python dataset_stats.py orthrus CADETS_E3 --wandb && \
python dataset_stats.py orthrus THEIA_E3 --wandb && \
python dataset_stats.py orthrus CLEARSCOPE_E3 --wandb && \
python dataset_stats.py orthrus CADETS_E5 --wandb && \
python dataset_stats.py orthrus THEIA_E5 --wandb && \
python dataset_stats.py orthrus CLEARSCOPE_E5 --wandb" > dataset_stats.log 2>&1 &
