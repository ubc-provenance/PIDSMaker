nohup sh -c "\
./run_serial.sh nodlink CADETS_E5 --wandb && \
./run_serial.sh nodlink THEIA_E5 --wandb && \
./run_serial.sh nodlink CLEARSCOPE_E5 --wandb" &