nohup sh -c "\
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=1000 --exp=mimicry_1000 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=2000 --exp=mimicry_2000 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=3000 --exp=mimicry_3000 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=4000 --exp=mimicry_4000 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=5000 --exp=mimicry_5000 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=1500 --exp=mimicry_1500 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=2500 --exp=mimicry_2500 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=3500 --exp=mimicry_3500 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=4500 --exp=mimicry_4500" &