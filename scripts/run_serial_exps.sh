nohup sh -c "\
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=500 --exp=new_gt_500 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=1000 --exp=new_gt_1000 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=1500 --exp=new_gt_1500 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=2000 --exp=new_gt_2000 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=2500 --exp=new_gt_2500" &

nohup sh -c "\
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=3000 --exp=new_gt_3000 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=3500 --exp=new_gt_3500 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=4000 --exp=new_gt_4000 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=4500 --exp=new_gt_4500 && \
./run_serial.sh mimicry THEIA_E3 --wandb --preprocessing.build_graphs.mimicry_edge_num=5000 --exp=new_gt_5000" &
