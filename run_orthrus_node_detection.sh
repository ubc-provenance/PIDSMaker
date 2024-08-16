nohup sh -c "\
./run_serial.sh best_model THEIA_E3 --exp=THEIA_E3 && \
./run_serial.sh best_model CADETS_E3 --exp=CADETS_E3 --detection.gnn_training.num_epochs=20 --detection.gnn_training.node_out_dim=128  && \
./run_serial.sh best_model CLEARSCOPE_E3 --exp=CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.evaluation.node_evaluation.kmeans_top_K=50" &
nohup sh -c "\
./run_serial.sh best_model THEIA_E5 --exp=THEIA_E5 --detection.gnn_training.lr=0.000005 && \
./run_serial.sh best_model CADETS_E5 --exp=CADETS_E5 && \
./run_serial.sh best_model CLEARSCOPE_E5 --exp=CLEARSCOPE_E5 --detection.evaluation.node_evaluation.kmeans_top_K=100" &
