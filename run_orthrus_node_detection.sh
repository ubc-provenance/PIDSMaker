nohup sh -c "\
./run_serial.sh best_model THEIA_E3 && \
./run_serial.sh best_model CADETS_E3 --detection.gnn_training.num_epochs=20 --detection.gnn_training.encoder.graph_attention.dropout=0.1 && \
./run_serial.sh best_model CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.encoder.graph_attention.dropout=0.1" &
nohup sh -c "\
./run_serial.sh best_model THEIA_E5 --detection.gnn_training.lr=0.000005 --preprocessing.build_graphs.use_all_files=False && \
./run_serial.sh best_model CADETS_E5 && \
./run_serial.sh best_model CLEARSCOPE_E5 --detection.gnn_training.num_epochs=10 --detection.gnn_training.lr=0.0001" &
