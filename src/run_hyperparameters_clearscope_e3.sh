nohup sh -c "\
./run_serial.sh orthrus CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=20 --exp=CLEARSCOPE_E3_1min_neig_10 --detection.gnn_training.encoder.tgn.tgn_neighbor_size=10 && \
./run_serial.sh orthrus CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=20 --exp=CLEARSCOPE_E3_1min_dropout_0.1 --detection.gnn_training.encoder.graph_attention.dropout=0.1 && \
./run_serial.sh orthrus CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=20 --exp=CLEARSCOPE_E3_1min_h_32 --detection.gnn_training.node_out_dim=32" &

nohup sh -c "\
./run_serial.sh orthrus CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=20 --exp=CLEARSCOPE_E3_1min_h_128 --detection.gnn_training.node_out_dim=128 && \
./run_serial.sh orthrus CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=20 --exp=CLEARSCOPE_E3_1min_word2vec_h_32 --featurization.embed_nodes.emb_dim=32" &

nohup sh -c "\
./run_serial.sh orthrus CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=20 --exp=CLEARSCOPE_E3_1min_word2vec_h_64 --featurization.embed_nodes.emb_dim=64 && \
./run_serial.sh orthrus CLEARSCOPE_E3 --preprocessing.build_graphs.time_window_size=1.0 --detection.gnn_training.num_epochs=20 --exp=CLEARSCOPE_E3_1min_lr_0.0001 --detection.gnn_training.lr=0.0001" &
