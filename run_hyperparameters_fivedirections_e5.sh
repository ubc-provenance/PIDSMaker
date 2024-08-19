nohup sh -c "\
./run_serial.sh best_model FIVEDIRECTIONS_E5 --detection.gnn_training.num_epochs=10 --exp=FIVEDIRECTIONS_E5_h_128 --detection.gnn_training.node_out_dim=128 && \
./run_serial.sh best_model FIVEDIRECTIONS_E5 --detection.gnn_training.num_epochs=10 --exp=FIVEDIRECTIONS_E5_1min_word2vec_h_64 --featurization.embed_nodes.emb_dim=64 && \
./run_serial.sh best_model FIVEDIRECTIONS_E5 --detection.gnn_training.num_epochs=10 --exp=FIVEDIRECTIONS_E5_neig_10 --detection.gnn_training.encoder.tgn.tgn_neighbor_size=10 && \
./run_serial.sh best_model FIVEDIRECTIONS_E5 --detection.gnn_training.num_epochs=10 --exp=FIVEDIRECTIONS_E5_lr_0.000005 --detection.gnn_training.lr=0.000005" &
