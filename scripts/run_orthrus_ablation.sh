nohup sh -c "\
./run_serial.sh orthrus THEIA_E3 --exp=E3_embedding --featurization.feat_training.used_method=hierarchical_hashing && \
./run_serial.sh orthrus THEIA_E5 --exp=E5_embedding --featurization.feat_training.used_method=hierarchical_hashing" &

nohup sh -c "\
./run_serial.sh orthrus THEIA_E3 --exp=E3_embedding_alacarte --featurization.feat_training.used_method=alacarte && \
./run_serial.sh orthrus THEIA_E5 --exp=E5_embedding_alacarte --featurization.feat_training.used_method=alacarte" &

nohup sh -c "\
./run_serial.sh orthrus THEIA_E3 --exp=E3_encoding --detection.gnn_training.encoder.tgn.use_memory=True --detection.gnn_training.decoder.predict_edge_type.used_method=kairos  && \
./run_serial.sh orthrus THEIA_E5 --exp=E5_encoding --detection.gnn_training.encoder.tgn.use_memory=True --detection.gnn_training.decoder.predict_edge_type.used_method=kairos" &

nohup sh -c "\
./run_serial.sh orthrus THEIA_E3 --exp=E3_clustering --detection.evaluation.node_evaluation.use_kmeans=False  && \
./run_serial.sh orthrus THEIA_E5 --exp=E5_clustering --detection.evaluation.node_evaluation.use_kmeans=False" &
