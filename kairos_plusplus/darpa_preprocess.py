from config import *
from provnet_utils import *
import argparse
import os
import torch


def preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='input train log files path', required=True)
    parser.add_argument('--validation', help='input validation log files path', required=True)
    parser.add_argument('--test', help='input validation log files path', required=True)
    parser.add_argument('--walk_len', default=10, help='input the length of each random walk', type=int, required=False)
    args = parser.parse_args()


    # preparation
    train_adj_dir = preprocessed_dir + "train_adj_dir/"
    val_adj_dir = preprocessed_dir + "val_adj_dir/"
    test_adj_dir = preprocessed_dir + "test_adj_dir/"

    corpus = preprocessed_dir + "corpus/"
    training_set_corpus = preprocessed_dir + "training_set_corpus/"
    validation_set_corpus = preprocessed_dir + "validation_set_corpus/"
    testing_set_corpus = preprocessed_dir + "testing_set_corpus/"

    os.system("mkdir -p {}".format(train_adj_dir))
    os.system("mkdir -p {}".format(val_adj_dir))
    os.system("mkdir -p {}".format(test_adj_dir))


    os.system("mkdir -p {}".format(corpus))
    os.system("mkdir -p {}".format(training_set_corpus))
    os.system("mkdir -p {}".format(validation_set_corpus))
    os.system("mkdir -p {}".format(testing_set_corpus))


    # train
    train_files = sorted(os.listdir(args.train))
    train_g = []
    train_graph_info = open(f"{preprocessed_dir}/train_graph_info.csv","w")
    writer = csv.writer(train_graph_info)
    random_walks_file = corpus + "/" + "train.csv"
    random_walks_file_fd = open(random_walks_file, 'w')
    for file in train_files:
        adjacency_file = train_adj_dir + "/" + file + "-train.csv"
        full_file = args.train + '/' + file
        print("load file: ", full_file)
        graph = torch.load(full_file)
        gen_darpa_adj_files(graph, adjacency_file)
        train_g.append(graph)
        writer.writerow([
            adjacency_file,
            len(graph.nodes)
        ])
        train_corpus_file = training_set_corpus + "/" + file + "-train.csv"
        gen_darpa_rw_file(
            graph=graph,
            walk_len=args.walk_len,
            filename=train_corpus_file,
            adjfilename=adjacency_file,
            overall_fd=random_walks_file_fd)
    train_graph_info.close()
    random_walks_file_fd.close()

    # validation
    validation_files = sorted(os.listdir(args.validation))
    validation_g = []
    validation_graph_info = open(f"{preprocessed_dir}/val_graph_info.csv", "w")
    writer = csv.writer(validation_graph_info)
    random_walks_file = corpus + "/" + "validation.csv"
    random_walks_file_fd = open(random_walks_file, 'w')
    for file in validation_files:
        adjacency_file = val_adj_dir + "/" + file + "-validation.csv"
        full_file = args.validation + '/' + file
        print("load file: ", full_file)
        graph = torch.load(full_file)
        gen_darpa_adj_files(graph, adjacency_file)
        validation_g.append(graph)
        writer.writerow([
            adjacency_file,
            len(graph.nodes)
        ])
        val_corpus_file = validation_set_corpus + "/" + file + "-validation.csv"
        gen_darpa_rw_file(
            graph=graph,
            walk_len=args.walk_len,
            filename=val_corpus_file,
            adjfilename=adjacency_file,
            overall_fd=random_walks_file_fd)
    validation_graph_info.close()
    random_walks_file_fd.close()

    # test
    test_files = sorted(os.listdir(args.test))
    test_g = []
    test_graph_info = open(f"{preprocessed_dir}/test_graph_info.csv", "w")
    writer = csv.writer(test_graph_info)
    random_walks_file = corpus + "/" + "test.csv"
    random_walks_file_fd = open(random_walks_file, 'w')
    for file in test_files:
        adjacency_file = test_adj_dir + "/" + file + "-test.csv"
        full_file = args.test + '/' + file
        print("load file: ", full_file)
        graph = torch.load(full_file)
        gen_darpa_adj_files(graph, adjacency_file)
        test_g.append(graph)
        writer.writerow([
            adjacency_file,
            len(graph.nodes)
        ])
        test_corpus_file = testing_set_corpus + "/" + file + "-test.csv"
        gen_darpa_rw_file(
            graph=graph,
            walk_len=args.walk_len,
            filename=test_corpus_file,
            adjfilename=adjacency_file,
            overall_fd=random_walks_file_fd)
    test_graph_info.close()
    random_walks_file_fd.close()


if __name__ == "__main__":
    preprocess()
