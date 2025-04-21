##########################################################################################
# Some of the code is adapted from:
# https://github.com/NLPrinceton/ALaCarte/blob/master/alacarte.py
##########################################################################################

import os
from collections import Counter, OrderedDict, defaultdict
from unicodedata import category

import numpy as np
import torch
from gensim.models import Word2Vec

from pidsmaker.utils import log, log_start

FLOAT = np.float32
INT = np.uint64
CATEGORIES = {"M", "P", "S"}
MAXTOKLEN = 1000


def load_data(fname):
    """Load path data (from walks) to memory.

    :param fname: the name of the file (string) that contains path data
    :return: List object that contains all walk paths
    """
    paths = []
    with open(fname, "r") as f:
        path = f.readline()
        while path:
            paths.append(path.replace("\n", "").split(","))
            path = f.readline()

    return paths


def load_vectors(wv):
    """loads word embeddings from word2vec model.wv
    Args:
        wv: word2vec output keyed-vectors
    Returns:
        (word, vector) generator
    """
    words = set()
    for w in wv.key_to_index:
        if w not in words:
            words.add(w)
            yield w, np.array(wv[w], dtype=FLOAT)


def ranksize(comm=None):
    """returns rank and size of MPI Communicator
    Args:
        comm: MPI Communicator
    Returns:
        int, int
    """

    if comm is None:
        return 0, 1
    return comm.rank, comm.size


def checkpoint(comm=None):
    """waits until all processes have reached this point
    Args:
        comm: MPI Communicator
    """

    if comm is not None:
        comm.allgather(0)


def is_punctuation(char):
    """checks if unicode character is punctuation"""

    return category(char)[0] in CATEGORIES


class ALaCarteReader:
    """reads documents and updates context vectors"""

    def __init__(
        self, w2v, targets, wnd=10, checkpoint=None, interval=[0, float("inf")], comm=None
    ):
        """initializes context vector dict as self.c2v and counts as self.target_counts
        Args:
            w2v: {word: vector} dict of source word embeddings
            targets: iterable of targets to find context embeddings for
            wnd: context window size (uses this number of words on each side)
            checkpoint: path to HDF5 checkpoint file (both for recovery and dumping)
            interval: corpus start and stop positions
            comm: MPI Communicator
        """

        self.w2v = w2v
        self.combined_vocab = self.w2v

        gramlens = {len(target.split(",")) for target in targets if target}
        self.max_n = max(gramlens)
        if self.max_n > 1:
            self.targets = [tuple(target.split()) for target in targets]
            self.target_vocab = set(self.targets)
            self.combined_vocab = {word for target in targets for word in target.split()}.union(
                self.combined_vocab
            )
        else:
            self.targets = targets
            self.target_vocab = set(targets)
            self.combined_vocab = self.target_vocab.union(self.combined_vocab)
        self.target_counts = Counter()

        dimension = next(iter(self.w2v.values())).shape[0]
        self.dimension = dimension
        self.zero_vector = np.zeros(self.dimension, dtype=FLOAT)
        self.c2v = defaultdict(lambda: np.zeros(dimension, dtype=FLOAT))

        self.wnd = wnd
        self.learn = len(self.combined_vocab) == len(self.target_vocab) and self.max_n == 1

        self.datafile = checkpoint
        self.comm = comm
        self.rank, self.size = ranksize(comm)
        position = interval[0]

        if self.rank:
            self.vector_array = FLOAT(0.0)
            self.count_array = INT(0)

        elif checkpoint is None or not os.path.isfile(checkpoint):
            self.vector_array = np.zeros((len(self.targets), dimension), dtype=FLOAT)
            self.count_array = np.zeros(len(self.targets), dtype=INT)

        else:
            import h5py

            f = h5py.File(checkpoint, "r")
            position = f.attrs["position"]
            assert interval[0] <= position < interval[1], (
                "checkpoint position must be inside corpus interval"
            )
            self.vector_array = np.array(f["vectors"])
            self.count_array = np.array(f["counts"])

        self.position = comm.bcast(position, root=0) if self.size > 1 else position
        self.stop = interval[1]

    def reduce(self):
        """reduces data to arrays at the root process"""

        comm, rank, size = self.comm, self.rank, self.size
        targets = self.targets

        c2v = self.c2v
        dimension = self.dimension
        vector_array = np.vstack(
            [c2v.pop(target, np.zeros(dimension, dtype=FLOAT)) for target in targets]
        )

        target_counts = self.target_counts
        count_array = np.array([target_counts.pop(target, 0) for target in targets], dtype=INT)

        if rank:
            comm.Reduce(vector_array, None, root=0)
            comm.Reduce(count_array, None, root=0)
        elif size > 1:
            comm.Reduce(self.vector_array + vector_array, self.vector_array, root=0)
            comm.Reduce(self.count_array + count_array, self.count_array, root=0)
        else:
            self.vector_array += vector_array
            self.count_array += count_array

    def checkpoint(self, position):
        """dumps data to HDF5 checkpoint
        Args:
            position: reader position
        Returns:
            None
        """

        datafile = self.datafile
        assert datafile is not None, "no checkpoint file specified"
        self.reduce()

        if not self.rank:
            import h5py

            f = h5py.File(datafile + "~tmp", "w")
            f.attrs["position"] = position
            f.create_dataset("vectors", data=self.vector_array, dtype=FLOAT)
            f.create_dataset("counts", data=self.count_array, dtype=INT)
            f.close()
            if os.path.isfile(datafile):
                os.remove(datafile)
            os.rename(datafile + "~tmp", datafile)
        self.position = position

    def target_coverage(self):
        """returns fraction of targets covered (as a string)
        Args:
            None
        Returns:
            str (empty on non-root processes)
        """

        if self.rank:
            return ""
        return str(sum(self.count_array > 0)) + "/" + str(len(self.targets))

    def read_ngrams(self, tokens):
        """reads tokens and updates context vectors
        Args:
            tokens: list of strings
        Returns:
            None
        """

        import nltk

        # gets location of target n-grams in document
        target_vocab = self.target_vocab
        max_n = self.max_n
        ngrams = dict()
        for n in range(1, max_n + 1):
            ngrams[n] = list(
                filter(lambda entry: entry[1] in target_vocab, enumerate(nltk.ngrams(tokens, n)))
            )

        for n in range(1, max_n + 1):
            if ngrams[n]:
                # gets word embedding for each token
                w2v = self.w2v
                zero_vector = self.zero_vector
                wnd = self.wnd
                start = max(0, ngrams[n][0][0] - wnd)
                vectors = [None] * start + [
                    w2v.get(token, zero_vector) if token else zero_vector
                    for token in tokens[start : ngrams[n][-1][0] + n + wnd]
                ]
                c2v = self.c2v
                target_counts = self.target_counts

                # computes context vector around each target n-gram
                for i, ngram in ngrams[n]:
                    c2v[ngram] += sum(vectors[max(0, i - wnd) : i], zero_vector) + sum(
                        vectors[i + n : i + n + wnd], zero_vector
                    )
                    target_counts[ngram] += 1

    def read_document(self, document):
        """reads document and updates context vectors
        Args:
            document: str
        Returns:
            None
        """

        # tokenizes document
        tokens = [token for token in document.split(",")]
        if self.max_n > 1:
            return self.read_ngrams(tokens)

        # eliminates tokens not within the window of a target word
        T = len(tokens)
        wnd = self.wnd
        learn = self.learn
        if learn:
            check = bool
        else:
            target_vocab = self.target_vocab
            check = lambda token: token in target_vocab
        try:
            start = max(0, next(i for i, token in enumerate(tokens) if check(token)) - wnd)
        except StopIteration:
            return None
        stop = (
            next(i for i, token in zip(reversed(range(T)), reversed(tokens)) if check(token))
            + 1
            + wnd
        )
        tokens = tokens[start:stop]
        T = len(tokens)

        # gets word embedding for each token
        w2v = self.w2v
        zero_vector = self.zero_vector
        vectors = [w2v.get(token, zero_vector) if token else zero_vector for token in tokens]
        context_vector = sum(vectors[: wnd + 1])
        c2v = self.c2v
        target_counts = self.target_counts

        # slides window over document
        for i, (token, vector) in enumerate(zip(tokens, vectors)):
            if token and (learn or token in target_vocab):
                c2v[token] += context_vector - vector
                target_counts[token] += 1
            if i < T - 1:
                right_index = wnd + 1 + i
                if right_index < T and tokens[right_index]:
                    context_vector += vectors[right_index]
                left_index = i - wnd
                if left_index > -1 and tokens[left_index]:
                    context_vector -= vectors[left_index]


def is_english(document):
    """checks if document is in English"""

    return True


def process_documents(func):
    """wraps document generator function to handle English-checking and lower-casing and to return data arrays"""

    def wrapper(string, reader, verbose=False, comm=None, english=False, lower=False):
        generator = (document for document in func(string, reader, verbose=verbose, comm=comm))
        # TODO: English-checking is not supported
        if english:
            generator = (document for document in generator if is_english(document))
        # TODO: lower-casing is not supported
        if lower:
            generator = (document.lower() for document in generator)

        for i, document in enumerate(generator):
            reader.read_document(document)

        reader.reduce()
        log("Finished Processing Corpus. Targets Covered: {}".format(reader.target_coverage()))
        return reader.vector_array, reader.count_array

    return wrapper


@process_documents
def corpus_documents(corpusfile, reader, verbose=False, comm=None):
    """iterates of text document
    Args:
        corpusfile: text file with a document on each line
        reader: ALaCarteReader object
        verbose: display progress
        comm: MPI Communicator
    Returns:
        str generator distributing documents across processes
    """

    position = reader.position
    # TODO: MPI is not supported
    rank, size = ranksize(comm)

    with open(corpusfile, "r") as f:
        f.seek(position)
        line = f.readline()
        i = 0
        while line:
            if i and not i % 1000000:
                reader.reduce()
                # TODO: checkpoint is not supported
                if reader.datafile is not None:
                    reader.checkpoint(f.tell())
            if i >= reader.stop:
                break

            # yielding every line
            if i % size == rank:
                yield line.strip()

            line = f.readline()
            i += 1


def dump_vectors(generator, vectorfile):
    """Saves embeddings to .txt
    Args:
        generator: (gram, vector) generator; vector can also be a scalar
        vectorfile: .txt file
    Returns:
        None
    """

    with open(vectorfile, "w") as f:
        for gram, vector in generator:
            numstr = " ".join(map(str, vector.tolist())) if vector.shape else str(vector)
            f.write(gram + " " + numstr + "\n")


def obtain_targets_from_file(input_path, w2v):
    """Get targets (unknown OOVs, not in w2v) from a file
    Args:
        input_path: the input file path to get targets
        w2v: dict that holds existing words
    Returns:
        list of str
    """
    ret = set()
    # csv_graph = pd.read_csv(input_path, header=None)
    f = open(input_path, "r")
    for line in f:
        edge = [""] + line.strip().split(",")

        # src_name = edge[3]
        src_name = str(edge[1])
        if src_name not in w2v:
            ret.add(src_name)

        # dst_name = edge[4]
        dst_name = str(edge[2])
        if dst_name not in w2v:
            ret.add(dst_name)

        # edge_name does not get segmented
        edge_name = edge[5]
        if edge_name not in w2v:
            ret.add(edge_name)

    return ret


def embed_nodes_for_one_split(
    split: str,
    use_corpus: bool,
    use_matrix_input: bool,
    use_pretrained_model: bool,
    cfg,
    verbose=True,
):
    out_dir = cfg.featurization.embed_nodes.word2vec._vec_graphs_dir
    adjacency_dir = os.path.join(
        cfg.featurization.embed_nodes.word2vec._random_walk_dir, f"{split}-adj"
    )
    dataset = os.path.join(
        cfg.featurization.embed_nodes.word2vec._random_walk_dir, f"{split}_set_corpus.csv"
    )
    corpus_dir = cfg.featurization.embed_nodes.word2vec._random_walk_corpus_dir
    corpus = dataset if use_corpus else None
    matrix_input = os.path.join(out_dir, "matrix.bin") if use_matrix_input else None
    model_input = os.path.join(out_dir, "model.bin") if use_pretrained_model else None

    epochs = cfg.featurization.embed_nodes.epochs
    emb_dim = cfg.featurization.embed_nodes.emb_dim
    window_size = cfg.featurization.embed_nodes.word2vec.context_window_size
    min_count = cfg.featurization.embed_nodes.word2vec.min_count
    use_skip_gram = cfg.featurization.embed_nodes.word2vec.use_skip_gram
    num_workers = cfg.featurization.embed_nodes.word2vec.num_workers
    compute_loss = cfg.featurization.embed_nodes.word2vec.compute_loss
    add_paths = cfg.featurization.embed_nodes.word2vec.add_paths
    use_seed = cfg.featurization.embed_nodes.use_seed
    SEED = 0

    log_dir = out_dir

    log(
        "=== PARAMETER SUMMARY ----------------------------------------------------------------------"
    )
    log("Training data: {}".format(dataset))
    log("Total number of epochs: {}".format(epochs))
    if model_input is None:
        log("Node vector dimensionality: {}".format(emb_dim))
        # adjust word2vec context window size for segmentation
        log("word2vec context window size: {}".format(window_size))
        log("Minimal count threshold: {}".format(min_count))
        log("Training algorithm: {}".format("CBOW" if not use_skip_gram else "Skip-Gram"))
        log("Number of working threads: {}".format(num_workers))
    else:
        log("Other parameters follow the input model from: {}".format(model_input))
        if add_paths:
            log("Current training data is additional training data: {}".format(add_paths))
    log("Context information for OOV embedding learning is in directory: {}".format(corpus_dir))
    # adjust A La Carte context window size for segmentation
    log("A La Carte context window size: {}".format(window_size))
    log("Model is used to fulfill adjacency lists in directory: {}".format(adjacency_dir))
    log(
        "=== ----------------------------------------------------------------------------------------"
    )

    # ===-----------------------------------------------------------------------===
    # Importing data (only if Word2Vec model needs to be trained or modified)
    # ===-----------------------------------------------------------------------===
    if model_input is None or add_paths:
        paths = load_data(dataset)
        if verbose:
            log("{} paths loaded".format(len(paths)))

    # ===-----------------------------------------------------------------------===
    # Training using Word2Vec if needed
    # ===-----------------------------------------------------------------------===
    if model_input is None:
        if use_seed:
            model = Word2Vec(
                paths,
                vector_size=emb_dim,
                window=window_size,
                min_count=min_count,
                sg=use_skip_gram,
                workers=num_workers,
                epochs=epochs,
                compute_loss=compute_loss,
                seed=SEED,
            )
        else:
            model = Word2Vec(
                paths,
                vector_size=emb_dim,
                window=window_size,
                min_count=min_count,
                sg=use_skip_gram,
                workers=num_workers,
                epochs=epochs,
                compute_loss=compute_loss,
            )
    else:
        log("Loading existing model from: {}".format(model_input))
        model = Word2Vec.load(model_input)
        if add_paths:
            log("Resuming training using additional data")
            model.train(paths, epochs=epochs, compute_loss=compute_loss)

    # Note: currently word2vec outputs normalized vectors (and replaces the original un-normalized ones)
    model.init_sims(replace=True)
    wv = model.wv
    log("Trained embedding vectors have shape: {}".format(wv.vectors.shape))

    # ===-----------------------------------------------------------------------===
    # Saving models
    # ===-----------------------------------------------------------------------===
    # Save the model, which supports later online training (e.g., add addition training paths)
    if model_input is None or add_paths:
        out_path = os.path.join(log_dir, "model.bin")
        log("Saving the model at: {}".format(out_path))
        model.save(out_path)

    # ===-----------------------------------------------------------------------===
    # Using trained embeddings to train A La Carte model and save the model
    # or use an existing A La Carte matrix model
    # ===-----------------------------------------------------------------------===
    log("Loading w2v embeddings as source embeddings to A La Carte model")
    w2v = OrderedDict(load_vectors(wv))

    if matrix_input is not None:
        M = np.fromfile(matrix_input, dtype=FLOAT)
        d = int(np.sqrt(M.shape[0]))
        assert d == next(iter(w2v.values())).shape[0], (
            "induction matrix dimension and word embedding dimension must be the same"
        )
        M = M.reshape(d, d)
    else:
        matrix_file = os.path.join(log_dir, "matrix.bin")
        log("Learning induction matrix and saving to {}".format(matrix_file))
        targets = w2v.keys()
        M = None

        alc = ALaCarteReader(w2v, targets, wnd=window_size, checkpoint=None, comm=None)

        log("Building A La Carte context vectors")
        if corpus:
            context_vectors = FLOAT(0.0)
            target_counts = INT(0)
            log("Source corpus: {}".format(corpus))
            context_vectors, target_counts = corpus_documents(
                corpus, alc, verbose=verbose, comm=None, english=None, lower=None
            )
        else:
            log("At least one corpus file is required by A La Carte model to learn")
            exit(1)

        nz = target_counts > 0

        # building A La Carte matrix
        from sklearn.linear_model import LinearRegression as LR
        from sklearn.preprocessing import normalize

        log("Learning induction matrix")
        X = np.true_divide(context_vectors[nz], target_counts[nz, None], dtype=FLOAT)
        Y = np.vstack([vector for vector, count in zip(w2v.values(), target_counts) if count])
        M = LR(fit_intercept=False).fit(X, Y).coef_.astype(FLOAT)
        log(
            "Finished learning transform; Average cosine similarity: {}".format(
                np.mean(np.sum(normalize(X.dot(M.T)) * normalize(Y), axis=1))
            )
        )

        log("Saving induction transform to {}".format(matrix_file))
        dump_vectors(zip(targets, target_counts), log_dir + "/source_vocab_counts.txt")
        context_vectors.tofile(log_dir + "/source_context_vectors.bin")
        M.tofile(matrix_file)

    log("Loading adjacency lists from: {}".format(adjacency_dir))

    # if split in {'train', 'val'}:
    if True:
        try:
            indexid2vec = torch.load(f"{out_dir}/indexid2vec")
        except:
            indexid2vec = {}

        try:
            edgelabel2vec = torch.load(f"{out_dir}/edgelabel2vec")
        except:
            edgelabel2vec = {}

        for filename in sorted(os.listdir(adjacency_dir)):
            if filename:
                log("Loading adjacency list: {}".format(filename))
                input_path = os.path.join(adjacency_dir, filename)

                # Going through the input data to find all the targets
                log("Loading targets from: {}".format(input_path))
                targets = obtain_targets_from_file(input_path, w2v)
                log("A total number of {} targets identified".format(len(targets)))
                # target_dict now holds all unknown OOV embeddings
                # TODO: we also manually add the new embeddings into the word2vec keyed vectors (see wv.add method)
                target_dict = {}
                if not len(targets):
                    log("No uncovered targets found")
                else:
                    # reloading ALC reader for new targets
                    alc = ALaCarteReader(w2v, targets, wnd=window_size, checkpoint=None, comm=None)
                    log("Rebuilding A La Carte context vectors for {}".format(filename))
                    corpus_file = os.path.join(adjacency_dir, filename)
                    test_context_vectors = FLOAT(0.0)
                    test_target_counts = INT(0)
                    log("Source corpus for {}: {}".format(filename, corpus_file))
                    test_context_vectors, test_target_counts = corpus_documents(
                        corpus_file, alc, verbose=verbose, comm=None, english=None, lower=None
                    )
                    test_nz = test_target_counts > 0

                    dump_vectors(
                        zip(targets, test_target_counts),
                        log_dir + "/" + filename + "_target_vocab_counts.txt",
                    )
                    # Generate feature vectors for targets
                    test_context_vectors[test_nz] = np.true_divide(
                        test_context_vectors[test_nz],
                        test_target_counts[test_nz, None],
                        dtype=FLOAT,
                    )
                    target_vecs = test_context_vectors.dot(M.T)
                    for gram, vector in zip(targets, target_vecs):
                        if np.count_nonzero(vector) == 0:
                            log("[!]Zero-vector target: {}".format(gram))
                            # assert (np.count_nonzero(vector) > 0), "zero-vector target could result in NaN"
                        else:
                            vector = vector / np.linalg.norm(vector)
                        target_dict[gram] = vector

                csv_graph = open(input_path, "r")
                for line in csv_graph:
                    edge = [""] + line.strip().split(",")

                    srcid = str(edge[1])
                    dstid = str(edge[2])
                    # src_name = edge[3]
                    # dst_name = edge[4]
                    src_name = str(edge[1])
                    dst_name = str(edge[2])

                    edge_name = edge[5]

                    # Source Node
                    if src_name in wv:
                        src_feature = wv[src_name]
                        assert len(src_feature) == emb_dim, (
                            "src feature dimension from wv is {}, not {}".format(
                                len(src_feature), emb_dim
                            )
                        )
                    # Note: if we manually expand wv, we should not take this branch
                    elif src_name in target_dict:
                        src_feature = target_dict[src_name]
                        assert len(src_feature) == emb_dim, (
                            "src feature dimension from alc is {}, not {}".format(
                                len(src_feature), emb_dim
                            )
                        )
                    else:
                        log(f"Unknown node name for source node {srcid}: {src_name}")

                    if srcid not in indexid2vec:
                        indexid2vec[srcid] = src_feature

                    # Destination Node
                    if dst_name in wv:
                        dst_feature = wv[dst_name]
                        assert len(dst_feature) == emb_dim, (
                            "dst feature dimension from wv is {}, not {}".format(
                                len(dst_feature), emb_dim
                            )
                        )
                    elif dst_name in target_dict:
                        dst_feature = target_dict[dst_name]
                        assert len(dst_feature) == emb_dim, (
                            "dst feature dimension from alc is {}, not {}".format(
                                len(dst_feature), emb_dim
                            )
                        )
                    else:
                        log(f"Unknown node name for destination node {dstid}: {dst_name}")

                    if dstid not in indexid2vec:
                        indexid2vec[dstid] = dst_feature

                    # Edge type
                    if edge_name in wv:
                        edge_feature = wv[edge_name]
                        assert len(edge_feature) == emb_dim, (
                            "edge feature dimension from wv is {}, not {}".format(
                                len(edge_feature), emb_dim
                            )
                        )
                    elif edge_name in target_dict:
                        edge_feature = target_dict[edge_name]
                        assert len(edge_feature) == emb_dim, (
                            "edge feature dimension from wv is {}, not {}".format(
                                len(edge_feature), emb_dim
                            )
                        )
                    else:
                        log("Unknown edge name: {}".format(edge_name))

                    if edge_name not in edgelabel2vec:
                        edgelabel2vec[edge_name] = edge_feature

        # The one after validation is used in embedding.py
        # torch.save(indexid2vec, f"{out_dir}/indexid2vec_{split}")
        torch.save(indexid2vec, f"{out_dir}/indexid2vec")
        torch.save(edgelabel2vec, f"{out_dir}/edgelabel2vec")

    # ===-----------------------------------------------------------------------===
    # Saving embeddings (for reuse or checking embedding quality)
    # ===-----------------------------------------------------------------------===
    embed_file_name = "model.wv"
    log("Saving the embeddings at: {}".format(log_dir + "/" + embed_file_name))
    torch.save(wv, log_dir + "/" + embed_file_name)
    # ===-----------------------------------------------------------------------===
    # Output some stats and results
    # ===-----------------------------------------------------------------------===
    log(
        "=== RESULTS REPORT -------------------------------------------------------------------------"
    )
    w2v_loss = model.get_latest_training_loss()
    log("word2vec Training loss: {}".format(w2v_loss))
    log(
        "=== ----------------------------------------------------------------------------------------"
    )


def main(cfg):
    log_start(__file__)

    os.makedirs(cfg.featurization.embed_nodes.word2vec._vec_graphs_dir, exist_ok=True)

    # In test mode, we only have access to
    if cfg._test_mode:
        embed_nodes_for_one_split(
            "train", use_corpus=True, use_matrix_input=False, use_pretrained_model=False, cfg=cfg
        )
    else:
        embed_nodes_for_one_split(
            "train", use_corpus=True, use_matrix_input=False, use_pretrained_model=False, cfg=cfg
        )
        embed_nodes_for_one_split(
            "val", use_corpus=False, use_matrix_input=True, use_pretrained_model=True, cfg=cfg
        )
        embed_nodes_for_one_split(
            "test", use_corpus=False, use_matrix_input=True, use_pretrained_model=True, cfg=cfg
        )
