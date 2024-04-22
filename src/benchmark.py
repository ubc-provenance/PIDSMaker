from preprocessing import (
    build_graphs,
)
from featurization import (
    build_random_walks,
    embed_nodes,
    embed_edges,
)
from detection import (
    gnn_training,
    gnn_testing,
)

from config import (
    get_yml_cfg,
    get_runtime_required_args,
)


def main(cfg):
    modified_tasks = {subtask: restart for subtask, restart in cfg._subtasks_should_restart}
    should_restart = {subtask: restart for subtask, restart in cfg._subtasks_should_restart_with_deps}
    
    print("\n" + ("*" * 100))
    print("Tasks modified since last runs:")
    print("  =>  ".join([f"{subtask}({restart})" for subtask, restart in modified_tasks.items()]))

    print("\nTasks requiring re-execution:")
    print("  =>  ".join([f"{subtask}({restart})" for subtask, restart in should_restart.items()]))
    print(("*" * 100) + "\n")


    # Preprocessing
    if should_restart["build_graphs"]:
        build_graphs.main(cfg)
    
    # Featurization
    if should_restart["build_random_walks"]:
        build_random_walks.main(cfg)
    if should_restart["embed_nodes"]:
        embed_nodes.main(cfg)
    if should_restart["embed_edges"]:
        embed_edges.main(cfg)
    if should_restart["gnn_training"]:
        gnn_training.main(cfg)
    if should_restart["gnn_testing"]:
        gnn_testing.main(cfg)


if __name__ == '__main__':
    args = get_runtime_required_args()
    cfg = get_yml_cfg(args)

    main(cfg)
