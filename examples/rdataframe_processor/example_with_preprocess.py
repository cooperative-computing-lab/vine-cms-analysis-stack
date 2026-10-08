#!/usr/bin/env python3
"""
Example showing how to use preprocessing with RDataFrame and VineReduce.
"""

import os.path as osp
import ndcctools.taskvine as vine
from vine_reduce import VineReduce, TaskVineDistributor
from vine_reduce.coffea_dataset_tools import preprocess
from rdataframe_processor import (
    make_source_instantiation,
    simple_rdataframe_processor,
    rdataframe_accumulator,
)


def main():
    manager = vine.Manager(port=0, name="rdataframe-preprocess-example")
    port = manager.port

    data = {
        "ZJets": {
            "files": {
                osp.abspath("samples/nano_dy.root"): {
                    "object_path": "Events",
                    "metadata": {
                        "dataset": "ZJets",
                        "checkusermeta": True,
                        "someusermeta": "hello",
                    },
                },
            },
            "metadata": {"checkusermeta": True, "someusermeta": "hello"},
        },
        "Data": {
            "files": {
                osp.abspath("samples/nano_dimuon.root"): {
                    "object_path": "Events",
                    "metadata": {
                        "dataset": "Data",
                        "checkusermeta": True,
                        "someusermeta2": "world",
                    },
                }
            },
            "metadata": {"checkusermeta": True, "someusermeta2": "world"},
        },
    }

    print("Original data spec:")
    for dataset_name, dataset_info in data.items():
        print(f"  {dataset_name}:")
        for file_path, file_info in dataset_info["files"].items():
            print(f"    {file_path}: {file_info}")

    workers = vine.Factory(manager_host_port=f"localhost:{port}", batch_type="local")
    workers.min_workers = 1
    workers.max_workers = 2
    workers.cores = 2
    workers.disk = 4096

    with workers:
        print("\nPreprocessing data with TaskVine...")
        preprocessed_data = preprocess(
            manager=manager,
            data=data,
            tree_name="Events",
            timeout=60,
            max_retries=3,
            show_progress=True,
            batch_size=5,
        )

        vr_data = {}
        for dataset_name, dataset_info in preprocessed_data.items():
            files_list = []
            for file_path, file_info in dataset_info["files"].items():
                meta = file_info.get("metadata", {}).copy()
                meta.setdefault("dataset", dataset_name)
                meta["object_path"] = file_info.get("object_path", "Events")
                files_list.append(
                    {
                        "file": file_path,
                        "numentries": file_info.get("num_entries", 0),
                        "metadata": meta,
                    }
                )
            vr_data[dataset_name] = {
                "metadata": dict(dataset_info.get("metadata") or {}),
                "files": files_list,
            }

        print("\nPreprocessed data spec:")
        for dataset_name, dataset_info in vr_data.items():
            print(f"  {dataset_name}:")
            for file_info in dataset_info["files"]:
                print(f"    {file_info['file']}: {file_info}")

        print("\nUsing preprocessed data with RDataFrame VineReduce...")
        distributor = TaskVineDistributor(
            manager=manager,
            resources_processing={
                "cores": 1,
                "disk": 1024,
            },
            verbose=True,
        )
        run = VineReduce(
            distributor=distributor,
            processors={"proc": simple_rdataframe_processor},
            data=vr_data,
            accumulator=rdataframe_accumulator,
            source_instantiation=make_source_instantiation(treepath="Events"),
            default_step_size=10000,
        )
        results = run.compute()

        print("\nResults:")
        print(results)


if __name__ == "__main__":
    main()
