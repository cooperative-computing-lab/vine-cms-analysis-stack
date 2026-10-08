#!/usr/bin/env python3
"""
Example showing how to use VineReduce with ROOT RDataFrame.
"""

import os.path as osp
import ndcctools.taskvine as vine
from vine_reduce import VineReduce, TaskVineDistributor, Step


class RDataFrameSource:
    """Worker-side Step → RDataFrame materialization."""

    def __init__(self, treepath: str = "Events"):
        self.treepath = treepath

    def source_instantiation(self, step: Step, **kwargs):
        import ROOT

        file_meta = dict(step.metadata.get("file") or {})
        dataset_meta = dict(step.metadata.get("dataset") or {})
        tree = (
            file_meta.get("object_path")
            or file_meta.get("treepath")
            or self.treepath
        )
        rdf = ROOT.RDataFrame(tree, step.file)
        rdf_chunk = rdf.Range(step.start, step.stop)
        chunk_info = {
            "file": step.file,
            "treepath": tree,
            "entry_start": step.start,
            "entry_stop": step.stop,
            "num_entries": step.size,
            "metadata": {
                **dataset_meta,
                **file_meta,
                "dataset": dataset_meta.get("dataset", step.dataset),
            },
        }
        return (rdf_chunk, chunk_info)


def simple_rdataframe_processor(data):
    """Simple processor that demonstrates basic RDataFrame operations."""
    rdf, chunk_info = data
    dataset_name = chunk_info.get("metadata", {}).get("dataset", "unknown")

    count = rdf.Count().GetValue()
    result = {
        "count": count,
        "dataset": dataset_name,
    }

    try:
        column_names = [str(b) for b in rdf.GetColumnNames()]
        if "pt" in column_names:
            hist_pt = rdf.Histo1D(("pt", "pt", 50, 0, 200), "pt")
            result["pt_mean"] = hist_pt.GetMean()
            result["pt_entries"] = hist_pt.GetEntries()
    except Exception:
        pass

    try:
        column_names = [str(b) for b in rdf.GetColumnNames()]
        if "mass" in column_names:
            hist_mass = rdf.Histo1D(("mass", "mass", 50, 0, 200), "mass")
            result["mass_mean"] = hist_mass.GetMean()
            result["mass_entries"] = hist_mass.GetEntries()
    except Exception:
        pass

    return result


def rdataframe_accumulator(a, b):
    """Accumulate results from two chunks."""
    result = {
        "count": a["count"] + b["count"],
        "dataset": a["dataset"],
    }

    if "pt_entries" in a and "pt_entries" in b:
        total_pt_entries = a["pt_entries"] + b["pt_entries"]
        if total_pt_entries > 0:
            result["pt_mean"] = (
                a["pt_mean"] * a["pt_entries"] + b["pt_mean"] * b["pt_entries"]
            ) / total_pt_entries
            result["pt_entries"] = total_pt_entries

    if "mass_entries" in a and "mass_entries" in b:
        total_mass_entries = a["mass_entries"] + b["mass_entries"]
        if total_mass_entries > 0:
            result["mass_mean"] = (
                a["mass_mean"] * a["mass_entries"] + b["mass_mean"] * b["mass_entries"]
            ) / total_mass_entries
            result["mass_entries"] = total_mass_entries

    return result


def make_source_instantiation(treepath="Events", **kwargs):
    return RDataFrameSource(treepath=treepath).source_instantiation


def main():
    port = 9123
    mgr = vine.Manager(port=port)

    filelist = {
        "ZJets": {
            "metadata": {},
            "files": [
                {
                    "file": osp.abspath("../samples/nano_dy.root"),
                    "numentries": 40,
                    "metadata": {
                        "dataset": "ZJets",
                        "object_path": "Events",
                        "checkusermeta": True,
                        "someusermeta": "hello",
                    },
                },
            ],
        },
        "Data": {
            "metadata": {},
            "files": [
                {
                    "file": osp.abspath("../samples/nano_dimuon.root"),
                    "numentries": 40,
                    "metadata": {
                        "dataset": "Data",
                        "object_path": "Events",
                        "checkusermeta": True,
                        "someusermeta2": "world",
                    },
                }
            ],
        },
    }

    rdf_source = RDataFrameSource(treepath="Events")
    distributor = TaskVineDistributor(
        manager=mgr,
        resources_processing={
            "cores": 1,
            "disk": 1024,
        },
        verbose=True,
    )
    run = VineReduce(
        distributor=distributor,
        data=filelist,
        processors={"proc": simple_rdataframe_processor},
        accumulator=rdataframe_accumulator,
        source_instantiation=rdf_source.source_instantiation,
        default_step_size=10000,
    )

    try:
        from ndcctools.taskvine import Factory

        workers = Factory(manager_host_port=f"localhost:{port}", batch_type="local")
        workers.min_workers = 1
        workers.max_workers = 1
        workers.cores = 2
        workers.disk = 4096

        with workers:
            results = run.compute()

        print("Results:")
        print(results)

    except (ImportError, RuntimeError, FileNotFoundError) as e:
        print(f"Error: {e}")
        print("Note: This example requires ROOT, TaskVine, and sample ROOT files.")


if __name__ == "__main__":
    main()
