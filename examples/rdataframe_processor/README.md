# RDataFrame Processor Example

This example demonstrates how to use `VineReduce` with ROOT's `RDataFrame` instead of Coffea for processing ROOT files.

## Overview

The example shows:
- How to materialize Steps into RDataFrame chunks using `source_instantiation`
- How to process RDataFrame chunks with a custom processor
- How to accumulate results from multiple chunks

## Files

- `rdataframe_processor.py`: Basic example showing RDataFrame processing
- `example_with_preprocess.py`: Example that includes preprocessing to count events in ROOT files

## Requirements

- ROOT (with Python bindings)
- TaskVine (ndcctools)
- Sample ROOT files (e.g., `samples/nano_dy.root`, `samples/nano_dimuon.root`)

## Usage

### Basic Example

```bash
cd examples/rdataframe_processor
python rdataframe_processor.py
```

### With Preprocessing

```bash
cd examples/rdataframe_processor
python example_with_preprocess.py
```

## Key Components

### Source Instantiation

The `make_source_instantiation` function materializes a Step into an RDataFrame chunk:
- Creates an RDataFrame for the specified tree
- Applies a range filter to process only the chunk
- Returns the RDataFrame and chunk metadata

### Processor

The `simple_rdataframe_processor` function:
- Counts events in the chunk
- Creates histograms for common branches (if they exist)
- Returns a dictionary with results

### Accumulator

The `rdataframe_accumulator` function:
- Combines results from multiple chunks
- Accumulates event counts
- Computes weighted means for histogram statistics

## Customization

To customize for your own ROOT files:

1. **Adjust branch names**: Modify `simple_rdataframe_processor` to use your actual branch names
2. **Change chunk size**: Adjust `default_step_size` on `VineReduce`
3. **Modify processing**: Update `simple_rdataframe_processor` to perform your analysis
4. **Update accumulator**: Modify `rdataframe_accumulator` to combine your specific results

## Differences from Coffea Example

- Uses `VineReduce` directly instead of `CoffeaVineReduce`
- Uses ROOT's RDataFrame API instead of Coffea's NanoEvents
- Requires manual setup of `source_instantiation`
- Dataset name must be manually added to file metadata

## Notes

- Ensure ROOT is properly installed and accessible in your Python environment
- The dataset name should be included in each file's metadata
- The `num_entries` field should be set via preprocessing or manually
- RDataFrame operations are lazy - results are computed when `.GetValue()` is called
