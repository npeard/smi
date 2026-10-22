# Data Storage Format Evaluation

Evaluation of on-disk storage formats for self-interferometry acquisition data.
This is an **evaluate-and-recommend** study; it did **not** change the existing
data pipeline. The conclusion (keep HDF5) has been adopted, so the one-off
benchmark script and its dependencies (pyarrow/zarr/duckdb/hdf5plugin) were
removed afterward. The benchmark code is preserved in git history if it ever
needs to be re-run on different hardware or data.

## 1. The workload

Each acquisition file currently holds four datasets -- `RP1_CH1`, `RP1_CH2`,
`RP2_CH1`, `RP2_CH2` -- each of shape `(N_shots, 16384)` float32. One row across
all four channels is one acquisition "shot". A single shot is
`16384 * 4 bytes = 64 KiB` per channel (256 KiB across all four).

Two distinct access patterns matter:

1. **Training (random single-row reads).** The PyTorch `DataLoader` reads ONE
   shot per `__getitem__`. With shuffling, this is effectively uniform random
   single-row access. Per-row latency dominates throughput here.
2. **EDA / full-file analysis.** Polars/pandas exploration and full-array loads
   for plotting, statistics, and model preprocessing. Bulk full-load time and
   on-disk size dominate here.

The current baseline is HDF5 (h5py), described as gzip level 4 + chunked. (Note:
the specific sample file `circuit-noise-600.h5` happens to be stored
contiguous/uncompressed; the benchmark builds the gzip-4 chunked baseline itself
from the loaded subset so the comparison is apples-to-apples.)

## 2. Candidate matrix

| Format | Read/write lib (this env) | Columnar vs row fit for `(N,16384)` f32 | Compression | Random-row access | Polars EDA ergonomics |
|---|---|---|---|---|---|
| **HDF5** (baseline + tuned) | `h5py` 3.16, `hdf5plugin` 6.0 | Native N-D array store; one dataset per channel; rows = hyperslabs. Excellent fit. | gzip 0-9, lzf (built in), szip, Blosc/LZ4/Zstd via hdf5plugin | Excellent -- chunk `(1, 16384)` => one chunk per shot, decode exactly one shot per read | Indirect: read to NumPy, then `pl.DataFrame`. No native Polars reader. |
| **Parquet** | `pyarrow` 24.0; Polars `read_parquet` | Columnar, row-group oriented. A 16384-wide row stored as a `list<float32>` cell. Built for column scans, not array slices. | snappy (default), gzip, zstd, lz4, brotli | Poor -- no row index; a single logical row requires decoding its whole row group (data "can't be directly mapped from disk"). | Excellent -- first-class `pl.read_parquet`, predicate/projection pushdown, lazy scan. |
| **Arrow / Feather IPC** | `pyarrow` 24.0 (`feather`); Polars `read_ipc` | Columnar Arrow buffers on disk; memory-mappable. Same list-column shape as Parquet. | uncompressed, lz4, zstd (V2) | Moderate -- mmap + zero-copy slice avoids full decompress when uncompressed; compressed defeats zero-copy. | Excellent -- `pl.read_ipc` / `scan_ipc`, zero-copy from Arrow. |
| **Zarr** | `zarr` 3.2, `numcodecs` 0.16 | Native chunked N-D array store (like HDF5 but directory/object-store backed). Excellent fit. | Blosc (zstd/lz4 + shuffle, default zstd), gzip, zstd, others via numcodecs | Excellent -- chunk `(1, 16384)` => one chunk per shot. | Indirect: read to NumPy then `pl.DataFrame`. No native Polars reader. |
| **NumPy `.npy`** (mmap) | `numpy` 2.4 | One contiguous array per channel. Excellent fit; `mmap_mode='r'` gives true random row access. | None (raw) | Excellent -- memory-mapped, OS pages in only the requested row. | Indirect: load array then `pl.DataFrame`. |
| **NumPy `.npz`** | `numpy` 2.4 | Zip of `.npy` members. | None or per-member deflate | Poor -- zip member must be fully extracted/decompressed per access; no intra-array random access. | Indirect. |
| **SQLite (BLOB)** | `sqlite3` (stdlib) | Row store; each shot stored as a 64 KiB BLOB keyed by `(shot, channel)`. | None (could app-compress blobs) | Good -- indexed `WHERE shot=?` fetches just that row's blobs. | Poor for arrays -- Polars can read tables but BLOBs need manual `np.frombuffer` decode. |
| **DuckDB** (native + over-Parquet) | `duckdb` 1.5 | OLAP columnar engine. `FLOAT[]` list column per channel, or query Parquet directly. | zstd/snappy/lz4 (Parquet), internal columnar compression | Poor for single rows -- analytical engine optimized for column scans + projection/filter pushdown, not point lookups. | Excellent -- SQL over Parquet, integrates with Polars/Arrow zero-copy. |

Official documentation consulted (see Section 6 for URLs): h5py datasets,
pyarrow Parquet & Feather, Zarr arrays, NumPy save/load/memmap, DuckDB Parquet,
SQLite internal-vs-external BLOB.

## 3. Benchmark results (measured)

Subset: first 200 shots x 4 channels from `circuit-noise-600.h5`
(`52.4 MB` raw float32 in memory). Full-load = best of 3; random-row metric =
mean over 50 uniform random shot indices, reusing one open handle/connection.
Hardware: the developer's macOS machine; treat absolute numbers as relative,
not portable. Sorted by random-row read time (the metric that dominates training
throughput).

| format | write_s | size_MB | ratio | load_s | row_ms |
|---|---:|---:|---:|---:|---:|
| sqlite (BLOB rows) | 0.165 | 54.1 | 0.97x | 0.019 | **0.370** |
| hdf5 (uncompressed, chunked) | 0.092 | 52.5 | 1.00x | 0.050 | 1.446 |
| npy (mmap, uncompressed) | 0.185 | 52.4 | 1.00x | 0.010 | 1.523 |
| hdf5 (lzf, chunked) | 0.217 | 25.7 | 2.04x | 0.125 | 1.870 |
| hdf5 (blosc-zstd+shuffle, chunked) | 1.608 | 25.1 | 2.09x | 0.132 | 1.943 |
| feather (uncompressed, arrow IPC) | 0.100 | 52.4 | 1.00x | 0.024 | 2.123 |
| hdf5 (gzip-9, chunked) | 5.900 | 14.3 | 3.67x | 0.220 | 2.430 |
| **hdf5 (gzip-4, chunked) [baseline]** | 0.981 | 15.0 | 3.50x | 0.240 | 2.484 |
| zarr (blosc-zstd, per-shot chunks) | 1.790 | 25.0 | 2.09x | 0.513 | 9.849 |
| feather (lz4, arrow IPC) | 0.086 | 26.1 | 2.01x | 0.035 | 19.140 |
| feather (zstd, arrow IPC) | 0.129 | 18.4 | 2.85x | 0.042 | 37.938 |
| npz (uncompressed) | 0.127 | 52.4 | 1.00x | 0.043 | 40.537 |
| duckdb (native) | 1.554 | 25.7 | 2.04x | 3.272 | 52.311 |
| parquet (snappy, pyarrow) | 0.558 | 15.3 | 3.43x | 0.079 | 64.926 |
| parquet (zstd, pyarrow) | 0.581 | 14.8 | 3.53x | 0.086 | 69.992 |
| npz (compressed) | 2.640 | 13.9 | 3.78x | 0.175 | 180.980 |
| duckdb (over parquet) | 0.576 | 14.8 | 3.53x | 3.530 | 181.427 |

`write_s` = serialize time; `size_MB` = on-disk bytes; `ratio` = raw/disk
compression; `load_s` = full read of all channels; `row_ms` = mean ms per random
single-shot read across all channels.

## 4. Analysis: training-load vs EDA vs disk-size tradeoffs

**Training (random single-row reads dominate).** The PyTorch `DataLoader` with
shuffling pulls one random shot per `__getitem__`, so `row_ms` is the metric that
governs training throughput. The per-shot-chunked array stores (HDF5, npy-mmap)
win decisively at 1.4-2.5 ms because chunk shape `(1, 16384)` means one chunk
decodes exactly one shot. The columnar/row-group formats are 10-70x slower
(Parquet 65-70 ms, Feather-zstd 38 ms, DuckDB 52-181 ms): a single logical row
forces decoding its whole row group. This is the predicted columnar penalty
(Section 2), now measured. npz is also poor (40-181 ms) -- every read must reopen
the zip and decompress the full channel array.

**Disk size.** Deflate-family compression on this signal data is strong: the
baseline gzip-4 reaches 3.50x, and gzip-9 only nudges that to 3.67x while costing
6x more write time (5.9 s vs 1.0 s). Blosc-zstd+shuffle and lzf only reach ~2.0x
here -- this particular data compresses much better with deflate than with
byte-shuffle+LZ. So the baseline's choice of gzip is well-matched to the data.

**EDA.** Parquet/Feather are first-class in Polars (lazy scan, predicate/projection
pushdown) and are the ergonomic choice for tabular exploration -- but that strength
does not transfer to the array-slice access pattern of training. They are best
positioned as an *export* format for EDA, derived on demand from the HDF5 store,
not as the primary store.

**Combined, read both ways:** HDF5 gzip-4 is the only candidate that is
simultaneously near-best on compression (3.50x) and fast on the dominant
random-row read (2.48 ms), with a cheap full-load (0.24 s). Nothing else matches
both axes: sqlite/npy are fast but uncompressed; parquet/npz compress well but are
an order of magnitude slower per shot.

## 5. Verdict on SQL (SQLite / DuckDB): overkill?

**Yes -- SQL is the wrong tool here.**

- **DuckDB** is an OLAP engine built for column scans with projection/filter
  pushdown. On this workload it is the worst full-loader (3.3-3.5 s, ~14x the
  HDF5 baseline) and among the slowest per-shot readers (52-181 ms). Point lookups
  of whole-array rows are exactly what it is not optimized for. It adds a query
  engine, a dependency, and a query language for zero benefit over a chunked array
  store.
- **SQLite** with one 64 KiB BLOB per (shot, channel) is actually the *fastest*
  random-row reader (0.37 ms, indexed primary-key fetch) -- but it offers no
  compression (0.97x, slightly larger than raw), and arrays must be hand-decoded
  via `np.frombuffer`, so it has poor EDA ergonomics and no Polars story.

Neither earns its complexity. A relational/analytical database makes sense when
you need joins, ad-hoc SQL, or selective column scans over many heterogeneous
fields -- none of which describe "load one fixed-width float32 waveform per shot."

## 6. Recommendation

**Keep HDF5, chunked per shot `(1, n_samples)`, with gzip-4 as the default
compression.** The current baseline is already near-optimal: it ties for best
compression class (3.50x) while staying fast on the random single-shot reads that
dominate training (2.48 ms) and cheap to fully load (0.24 s). No migration is
warranted.

Optional, situational tweaks (not required):

- If acquisition-time *write* speed ever becomes a bottleneck, `lzf` cuts write
  time ~5x and reads are marginally faster (1.87 ms), at the cost of compression
  (2.04x vs 3.50x -- ~1.7x more disk).
- Do **not** raise gzip to level 9: +0.17x ratio for 6x write cost.
- For Polars-based EDA, export selected shots to Parquet/Feather on demand; do not
  adopt them as the primary training store (10-28x slower per-shot reads).
- The benchmark script was removed after this decision; recover it from git
  history (and re-add pyarrow/zarr/duckdb/hdf5plugin) if the data
  characteristics or access pattern change enough to warrant re-evaluation.

## 7. Official documentation cited

- HDF5 / h5py datasets, chunking & compression filters (gzip/lzf/szip,
  `compression_opts`, chunk-shape guidance, whole-chunk read on access):
  https://docs.h5py.org/en/stable/high/dataset.html
- hdf5plugin (Blosc/LZ4/Zstd/ZFP HDF5 filters):
  https://docs.h5py.org/en/stable/high/dataset.html (custom filters section)
- Apache Parquet via pyarrow (`read_table`/`write_table`, codecs, row groups,
  "data can't be directly mapped from disk"):
  https://arrow.apache.org/docs/python/parquet.html
- Apache Arrow / Feather IPC via pyarrow (`write_feather`/`read_feather`, LZ4/
  ZSTD/uncompressed, V2 == Arrow IPC file):
  https://arrow.apache.org/docs/python/feather.html
- Zarr arrays (chunking, Blosc/Zstd/gzip codecs, fancy/block indexing):
  https://zarr.readthedocs.io/en/stable/user-guide/arrays/
- NumPy format spec and memory-mapping:
  https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html ,
  https://numpy.org/doc/stable/reference/generated/numpy.load.html ,
  https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
- DuckDB reading Parquet (OLAP positioning, projection/filter pushdown):
  https://duckdb.org/docs/current/data/parquet/overview.html
- SQLite internal vs external BLOB storage (35% faster in-DB for ~10 KiB; cross
  over near 100 KiB):
  https://www.sqlite.org/intern-v-extern-blob.html
