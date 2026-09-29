# Terrain Models Under Tall Amazon Canopy — audit scripts and stamped artifacts

Reproducibility package for L. F. C. Seelig and R. M. Salles, "Terrain
Models Under Tall Amazon Canopy: Label and Reference Error Outweighs the
Choice of Estimator" (IEEE Journal of Selected Topics in Applied Earth
Observations and Remote Sensing).

This repository contains the audit scripts that produce every number cited
in the manuscript and the stamped JSON artifacts those scripts read and
write. The central piece is the fact sheet
(`audit/folha_de_fatos_jstars_v26.py` and its output
`audit/folha_de_fatos_jstars_v26.json`): every number in the text has an
entry in the fact sheet pointing to a stamped artifact and field; a number
without an entry does not exist. The fact sheet was built in eight
successive versions (`folha_de_fatos_jstars.py`, `_v2`, `_v21` to `_v26`),
each one copying the previous version's stamped output unchanged and adding
the facts of the checks introduced since; all eight are included so the
lineage can be followed end to end.

Raw rasters, training checkpoints and prediction arrays are not distributed
here; the scripts that require them document their expected paths and fail
loudly when an input is missing. The Data Availability statement in the
manuscript covers access to those inputs.

## Layout

- `audit/` — the fact-sheet lineage, the individual audit scripts and their
  stamped JSON outputs, and `audit/results/` (skill and evaluation outputs
  the fact sheet draws on)
- `audit/fig/` — the one surviving figure generator (`gerar_fig1.py`) and
  its numeric sidecar; a second sidecar (`fig9_numeros.json`) is kept for
  lineage completeness even though the panel it describes was cut from the
  submitted manuscript (see `PROVENANCE.md`)
- `audit/_pareceres_2026-09-28_execucao/` — two small reference-product
  artifacts (ANADEM sampling and a datum sanity check) read by the fact
  sheet; the folder keeps its original name because the artifacts point to
  it
- `PROVENANCE.md` — sha256 of every file in this repository, what differs
  from the copy in the working experiment tree, and every artifact whose
  generating script could not be recovered
- `verify_payload.py` and `PAYLOAD_SHA256` — the check that the published
  artifacts carry the same numbers as the originals (see below)
- `verify_code.py` and `CODE_AST_SHA256` — the check that the published
  scripts contain the same code as the originals (see below)
- `CITATION.cff` — citation metadata

## Licence

Everything here is released under the MIT licence; see `LICENSE`.

## Checking a number

Every fact in `audit/folha_de_fatos_jstars_v26.json` carries `valor`,
`unidade`, `artefato` and `campo`. To check a number quoted in the paper,
find its fact, open the named artifact under `audit/`, and read the named
field. Each stamped artifact also records, under `_fontes`, the sha256 of
every file it read, so the chain can be followed back to the training and
evaluation outputs in `audit/results/`.

To confirm that the files here are the ones described, and that the
published artifacts carry exactly the numbers of the originals, run at the
repository root (Python 3, standard library only):

    sha256sum -c SHA256SUMS
    python verify_payload.py
    python3.11 verify_code.py

## Re-running the scripts

The scripts (Python 3.11, numpy, scipy, pandas, xgboost) were written to run
at the root of the full experiment tree. The audit scripts read raster
inputs, per-cell prediction arrays and model checkpoints that are not
distributed here, and the fact-sheet builders also read the project's
internal design and review notes, so neither completes from this repository
alone; both fail loudly and name the missing input. They are included so
that the computation behind each number can be read line by line.

## Two things worth knowing before comparing hashes

The Python sources in this repository had internal working comments
removed before publication, and their comments, docstrings and the string
literals that cited internal review steps (notes written into the JSON
outputs, log lines, error messages) were rewritten in English. The
executable code is unchanged. `verify_code.py` proves it: it hashes each
script's syntax tree with docstrings dropped and free-text strings masked
(an f-string keeps its interpolated expressions, in order) and compares the
result with the digest of the original script in `CODE_AST_SHA256`. Names,
calls, operators, numbers, keys, labels and file paths all enter that
digest. Short strings that code reads or writes as keys, labels or paths
keep their original Portuguese names. The sha256 of each script differs
from the `sha256_script` recorded inside the stamps, which documents the
script as it stood when the artifact was produced; `PROVENANCE.md` lists
both.

The stamped JSON artifacts had their free-text notes translated into
English, with references to the project's internal review steps removed and
internal-note file names replaced by neutral ones (`internal/note_NN.md`).
Nothing else changed. `verify_payload.py` proves it: for each artifact it
hashes the canonical JSON with every free-text string masked, and compares
the result with the digest of the original artifact, recorded in
`PAYLOAD_SHA256` before translation. Every number, boolean, key, dotted field
pointer, fact id, artifact name, short label and sha256 enters that digest,
so a single changed value fails the check. Because the file bytes changed,
a sha256 recorded inside a stamp (`_fontes`) refers to the original file;
`PROVENANCE.md` maps each original sha256 to the published one. Keys, fact
ids and short verdict labels (e.g. `nao demonstrado`) stay in Portuguese
because code and pointers refer to them. A few artifacts carry no
`_proveniencia` record pointing to a generating script; `PROVENANCE.md`
lists them.
