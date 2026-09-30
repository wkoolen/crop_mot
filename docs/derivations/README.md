# Derivations (mirrors of the thesis vault)

Every mathematical body in `crop_mot/` must be traceable to one of these notes. They are
**generated mirrors** of the author's Obsidian vault notes, converted so a text-only
reader can use them: frontmatter removed, `[[wikilinks]]` flattened to plain text, image
embeds replaced by a visible `[figure omitted in mirror: ...]` line. The LaTeX is
untouched, so an expression here is character-for-character the one in the thesis.

| Tag | Title | Mirror | Backs |
|---|---|---|---|
| A0 | Measurement modelling and handling clutter | [A0.md](A0.md) | `sensor/` — measurement model, `c(z)`, clutter sampling |
| A1 | Bayes to Kalman derivation | [A1.md](A1.md) | `filters/kalman.py` |
| A2 | Bernoulli existence update | [A2.md](A2.md) | `filters/bernoulli.py`, `analysis/analytic.py` |
| A3 | Normalisation and pruning across competing global hypotheses | [A3.md](A3.md) | phase 2 (PMB/PMBM) |

## Rules

1. **Do not edit a mirror.** They are regenerated and your edit would be lost. Edit the
   vault note, then re-sync (below).
2. **Cite in code as `[A2 §2]`** — tag plus section — in the docstring or on the line that
   implements the expression. `tests/test_derivations_in_sync.py` fails if a tag names a
   derivation that does not exist.
3. **If code and mirror disagree, the vault note wins.** Stop and ask the author rather
   than changing either one.
4. **Figures are missing on purpose.** A0 and A3 embed images that hold part of the
   argument. If a `[figure omitted]` line sits exactly where you need the expression, ask
   the author for it in text; do not reconstruct it.

## Keeping them level

Inside the container (no vault access, runs in CI and in pytest):

```bash
python3 -m pytest tests/test_derivations_in_sync.py -q
```

In WSL, where `/mnt/c` is visible — this is the half that notices the author changed a
derivation:

```bash
python3 scripts/sync_derivations.py --check --vault "/mnt/c/Users/<you>/OneDrive/Documenten/9_Vault_backup/Obsidian/University/Thesis/2_Project/2_2_Derivations"
```

It reports each derivation that has changed since it was mirrored, **together with the
code files that cite it** — that list is the re-reading you owe before running
`--update`. Run the check after any derivation session, and before writing thesis text
from the code.

## Derivation → code map

Which sections of which note back which functions. The sync report uses the table above
for file-level granularity; this is the finer-grained version, and it is also the list of
places to re-read when a section changes.

| Derivation section | Implemented in | Note |
|---|---|---|
| A0 §Measurement model | `sensor/sensor_model.py::clutter_density`, `sensor/detector.py::sample_scan` | clutter intensity `lambda_FA * c(z)`, `c(z)` uniform over the FOV area (D5) |
| A1 §Result (Kalman filter update) | `filters/kalman.py::kf_update` | Joseph form, see the docstring |
| A2 §2 Misdetection update | `filters/bernoulli.py::BernoulliFilter._update_existing` (no gated detection); `analysis/analytic.py::BernoulliExistenceReference.r_sequence` (miss case) | `r+ = r(1 - p_D) / (1 - r p_D)`; `p_D` at the predicted mean stands in for `p_D_bar` [A2 §2.1] |
| A2 §3.1 What r+ = 1 is | `BernoulliFilter._update_existing` (gated detections); `BernoulliFilter.extract`; `r_sequence` (detection cases) | the detected and missed rows averaged into `r_marg`, which `extract` reports. Two or more detections combine it with A0 §Measurement model (D2), not yet its own section. The density is collapsed by `filters/collapse.py::KeepBestBranch` instead of kept as the mixture (D1) |
| A2 §4 Birth from a measurement | `filters/birth.py::SingleFromMeasurement`; `BernoulliFilter.update` (after the update, D3); `r_sequence` (birth step) | `r_b` is a configured constant standing in for `e / (e + lambda_FA(z))` |
| A2 §5 Cromwell's rule | `filters/bernoulli_bank.py::BernoulliBankFilter.predict` | pruning below `r_min` (D13); `r_sequence` has no deletion, so B3 reads the unpruned log (D14) |
