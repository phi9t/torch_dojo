# Patterns worth keeping

A motif is an idea you can reuse across workouts. The library starts with 16:
scatter collisions, group totals, weighted means, compaction, flat addresses,
unique inverse maps, gather gradients, broadcasting, safe masking, stable
normalisation, coordinate transforms, half-open bounds, layouts, parameter
snapshots, gradient paths, and merging attention blocks.

Each recall card gives you a prediction, an explanation, a runnable NumPy probe,
and related workouts. The probes work in both the browser and native kernels.
Some linked workouts require PyTorch and are available only in the native flow.

## Browser

The gym home page has a **Patterns worth keeping** section. **Review next motif**
selects the oldest due card, then a new card in library order. Make a prediction
before you reveal the explanation. Copy the probe into a workout scratch cell
if you want to try it. Browse the library by idea, topic, or workout number.

Your rating schedules another visit:

| Rating | Next visit |
| --- | --- |
| Again | In 10 minutes |
| With effort | In 1 day |
| Comfortable | In 3 days initially; then twice the previous interval, up to 90 days |

These are simple scheduling rules. They do not estimate mastery. There is no
streak to maintain, and you can stop after any card. Browsing lets you practise
a card early; rating early practice also updates its schedule. Reviews do not
change workout reps, timers, or checks.

## Jupyter and the rootfs

Open [motif_review.ipynb](../../motif_review.ipynb) in the repository's Python
kernel. The same flow also works in Python cells in Emacs/Jupyter:

```python
from torch_judge.review import ReviewSession
review = ReviewSession()
review.next()
```

After predicting, `review.reveal()` shows the explanation and probe. Use
`review.rate('again')`, `review.rate('hard')`, or `review.rate('good')` to save
your rating. Then `review.next()` opens another card. `review.library()` lists
motifs and the paths to their workout scripts. `review.next('scatter-collisions')`
opens a specific card for extra practice.

Probes are displayed as code. Run them in a scratch cell when you choose;
revealing a card never executes its probe.

## Keep your progress

The browser saves reviews in this site's local storage. The native flow saves
`~/.local/share/torch-dojo/motif-review.json`, inside the persistent role home
when you use the rootfs. A custom path can be passed to `ReviewSession(path)`.
Closing the page or kernel preserves ratings. Unrated cards remain available.

Browser and native progress are separate until you transfer an archive. The
browser has **Export motif reviews** and **Import motif reviews** buttons.
The native equivalents are:

```python
review.export('motif-reviews.json')
review.import_archive('motif-reviews.json')
```

Import validates the entire file before saving and keeps the most recent
review for each card. Unknown card IDs and unsupported archive versions are
rejected. Corrupt stored data is reported rather than silently replaced.
Browser rep archives and motif archives are separate; export both if you want
to preserve both kinds of practice history. Clearing site storage removes
browser reviews unless you have exported them.

## Add a motif

The shared content is in `torch_judge/motifs.json`. Keep IDs stable because
review archives refer to them. Each card needs a concrete prediction, an answer
that explains the mechanism, a short NumPy probe, a transfer cue, and existing
four-digit workout IDs. Browser builds copy this library into their offline
assets. The native package includes the same file.
