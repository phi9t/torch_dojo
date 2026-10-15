# Library and spaced-review queue

Status: resolved

Implement the approved library + queue direction from the parent spec.

## Implementation

Shared content contains 16 motifs. The browser home page supports review,
search, reveal, rating, related workouts and archive transfer. Native Jupyter
has a starter notebook and a persistent ReviewSession with rich/text displays.
Python and TypeScript share scheduling fixtures and the version 1 archive
contract. Offline builds include the motif catalog.

## Validation

- Eight native review tests pass, including executable probes for all 16 motifs,
  real workout links, archive validation, due ordering, persistence and imports.
- 92 TypeScript tests pass; type checking passes. Shared fixtures exercise the
  same scheduler in Python and TypeScript.
- 98 browser Python checks pass. The updated offline build test confirms the
  shipped motif library matches the native content and appears in the manifest.
- Seven browser end-to-end tests pass: existing home flows plus motif recall,
  saved ratings, search, transfer, quota failures and an offline reload.
- The starter notebook executes in a fresh ipykernel inside the Jupyter rootfs,
  including HTML display, reveal, rating, saved progress and library browsing.
- Desktop and 390px layouts inspected; the narrow page has no horizontal overflow.

Earlier architecture validation is recorded in the learning-flow-architecture
issue. Review progress remains local; browser/native transfer is explicit.
