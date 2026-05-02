# Vendored canonical schemas

These JSON Schema files are **a snapshot copy** of the canonical synth-parameter
ontology. The single source of truth lives in the sibling
`reverse-synth-research` repo (private), at:

```
reverse-synth-research/parameter-mapping/
  subtractive.schema.json
  synth-base.schema.json
```

We vendor them here so CI on `audio-analysis-mcp` (which cannot reach the
private sibling repo with the default `GITHUB_TOKEN`) still has the schema
available for `tone_generation` tests. CI sets:

```
TONE_GEN_SCHEMA_DIR=${{ github.workspace }}/schemas
```

Locally, `TONE_GEN_SCHEMA_DIR` is unset and the loader defaults to the
sibling-repo path (`../reverse-synth-research/parameter-mapping`), so dev
workflow reads the live source — there is no risk of editing the snapshot
and getting silent stale-schema bugs in dev.

## Sync process

When the canonical schema changes in `reverse-synth-research`, re-vendor:

```sh
cp ../reverse-synth-research/parameter-mapping/subtractive.schema.json \
   ../reverse-synth-research/parameter-mapping/synth-base.schema.json \
   schemas/
```

Then commit. Last sync: reverse-synth-research @ `4961d9d`.
