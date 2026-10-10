# Codebook

One row per field: what to extract from each paper.

| Type | Value |
|---|---|
| `category` | one of the listed values |
| `multi_category` | one or more values |
| `integer`, `number` | a reported number |
| `text` | a short answer, from a model |

## Search terms

Values are separated by `;` and terms by `,`:

```text
group: randomly assigned, control group; single_case: multiple baseline
```

## Excel

Import and Export read and write the same table as an Excel sheet.

## Status

`extracted` (found), `review` (a coder decides), `not_found` (not in the paper), or `refused` (the license check stopped the model).
