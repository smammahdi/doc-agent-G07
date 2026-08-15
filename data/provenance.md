# Corpus provenance

## Source

- Corpus: R. V. Pierce, *The People's Common Sense Medical Adviser in
  Plain English, or, Medicine Simplified* (1890).
- Canonical record: https://archive.org/details/peoplescommonsen00pier
- Contributor: University of Massachusetts Medical School, Lamar Soutter
  Library.
- Usage rights: public domain. The Internet Archive record marks the item
  `NOT_IN_COPYRIGHT`.
- Project scope: this Pierce volume only. Gray's *Anatomy of the Human Body*
  was considered during early planning but is not part of this corpus.

## Reproducibility record

- Expected file: `data/raw/pierce-peoples-common-sense-medical-adviser-1890.pdf`
- Extracted PDF pages: 1,034.
- Formal A1 usable-word count: 354,367.
- File size: 65,311,598 bytes.
- SHA-256: `841b1feb55ff0aff5735c3aeb308eb52e217f91ae55c5d34e21feb6a640c8896`.
- The Internet Archive catalogue reports 1,050 scan images; the distributed
  PDF used by this project yields 1,034 pages after covers and scan inserts are
  resolved by the PDF export.

## Corpus difficulty and split policy

- Declared data speciality: dirty OCR caused by period typography, archaic
  spellings, uneven/faint ink, display typefaces, and figure-adjacent text.
- Measured pages are predominantly single-column; multi-column layout and a
  corpus-wide foxing gradient are not claimed as defining properties.
- The complete book is indexed once (1,034 total pages, 409,102 words). Chapter
  boundaries define the partitions used for transcription labels, tuning questions,
  and held-out evaluation so nearby pages from one topic cannot leak across splits.
- Evaluation uses 24 hand-verified held-out ground truth page transcriptions
  (`grading_kit/labels.jsonl`, `p0024`–`p0047`), achieving Macro Word-F1 = 0.9592,
  Micro CER = 0.1334, and Micro WER = 0.1840.

The corpus is a historical source, not current medical guidance. Answers must
remain grounded in the book, cite the source page, and avoid presenting the
1890 text as modern clinical advice.
