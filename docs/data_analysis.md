# RSNA Knee Abnormality Detection - data analysis

Source: our copy at `gpu-server2:/raid/sroziewski/data/rsna_kad/` (CRC-verified against the Kaggle zip,
manifest in `.download/manifest.tsv`). Every number below was produced by `analysis/eda.py`
(`scan` -> `analyze` -> `decode` -> `throughput` -> `analyze`); the raw fact dump is
`analysis/out/report_facts.txt` and each table is a CSV next to it. The full header table
(`headers.parquet`, 140 MB, one row per file) stays on the server under
`/raid/sroziewski/sandbox/rsna_kad/analysis/out/`.

## 1. train.csv and labels

- 4,407 rows = 4,407 unique StudyInstanceUID, no duplicates. Every row has an image directory and
  every image directory has a row (0 missing either way). test.csv / sample_submission: 3 studies.
- **Only 58 of 4,407 studies (1.3 %) carry labels.** Labels are all-or-nothing: 58 rows have all 12
  values, 4,349 have none. Values are exactly {0, 1}. All 58 labelled studies have images.
- No labelled study is all-negative; positives per labelled study range 1-9 (median 4).

| label | pos / 58 | prev. | label | pos / 58 | prev. |
|---|---|---|---|---|---|
| ACL | 24 | 0.41 | PF OA | 21 | 0.36 |
| MCL | 9 | 0.16 | Effusion | 35 | 0.60 |
| Medial Meniscus | 26 | 0.45 | Synovitis | 27 | 0.47 |
| Lateral Meniscus | 23 | 0.40 | Baker's | 12 | 0.21 |
| Medial OA | 15 | 0.26 | Contusion | 19 | 0.33 |
| Lateral OA | 11 | 0.19 | Fracture | 18 | 0.31 |

Co-occurrence (phi on the 58): two clusters - degenerative (Medial OA-Baker's 0.48, Medial
Meniscus-Medial OA 0.42, Lateral Meniscus-Lateral OA 0.42, Effusion-Synovitis 0.40) and traumatic
(ACL-Contusion 0.38, Contusion-Fracture 0.33, ACL-Fracture 0.27); OA labels are negatively correlated
with ACL/Contusion/Fracture (-0.13 to -0.33). Full matrices: `label_cooccurrence_counts.csv`,
`label_phi_correlation.csv`. With n=58 these are indicative only.

### Reports (all 4,407 rows have one; none empty)

- Length: median 977 chars / 129 words (p5 205 / 28, p95 2,453 / 330, max 4,743 / 690). 57 % (2,511)
  contain newlines (up to 72 lines) - hence the 58,556 physical lines in the CSV.
- Language (langdetect, seed 0): en 1,736 (39 %), es 682, tr 546, hr 406, el 321, de 262, bg 220,
  nl 153, fr 81. Labelled studies are spread proportionally (en 28, es 10, tr 6, hr 4, el 3, bg 3,
  de 2, nl 2, fr 0). "hr" may include Serbian/Bosnian; not verified by a human.
- De-identification placeholders: `[DATE]` in 710 reports, `[TIME]` 359, `[REDACTED]` 110, `[ID]` 94,
  `[Date]` 49, `[NAME]` 38, `[Age]` 17, `[YEAR]` 14; 714 rows contain some `[...]` token, 16 contain
  `<...>`. Only 8 rows still contain a literal d/m/y date. Structure words: "technique/tecnica" in 935
  rows, "findings" 859, "conclusion" 691, "impression" 676.
- Duplicates: 177 rows share their exact text with another row (4,276 distinct texts); 204 after
  lower-casing/whitespace normalisation. The largest cluster is a Turkish normal-knee template (37
  identical rows), then Spanish "Técnica: RMN de la rodilla. Resultados: ... Impresión: ..." one-liners
  (14, 12, 5, 5 rows) and an English "Technique: MRI of the knee. ACL normal. MCL normal..." template
  (6 + 5 rows). Short templated reports (p1 = 103 chars) are usually normal studies.
- Examples: 24 excerpts in `report_examples.csv` (not in the public repo; competition data).
- Naive keyword matching is not a label: among the 58 labelled, an ACL term appears in 58 % of
  ACL-positive and 53 % of ACL-negative reports (normal findings are stated explicitly); fracture terms
  44 % vs 20 %; "baker" 58 % vs 22 %. Negation-aware / LLM extraction is required.

## 2. Series (train_series.csv)

- 24,371 series over 4,407 studies; 5.53 series/study (3: 1 study, 4: 675, 5: 2,299, 6: 698, 7: 310,
  8: 145, 9: 176, 10: 74, 11-14: 29). CSV and disk agree exactly (24,371 = 24,371 pairs, 0 extra either
  side); every file's header StudyInstanceUID / SeriesInstanceUID / SOPInstanceUID equals its
  directory / filename. Same for test (15 series, 3 studies).
- **`Fluid_Sensitive` == `Fat_Suppression` in every row** (train and test), so there are only 6 types:

| plane / type | series | studies with >= 1 | share of studies |
|---|---|---|---|
| Sagittal FS=0 (T1/PD, no fat-sat) | 5,197 | 4,266 | 96.8 % |
| Axial FS=1 | 4,719 | 4,407 | 100 % |
| Sagittal FS=1 | 4,667 | 4,150 | 94.2 % |
| Coronal FS=1 | 4,624 | 4,248 | 96.4 % |
| Coronal FS=0 | 3,985 | 3,406 | 77.3 % |
| Axial FS=0 | 1,179 | 857 | 19.4 % |

  Every study has at least one series in each plane; 3,991 (90.6 %) have all three FS planes; 257
  studies lack a sagittal FS series and 159 lack a coronal FS series. 1,936 studies (44 %) have >= 2
  series of the same type (up to 6). The most common study composition (1,747 studies) is
  {Ax FS, Cor FS, Cor T1, Sag FS, Sag T1}.
- FS=1 series have TE median 33 ms / TR 3.2 s; FS=0 have TE 15 ms / TR 0.7 s (T1-like) with a long
  tail (p95 TE 89 ms: non-fat-sat T2/PD). ScanOptions contains "FS" in the vast majority of FS=1 series
  and "OTHER"/"PER"/blank in FS=0 (`scanoptions_vs_fatsup.csv`), i.e. the CSV flags are consistent with
  the headers.
- Slices per series: median 30, p5 18, p95 45, min 11, max 320 (2D: median 30, range 11-60; 3D: see
  below). Per type the median is 29-32 slices. 0 series have <= 3 files.

## 3. DICOM headers (all 819,635 files; 819,078 train, 557 test; 0 read errors, 0 zero-size)

- 83 distinct tags; 28 present in 100 % of files (incl. Rows/Columns, PixelSpacing, SliceThickness,
  IOP, IPP, InstanceNumber, SeriesNumber, FrameOfReferenceUID, WindowCenter/Width, PatientID,
  Manufacturer, ManufacturerModelName, ImageType). 86 distinct tag-set patterns. Full list with
  frequencies: `tag_presence.csv`. **Absent everywhere:** StudyDate/SeriesDate/any date or time,
  PatientAge, PatientBirthDate, AccessionNumber, StudyID, StudyDescription, ProtocolName,
  InstitutionName, StationName, DeviceSerialNumber, ImageLaterality, PixelPaddingValue,
  Smallest/LargestImagePixelValue, LossyImageCompression. No private tags survived.
- Pixel encoding is uniform: **TransferSyntax Explicit VR Little Endian (uncompressed) for 100 %**,
  MR Image Storage, MONOCHROME2, SamplesPerPixel 1, BitsAllocated 16, single frame. BitsStored 12 /
  unsigned for Siemens, Philips, Fujifilm, Hitachi (613,015 files); BitsStored 16 with signed
  PixelRepresentation=1 for GE, Canon, Toshiba (201,276 files); GEHC 16-bit unsigned (5,344).
  No pylibjpeg codecs are needed (they are installed anyway).
- Rows x Columns: 221 distinct shapes, 89.3 % square. Top: 512x512 (22.5 %), 640x640 (15.5 %),
  384x384 (10.2 %), 320x320 (6.8 %), 256x256 (5.3 %), 640x1280 (4.7 %, a Siemens 3D DESS "Patella"
  protocol), 1024x1024 (4.0 %). Range 160-1,280 rows, 160-1,444 columns. Median per series type is
  512x512 (Axial FS=0: 704x704). No series mixes shapes, spacing, thickness, transfer syntax, echo
  time or acquisition number internally.
- PixelSpacing: always isotropic in-plane; median 0.312 mm (p5 0.167, p95 0.625, range 0.073-1.17).
  In-plane FOV (rows x spacing) median 160 mm (p5 145, p95 190). SliceThickness median 3.0 mm
  (p5 0.6, p95 4.0, max 6). SpacingBetweenSlices absent in 84,055 files (1,585 series); the IPP step
  along the slice normal is available for all series (median 3.5 mm, p5 2.2, p95 5.0) and matches
  SpacingBetweenSlices within 0.1 mm in all but 1 series.
- Orientation: plane derived from ImageOrientationPatient agrees with the CSV `Anatomical_Plane` for
  **100 % of files (0 mismatching series)**. Obliquity from the nearest axis: median 6.5 deg, p90 15,
  p99 32, max 41.
- Slice ordering: InstanceNumber is present, unique and contiguous in every series (5 series start at
  a number other than 1); IPP is present and unique in every series; sorting by InstanceNumber is
  monotone along the slice normal in **all 24,386 series**. Either key works.
- MRAcquisitionType: 2D for 22,338 series, 3D for 837 (3.4 %; 50-320 slices, median 144; thickness
  0.6-3.2 mm, median 0.9), absent for 1,211 (Toshiba/Canon/old GE). All 809 series with > 60 slices
  are 3D. 3D series occur in Axial FS=1 (318), Sagittal FS=0 (269), Sagittal FS=1 (148), Axial FS=0
  (101); essentially none coronal.
- SeriesDescription: absent in 1,443 series and the literal string **"DummySeriesDesc!" in 2,879**
  (together 17.7 %); 777 distinct values otherwise (vendor protocol names such as
  `pd_tse_fs_sag_320`, `PDW_SPAIR`, `Sag PD FSE FS`, `COR T1`). SequenceName present in 47 %.
  The CSV flags are the only complete description of series type.
- Scanner / site (all site-identifying tags stripped): 12 Manufacturer strings (over the 4,410
  train+test studies: Siemens variants 1,952, GE 906, Philips 1,301, Toshiba/Canon 227,
  Fujifilm/Hitachi 24), 46 model names, field 1.5 T 2,545 / 3 T 1,601 / 1.16 T 24 / absent 239.
  Train-only key manu|model|field|software gives 112
  groups (largest "SIEMENS|Aera|1.5|syngo MR E11" 353 studies = 8 %); adding ImagingFrequency
  rounded to 0.01 MHz (a per-magnet fingerprint; 3,141 raw values, 34 rounded) gives 149 groups.
  Report language is almost a function of the scanner key (e.g. Avanto fit/XA61 -> 262/262 de;
  Ingenia/5.3.1.4 -> 220/220 bg; SonataVision -> 214/214 el; Aera/E11 -> 178 tr + 123 en + 52 hr),
  so scanner+language separates sites better than either alone:

| group key | groups | largest | largest % | groups < 10 studies | groups holding labelled | max labelled / group |
|---|---|---|---|---|---|---|
| Manufacturer | 12 | 1,053 | 23.9 | 1 | 8 | 16 |
| language | 9 | 1,736 | 39.4 | 0 | 8 | 28 |
| scanner_key | 112 | 353 | 8.0 | 51 | 33 | 5 |
| scanner_key + lang (`site_key`) | 123 | 262 | 5.9 | 56 | 35 | 5 |
| scanner_key + ImagingFreq | 149 | 249 | 5.7 | 67 | 40 | 4 |

  Per-study keys are in `study_summary.csv` (server) / `site_key_counts.csv`.
- PatientID: present in every file, 12-char de-identified hash, **one distinct ID per study
  (4,407 IDs for 4,407 train studies)**; the 3 test IDs do not occur in train. No patient-level
  leakage is detectable from headers. PatientSex M 2,077 / F 1,895 / O 199 / absent 239. No age.
- Laterality: consistently L in 1,087 studies, R in 1,091 (incl. 19 spelled LEFT/RIGHT), no usable
  value in 2,204 (tag empty in 1,220 studies, absent in 985); i.e. **only 49 % of studies say which
  knee it is.** 25 studies contain both L and R series (typically
  one series with the opposite tag, e.g. pattern L R R R R x10) - a tagging error, not bilateral
  exams. Some sites encode the side in SeriesDescription instead (`RT_`/`LT_`, trailing ` L`/` R`).
- BodyPartExamined: KNEE 3,410 studies, EXTREMITY 430, absent/blank 415, and header noise such as
  KNEE|WRIST 75, ANKLE 16, HEAD 10, BRAIN 6, LIVER 4 (plane check passes for those; images not eyeballed).
- 2 studies contain series from two different scanners (Siemens Aera + Philips Achieva dStream);
  same PatientID, `multi_manufacturer_studies.csv`.

## 4. Pixel data

- Decoded 602 files (50 per manufacturer string + the 2 truncated files below): 600 OK. Single-process
  cost: file read 0.9 ms median (p90 1.3), `pixel_array` 0.46 ms median (p90 0.66) - it is a memcpy.
- Intensities are raw vendor units, not comparable across (or within) vendors: median p99 per file
  ranges from 210 (Siemens) to 6,335 (Canon); 106/600 files exceed 4,095; Toshiba files are signed
  with negative values (min -1,276 in 50/50 sampled files); no file exceeds 2^BitsStored. Median
  distinct values per image 930 (Philips 12-bit as low as 526). RescaleSlope != 1 only on Philips
  (122,888 files, 3,944 distinct slopes such as 1.51 / 4.50 / 5.80, intercept always 0, RescaleType
  "normalized"); GE/Siemens have slope 1 or
  no rescale. WindowCenter/Width exist for every file but with 56k/75k distinct values (per-image).
- Throughput (`decode_throughput.csv`; 12,000 random files = 8.3 GB, page cache evicted with
  posix_fadvise before "cold"; /raid = 2x Samsung PM9A3 NVMe under LVM/XFS):

| procs | cache | files/s | MB/s | full train (819k files, 570 GB) |
|---|---|---|---|---|
| 24 | cold | 7,499 | 5,216 | ~1.8 min |
| 8 | cold | 4,244 | 2,952 | ~3.2 min |
| 1 | cold | 714 | 497 | ~19 min |
| 1 | warm | 1,160 | 807 | ~12 min |

  Reading + decoding is not the bottleneck; resizing/normalising will be. Header-only scan of all
  files took 78 s on 24 processes.

## 5. Public test set

3 studies / 15 series / 557 files: Siemens Avanto fit 1.5 T (5 series incl. a 160-slice 640x1280
3D DESS axial), Toshiba Vantage (5 series, 512-672 px), GE SIGNA Artist 1.5 T (5 series, 512x512).
All three have Ax FS=1, Cor FS=1 and Sag FS=0; the Toshiba and GE studies also have Sag FS=1, the
Siemens study has no sagittal FS series at all (two sagittal non-FS + the 3D axial). Same encoding
as train, `Fluid_Sensitive == Fat_Suppression`. They mirror train vendors and compositions; nothing
suggests a distribution shift, but 3 studies prove little about the hidden test.

## 6. Oddities

- **2 truncated files** (both single slices in 36/38-slice FS series of unlabelled studies; sizes
  match the Kaggle manifest, so the corruption is upstream): pixel payload is exactly half of
  Rows x Cols x 2 bytes and `pydicom` raises `ValueError` on `pixel_array`
  (`pixel_size_mismatch.csv`). Loaders must catch this and drop/zero the slice. No other size
  mismatch in 819,635 files; 12,600 files decoded without error.
- No series with 1 slice; no mixed shapes inside a series; no duplicate InstanceNumber or IPP.
- 1,585 series lack SpacingBetweenSlices; 1 series has an IPP step != SpacingBetweenSlices.
- "DummySeriesDesc!" / absent SeriesDescription (18 % of series); 3D series with up to 320 slices.
- 25 studies with contradictory Laterality tags; 50 % of studies without any laterality.
- BodyPartExamined noise (WRIST/HEAD/BRAIN/LIVER on knee-oriented series).
- 2 studies mixing two scanners; MagneticFieldStrength spelled "1.5" vs "1.500000", "3" vs "3.0".

## Implications for modelling

1. **Label source.** 58 gold-labelled studies cannot train a 12-label image model; the 4,349
   unlabelled studies must get labels from the reports (multilingual, 9 languages, negation-heavy,
   templated normals). Plan a report-to-label stage (LLM or multilingual NLI/classifier) and treat
   the 58 as a *validation/calibration* set for that stage and for the image model, not as training
   data. Expect label noise; prevalence in the 58 (Effusion 60 %, ACL 41 %) is likely enriched
   relative to the population (37+ duplicated "normal" templates exist), so do not calibrate priors
   on it. Metric is macro AUC, so rare-label ranking matters more than calibration.
2. **Series to use.** Every study has Axial FS; Sag FS and Cor FS are present in 94-96 %; Sag non-FS
   (T1/PD) in 97 %. A 3-input design (Sag FS, Cor FS, Ax FS, falling back to the same-plane non-FS
   series when FS is missing - 257 / 159 / 0 studies) covers everything; adding Sag FS=0 as a 4th
   input covers bone/marrow (Contusion/Fracture) and OA. Cor FS=0 (77 %) and Ax FS=0 (19 %) are
   optional. Select series by the CSV flags, not SeriesDescription. When a type occurs more than once
   (44 % of studies), prefer 2D over 3D and the one with the most slices, or use all.
3. **Geometry.** 2D stacks: ~30 slices x 3 mm, in-plane 0.31 mm / 160 mm FOV, 512x512 typical.
   Resampling every series to a fixed grid (e.g. 32 slices, 256-384 px, ~0.5 mm) loses little; 3D
   series (3.4 %) need slice subsampling (median 144 slices, 0.9 mm) or resampling to 3 mm. Sort by
   InstanceNumber (equivalent to IPP order everywhere). Plane can be trusted from the CSV.
4. **Intensity.** Per-image/per-series robust normalisation (percentile clip, z-score) is mandatory:
   raw ranges differ 30x between vendors, Toshiba is signed, Philips has a RescaleSlope that others
   lack. Do not use WindowCenter/Width or BitsStored as a global scale.
5. **Laterality matters for Medial/Lateral labels** and is missing for 50 % of studies (and
   contradictory in 25). Do not hard-code medial/lateral from headers; either infer the knee side from the images
   (fibula position in coronal/axial) as an auxiliary task, or let the network learn it end to end
   with flip augmentation restricted to left-right consistent transforms. IOP/IPP give patient
   L/R direction, so side + geometry would fix medial/lateral if side is known.
6. **CV grouping.** PatientID is unique per study, so patient leakage is not the risk; *site* shift
   is. Use `GroupKFold` on `site_key` (scanner_key + report language; 123 groups, largest 5.9 %),
   which also distributes the 58 labelled studies (max 5 per group). Stratify folds by language
   share since language ≈ site and drives the report-derived labels.
7. **Compute budget.** Full-resolution read of the train set is ~2 min on 24 cores from NVMe; a
   one-off preprocessing pass to a compact cache (e.g. uint8/float16 volumes at 256-384 px) is
   cheap and should be the design, with the two truncated slices handled. No GPU decode needed.
8. **Test time.** Hidden test likely follows the same 6-type composition; make the pipeline robust to
   missing series types, 3D-only inputs, and non-square images.

## Unknowns

- How the hidden test labels were produced (radiologist reading vs. report extraction) and their
  prevalence; the 58 gold rows are the only anchor.
- Whether the report-derived labels will match the annotator's definitions (e.g. "Medial OA" =
  cartilage loss vs. osteophytes; "Synovitis" vs. effusion).
- Language detection was automatic (langdetect); "hr" may mix Croatian/Serbian/Bosnian, and 81 "fr"
  rows were not inspected.
- Images were not visually inspected: BodyPartExamined=WRIST/HEAD studies are assumed to be knees
  because their orientation and series composition are knee-like.
- Only 12,600 files were fully decoded; corruption other than the two size-mismatch files is ruled
  out only by header/size checks.
- Whether a patient can appear under two PatientIDs (undetectable from the data).
