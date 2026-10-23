# OAI (NDA) as extra training data: research notes

Researched 2026-10-12. Every fact has a source. **UNVERIFIED** marks a claim I could not confirm from a primary source.
No account was registered and no OAI data was downloaded. I did read the public 3DReasonKnee CSVs on Hugging Face (CC BY 4.0), which are derived from OAI MOAKS tables.

Key sources:
- [NDA-CMS] Text of the NDA OAI pages, pulled from NDA's public Strapi CMS API that backs nda.nih.gov/oai. Some of these pages are login-gated on the website, but their text comes back from this API without login: `https://strapi.nimhda.org/api/static-contents?filters[urls][$containsi]=/oai/&populate=*`. The pages are /oai/accessing_images, /oai/querying_oai_data_in_nda, /oai/query_data_and_documentation, /oai/study_documentation, /oai/full_downloads, /oai/image-acquisitions, /nda/creating-an-nda-account and /nda/access-data-info.
- [NDATOOLS] Source of nda-tools at GitHub main, version 0.5.0: https://github.com/NDAR/nda-tools (`NDATools/__init__.py`, `Configuration.py`, `clientscripts/downloadcmd.py`, `README.md`, `CHANGELOG.md`).
- [IA-OVERVIEW] OAI "Overview and Description of Central Image Assessments" (12/1/2016): https://nda.nih.gov/static/docs/ImageAssessmentDataOverview.pdf
- [3DRK] 3DReasonKnee on Hugging Face: https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee (README, `Supplementary.pdf`, `code/data_preprocessing/dcm2nifti.py`, `data/labels/*.csv`, `data/split_files/*.csv`)
- [MOAKS] Hunter DJ et al. MOAKS. Osteoarthritis Cartilage 2011;19(8):990-1002: https://pmc.ncbi.nlm.nih.gov/articles/PMC4058435/
- [PETERFY] Peterfy CG et al. OAI MRI protocol design rationale. Osteoarthritis Cartilage 2008: https://pmc.ncbi.nlm.nih.gov/articles/PMC3048821/

---

## 1. Downloading OAI from NDA on a headless Linux server

### 1.1 Account and access steps for a new user
1. Get a RAS identity: Login.gov, eRA Commons, or an NIH PIV card. Without one, "create a Login.gov account using your institutional email". [NDA-CMS /nda/creating-an-nda-account; https://nda.nih.gov/nda/creating-an-nda-account]
2. Go to https://nda.nih.gov, click LOGIN, then "Sign in with RAS". Fill in the required fields and click "Create New NDA account". [same]
3. Request OAI access. Log in, open Data Permissions (https://nda.nih.gov/user/dashboard/data_permissions), and request the "Osteoarthritis Initiative" permission group. The Data Use Agreement has to be completed first. [OAI Knowledgebase FAQ, https://www.oaiknowledgebase.com/about/FAQs/ (403 to my fetcher, so this comes from the search snippet); NYU data catalog, https://datacatalog.med.nyu.edu/dataset/10162: "create a NDA account or link an existing NDA account, then agree to OAI data access terms."]
4. Approval timing:
   - NDA says "Open Access requests are approved upon submission." Broad Use and Controlled Access requests need an eRA Commons PI role, a Signing Official and DAC review. [NDA-CMS /nda/access-data-info; https://nda.nih.gov/nda/access-data-info]
   - **UNVERIFIED:** that OAI is an Open Access permission group, which would make approval instant. It fits with re3data listing the restriction type as "registration" (https://www.re3data.org/repository/r3d100011843) and with the NYU catalog wording ("agree to terms"). No source states it outright.
   - Access lasts one year and can be renewed. When it ends, NDA requires a closeout and destruction of the data. [NDA-CMS /nda/access-data-info]
   - Biospecimen requests are a separate NIAMS process (6-10 weeks). They are not needed here. [NYU catalog]
5. Set a separate password for nda-tools. "All NDA Tools now use RAS login, except for the Python nda-tools client, which does not currently support RAS. ... The username listed in your NDA Profile is the username used to log into the Python client. To set a password ... click 'Update Password' in your NDA Profile." [NDA-CMS /nda/creating-an-nda-account] Profile page: https://nda.nih.gov/user/dashboard/profile

### 1.2 Install
- `pip install nda-tools`. Current GitHub main is version 0.5.0. [NDATOOLS `__init__.py`]
- The client calls `check_version()` against PyPI and runs `sys.exit(1)` if the installed version is older than the newest release. Install the latest version. [NDATOOLS `__init__.py`]
- It imports `pkg_resources`, so setuptools must be installed. The README fix for a pkg_resources error is `python -m pip install setuptools`. [NDATOOLS README]

### 1.3 Credentials (from the source code)
- Username: `-u/--username`, or `[User] username` in `~/.NDATools/settings.cfg`. If neither is set, it asks interactively. [NDATOOLS `Configuration.py`, `__init__.py`]
- Password: there is no password flag and no environment variable. `--password` was removed in 0.2.26 [CHANGELOG]. The client calls `keyring.get_password('nda-tools', username)`. If that fails or finds nothing, it falls back to `getpass.getpass('Enter your NDA account password:')`. After a successful login it tries `keyring.set_password(...)`. [NDATOOLS `__init__.py` `_get_password`, `_try_save_password_keyring`]
  - Result: the first run needs a TTY (a screen session works), unless the password is already in a keyring backend.
- Pre-seeding the keyring, per the README: `python -c "import keyring; keyring.set_password('nda-tools', 'USER', 'PASS')"`. [NDATOOLS README]
- Headless keyring options:
  - README: "If there is no backend set up for keyring, try: `pip install secretstorage --upgrade keyrings.alt`". [NDATOOLS README]
  - HCP guide: `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`, then type the password at the prompt. [https://wiki.humanconnectome.org/docs/How%20to%20get%20data%20from%20the%20NDA%20using%20command%20line%20tools.html]
  - With `keyrings.alt` you can select a file backend such as `PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring`. It writes under the XDG data dir (`$XDG_DATA_HOME/python_keyring/`, default `~/.local/share/python_keyring`). [keyring/keyrings.alt docs: https://pypi.org/project/keyring/, https://pypi.org/project/keyrings.alt/ — **UNVERIFIED** for the exact path, which comes from library defaults and was not tested]
- Redirecting the home directory:
  - Every nda-tools path is built from `os.path.expanduser('~')`. That covers `~/.NDATools/settings.cfg`, `~/.NDATools/logging.yml`, `~/NDA/nda-tools/downloadcmd/packages/<id>` and `~/NDA/nda-tools/downloadcmd/logs`. [NDATOOLS `__init__.py` lines defining `NDA_TOOLS_SETTINGS_FOLDER` and `NDA_ORGINIZATION_ROOT_FOLDER`]
  - On Linux, `expanduser('~')` reads `$HOME`, so `HOME=/sandbox/ndahome downloadcmd ...` moves all of these. The folders are created on first run.
  - There is no dedicated NDA_* environment variable.
  - `-d <dir>` changes the download location and `--log-dir` changes the logs location, but settings.cfg is still written under `$HOME/.NDATools`.
  - Also set `XDG_DATA_HOME` and `XDG_CONFIG_HOME` inside the sandbox if you use a file keyring.

### 1.4 downloadcmd options (verbatim from `clientscripts/downloadcmd.py`)
- `-dp/--package <id>` (required). Downloads the whole package unless a filter is given.
- Positional `<S3_path_list>`: download only the listed `s3://...` files from the package.
- `-t/--txt <file>`: a text file of S3 links to download. This is how you select specific files.
- `-ds/--datastructure image03`: download only the files belonging to that data structure.
- `--file-regex <re>`: subset by regex. Works with `-dp`, `-ds` and `-t`.
- `-d/--directory <dir>`: download root.
- `-wt/--workerThreads N`: parallel downloads.
- `-s3 <bucket>`: copy straight to your own S3 bucket.
- `--verify`: starts no download. It writes `download-verification-report.csv` (package_file_id, expected location, nda_s3_url, exists, expected_size, actual_file_size) and `download-verification-retry-s3-links.csv`, which can be fed back with `-t`. It needs credentials and a package, but it gives you a full file list with sizes before you download anything.
- The package metadata file is `package_file_metadata_<id>.txt` in the packages folder. [NDATOOLS `__init__.py`]
- HCP pattern for subsetting: take the S3 links from the package manifest (`datastructure_manifest.txt`, column 6), filter them, then run `downloadcmd -dp <pkg> -t links.txt -d outdir -wt 8`. [HCP wiki above]

### 1.5 OAI image packages
- Pre-packaged images by visit, in zip format, are listed on the "OAI Image Dashboard": https://nda.nih.gov/user/dashboard/packages.html?dataset=Osteoarthritis+Initiative+%28OAI%29&type=shared_packages. Select "Shared Packages" and add one to your account with the Actions menu. [NDA-CMS /oai/accessing_images]
- Package sizes from that page:

  | Package | Zip size | Unzipped total |
  |---|---|---|
  | OAINonImaging | ~ | ~ |
  | OAIScreeningImages | 44.2 GB | |
  | OAIBaselineImages | 983.1 GB | ~2 TB |
  | OAI12MonthImages | 814.2 GB | |
  | OAI24MonthImages | 803.8 GB | |
  | OAI48MonthImages | 807.5 GB | |

  Other visits are listed on the same page. [NDA-CMS /oai/accessing_images]
- Custom packages are the alternative. "You can also build custom packages containing DICOM images zipped at the series level using the NDA query tools and the 'image' structure." The images are "subject to GZIP compression at the series level" (tar.gz per series). [NDA-CMS /oai/accessing_images, /oai/query_data_and_documentation, /oai/querying_oai_data_in_nda]
  - This route lets you pull only the needed series (for example SAG_IW_TSE and DESS for 2,399 knees) instead of the 983 GB baseline zip.
  - Packages are built in the web UI. Later, `downloadcmd -dp <id>` can run on the server.
  - Warning on the "Create Package" page: "Do NOT check the 'include data files' box, otherwise all images will attempt to download". This applies when you only want tables.
- Do not mix methods: "Data downloaded using different methods should not be combined ... the user ids and file structure are different when using multiple methods." NDA-format data uses NDA GUIDs. The original zip and ASCII/SAS releases use the OAI 7-digit ID. [NDA-CMS /oai/query_data_and_documentation]
- **UNVERIFIED:** whether `--file-regex` or `-t` can pull a single per-batch zip out of a shared visit package. A `--verify` run on the shared package would show which files it contains.
- Non-imaging data, including the MOAKS tables, comes in two forms:
  - "Full Downloads" zips in original ASCII/SAS: "MR Image Assessment_ASCII.zip / _SAS.zip" and "MRI MetaAnalysis_*.zip".
  - The OAINonImaging shared package, in NDA format. It includes `image03.txt`. [NDA-CMS /oai/full_downloads; uncbiag code below reads `OAINonImaging/image03.txt`]

## 2. OAI MRI file layout and sequences

### 2.1 Layout of the original per-visit zip
- Folder path pattern: `<release>/<ParticipantID>/<StudyDate yyyymmdd>/<Barcode/series number>/` holding DICOM files. Example: `0.C.2/9003380/20050110/10323514`. [denizlab CSV https://github.com/denizlab/OAI-MRI-TKR (csvs/HDF5_00_cohort_2_prime.csv); 3DReasonKnee split_files: `../data/OAI/00mDESS/0.C.2/9010308/20040915/10243212`]
- Baseline releases are `0.C.2` and `0.E.1`. Map of release to visit: `0.C.2/0.E.1`→V00, `1.C.2/1.E.1`→V01 (12m), `2.D.2`→18m, `3.C.2/3.E.1`→24m, `4.G.1`→30m, `5.C.1/5.E.1`→36m, `6.C.1/6.E.1`→48m, `8.C.1/8.E.1`→72m, `10.C.1/10.E.1`→96m. [imedslab OAProgressionMR `oaprmr/datasets/oai/_constants.py`, https://github.com/imedslab/OAProgressionMR]
- Each visit has an inventory CSV, `contents.<release>.csv` (for example `contents.0E1.csv` for baseline), with columns `Folder, ParticipantID, StudyDate, Barcode, StudyDescription, SeriesDescription`. Example row: `0.E.1/9000099/20050712/10424405, 9000099, 20050712, 10424405, OAI^MR^ENROLLMENT^LEFT, SAG_3D_DESS_LEFT`. [uncbiag OAI_analysis `data/make_data_sheet.py` and `data/SEG_3D_DESS_all.csv`, https://github.com/uncbiag/OAI_analysis]
- The barcode is the DICOM AccessionNumber and links to the image-assessment rows through `VxxMRBARCD` in the MRIxx datasets. [IA-OVERVIEW section 9]
- In NDA-format data, the manifest is the `image03` structure. Fields used: `image_file` (S3 path), `src_subject_id`, `interview_date`, `image03_id`, `image_description` (= SeriesDescription) and `comments_misc` (= study description, e.g. visit). [uncbiag `get_96mon_sheet()` in make_data_sheet.py]

### 2.2 Sequences: SeriesDescription values and protocol
- Full SeriesDescription list for knee MRI (from imedslab's mapping of the contents files):
  - `MP_LOCATOR_{LEFT,RIGHT}`
  - `COR_IW_TSE_{L,R}`
  - `SAG_3D_DESS_{L,R}`
  - `COR_MPR_{L,R}`
  - `AX_MPR_{L,R}`
  - `SAG_IW_TSE_{L,R}`
  - `COR_T1_3D_FLASH_{L,R}`
  - `SAG_T2_MAP_{L,R}`
  - `SAG_T2_CALC_{L,R}`
  - Thigh series: `MP_LOCATOR_THIGH`, `AX_T1_THIGH`, `PRESCRIPTION_THIGH`

  [imedslab `entry/Targets_meta_and_scans_from_OAI.ipynb`]
- Left or right is identified by the `_LEFT`/`_RIGHT` suffix of SeriesDescription. The StudyDescription also ends in `^LEFT`/`^RIGHT`. [same; uncbiag CSV]
  - 3DReasonKnee's `get_knee_side()` reads DICOM SeriesDescription: `'RIGHT' if series_desc == "SAG_3D_DESS_RIGHT" else 'LEFT'`. [3DRK dcm2nifti.py]
  - OAI tables code `SIDE=1` as right and `SIDE=2` as left. [IA-OVERVIEW section 4; imedslab `side_code_to_str = {1:'RIGHT', 2:'LEFT'}`]
- Protocol (3T Siemens Trio, both knees unless noted). [PETERFY Table I/II; NDA-CMS /oai/image-acquisitions]

  | SeriesDescription | Protocol name | Fat sat | Slices / thickness | Notes |
  |---|---|---|---|---|
  | SAG_IW_TSE | SAG IW 2D TSE FS | **Yes** | 37 / 3 mm, matrix 313x448 (recon 444x448, 0.357x0.357x3.0 mm per denizlab CSV) | Only fat-suppressed intermediate-weighted (PD-like) series. Used for MOAKS BML/effusion scoring. Closest to clinical PD-FS. |
  | COR_IW_TSE | COR IW 2D TSE | No | 35 slices; thickness see note | Non-FS intermediate-weighted coronal |
  | SAG_3D_DESS | SAG 3D DESS WE | water excitation | 160 / 0.7 mm, 384x384 (0.365x0.365x0.7 mm per denizlab CSV) | 3D gradient echo, not PD |
  | COR_MPR / AX_MPR | MPR of DESS | – | reformats (coronal 1.5 mm per PETERFY text) | |
  | COR_T1_3D_FLASH | COR T1W 3D FLASH WE | WE | 80 / 1.5 mm | Right knee only (left if right has metal) |
  | SAG_T2_MAP | SAG 2D MESE | No | 21 / 3 mm | Right knee only |

  Note: my automated extraction of PETERFY Table I gave COR IW TSE as "5/1 mm" (the same as the localizer). That may be a parsing slip, so check it against the paper. **UNVERIFIED** for COR_IW_TSE slice thickness.
- Approximate size per series is **ESTIMATED**, not measured. Uncompressed 16-bit pixel data:
  - DESS: ~384·384·160·2 B ≈ 47 MB
  - SAG_IW_TSE: ~444·448·37·2 B ≈ 15 MB
  - COR_IW_TSE: ~10–15 MB
  - OAIBaselineImages is 983.1 GB zipped for ~4,796 participants, which is about 205 MB per participant for both knees and all series. [NDA-CMS; participant count from uncbiag comment "baseline 4796"]
  - The competitor's 140 GB for 2,399 knees works out to ~58 MB per knee. That fits one DESS plus one IW-TSE series per knee (inference).

## 3. MOAKS variables (OAI kMRI_SQ_MOAKS_BICL00 and kMRI_FNIH_SQ_MOAKS_BICL00)

### 3.1 Datasets and keys
- Dataset names are `kMRI_SQ_MOAKS_BICLxx` and `kMRI_FNIH_SQ_MOAKS_BICLxx`, where xx = visit (00 = baseline). Vendor is BICL (Boston Imaging Core Lab). Project 22 is the FNIH OA Biomarkers Consortium, 600 knees at BL/12m/24m. [IA-OVERVIEW Table 2b]
  - kMRI_SQ_MOAKS_BICL00 also contains other projects. In the 3DReasonKnee copy, READPRJ counts are 65: 1157 rows, 22: 600, 30: 600, 63A-F: 660. That is 3,017 rows and **2,400 unique (ID, SIDE) knees**, which matches the competitor's 2,399. 544 knees have more than one row (several projects). [3DRK `data/labels/taxonomymatch_kMRI_SQ_MOAKS_BICL00.csv`, my count]
- Record key: "The variables uniquely identifying a record in these datasets are ID, SIDE, and READPRJ". SIDE=1 is right and SIDE=2 is left. For longitudinal comparison, use rows from the same READPRJ. [IA-OVERVIEW section 4]
  - The 3DReasonKnee copy also has a VERSION column (0.2 in SQ, 0.1 in FNIH). Its FNIH copy has an extra column with values like `R02`, probably a reader or read type (**UNVERIFIED**).
  - 3DReasonKnee chose "the most recent available MOAKS assessment" when a knee had several readings. [3DRK Supplementary.pdf]
- Where the official dictionary is (login required):
  - NDA OAI "Study Documentation" → "IMAGE ASSESSMENT DOCUMENTATION" (`Image Assessments.zip`, which holds the variable descriptions for the central image assessments).
  - "Variable Guide_Alphabetical" (`VG_Variable.pdf`) and "Variable Guide_CRF Based" (`VG_Form.pdf`).
  - Data files: `MR Image Assessment_ASCII.zip` / `_SAS.zip`.

  [NDA-CMS /oai/study_documentation, /oai/query_data_and_documentation, /oai/full_downloads]
  - The NDA-format data dictionary search for OAI: https://nda.nih.gov/general-query.html?q=query=data-structure%20~and~%20dataSources=Osteoarthritis%20Initiative
  - I could not open the official OAI variable guide. The names below come from the 3DReasonKnee header (`taxonomymatch_*`, i.e. as 3DReasonKnee re-labelled them) and are cross-checked against the sibling MOST MOAKS guide, which uses the same naming scheme (https://agingresearchbiobank.nia.nih.gov/studies/most/documents/download/Variable_Guides/VariableGuide_V79MOAKS.pdf/).

### 3.2 Variable names (prefix V00 = baseline)
Subregion codes:
- P = patella: PM = medial, PL = lateral
- F = femur: FMA/FLA = medial/lateral trochlea ("anterior"), FMC/FLC = central (weight-bearing), FMP/FLP = posterior
- T = tibia: TMA/TMC/TMP and TLA/TLC/TLP; SS = subspinous

[MOAKS paper; MOST guide labels, e.g. "V7MCMFMA ... femur medial anterior"]

| Feature | OAI column names (3DRK header) | Scale | Note |
|---|---|---|---|
| Cartilage, per subregion | `V00MCM{PM,PL,FMA,FLA,FMC,FLC,FMP,FLP,TMA,TLA,TMC,TLC,TMP,TLP}`. There are **two values per subregion**: area grade then full-thickness grade (3DRK names the second one `<name>.1`) | Area of any loss: 0 none, 1 <10%, 2 10–75%, 3 >75%. Full-thickness share: same 0–3 bins. Often written as the two-digit code "area.FT", e.g. 2.1 [MOAKS] | **UNVERIFIED:** the real OAI name of the second (FT) column. MOST uses `V7MCMxx` for the combined code plus `V7a_MCMxx` (area) and `V7b_MCMxx` (FT) |
| BML size | `V00MBMS{FMA,FLA,FMC,FLC,FMP,FLP,SS,TMA,TLA,TMC,TLC,TMP,TLP,PM,PL}` | 0 none, 1 <33%, 2 33–66%, 3 >66% of subregion volume [MOAKS] | MOST has the same (`V7MBMSFMA`) |
| BML % that is BML (vs cyst) | `V00MBMP{...}` | 0–3 | 3DRK header; MOST has `MBMC` "% cystic" instead (**UNVERIFIED** OAI meaning) |
| BML count | `V00MBMN{...}` | integer | MOST `V7MBMN..` "#of BMLS" |
| Osteophytes (12 sites) | `V00MOS{PS,PI,PM,PL,FMA,FLA,FMP,FLP,FMC,FLC,TM,TL}` (patella superior/inferior/medial/lateral, femur trochlea M/L, posterior M/L, central M/L, tibia M/L) | 0 none, 1 small, 2 medium, 3 large [MOAKS] | `MOSFMA` = "osteophyte ... femur medial anterior trochlear" is confirmed in an OAI paper (https://www.sciencedirect.com/science/article/pii/S000349672400565X, from the search snippet) |
| Hoffa-synovitis | `V00MSYIC` (3DRK header) | 0 normal, 1 mild, 2 moderate, 3 severe [MOAKS] | MOST equivalent `V7MSYIP` "MOAKS Synovitis - Hoffa (infra-patellar)". **UNVERIFIED:** exact OAI spelling (SYIC vs SYIP). `V00MSYNVTS` was not seen anywhere |
| Effusion-synovitis | `V00MEFFWK` | 0 physiologic, 1 small, 2 medium, 3 large [MOAKS Table 4] | MOST `V7MEFWK` "Effusion - whole knee" |
| Meniscus | `V00MMT{MA,LA,MB,LB,MP,LP}` (morphology code 0–8), `V00MMX{MM,MA,LA,LL}` (extrusion 0–3), `V00MMH..`, `V00MMRTM` | | |
| Others in header | `V00MACLTR`, `V00MPCLTR`, `V00MPTSIG`, `V00MANSBUR`, `V00MIPBUR`, `V00MPPBUR`, `V00MITBSIG`, `V00MPOPCYS` | | |

Compartment grouping (subregions from [MOAKS]):
- Patellofemoral: cartilage/BML PM, PL, FMA, FLA. Osteophytes PS, PI, PM, PL, FMA, FLA.
- Lateral tibiofemoral: cartilage/BML FLC, FLP, TLA, TLC, TLP. Osteophytes FLC, FLP, TL.
- Medial tibiofemoral: FMC, FMP, TMA, TMC, TMP. Osteophytes FMC, FMP, TM.

Coverage by project in the 3DRK copy of BICL00 (my count; values "." mean not read):
- Osteophytes: only READPRJ 22 (FNIH, 600 knees) has them. Projects 30, 63x and 65 are "." for all `V00MOS*`.
- `V00MSYIC` and `V00MEFFWK`: present in projects 22 and 65 only, about 1,755 rows.
- Cartilage and BML size: present in every project.
- So an osteophyte-based OA definition is only possible for the 600 FNIH knees.

## 4. 3DReasonKnee
- What it is: a dataset of 494k quintuples (question, 3D bounding box, clinician reasoning, MOAKS-structured answer) built on 7,970 OAI SAG 3D DESS volumes from baseline to 48 months. Paper: Sambara S, ..., Rajpurkar P. "3DReasonKnee: Advancing Grounded Reasoning in Medical Vision Language Models", arXiv 2510.20967 (https://arxiv.org/abs/2510.20967).
- Location: https://huggingface.co/datasets/rajpurkarlab/3DReasonKnee. No GitHub repo is linked. The code is in the HF repo under `code/`.
- Licence: "released under the Creative Commons Attribution 4.0 International (CC BY 4.0)". The README says raw OAI DICOMs must still be obtained from NDA. [3DRK README] The arXiv page also lists CC BY 4.0.
- What it maps:
  - `data/split_files/all_studies.csv` (8,020 rows) and `{train,val,test}_split.csv` have the columns `patient_id,time,study_folder,series_folder,knee_side,path,image_filepath,label_filepath`. Example: `9010308,00M,20040915,10243212,1,/data/OAI/00mDESS/0.C.2/9010308/20040915/10243212,...`.
  - That gives OAI ID + visit + study date + DESS series folder/barcode + side (1=R, 2=L), which you can join to MOAKS (ID, SIDE).
  - Baseline: 2,428 rows / 2,416 unique (ID, side). Releases 0.C.2: 1,470, 0.E.1: 958. [my count]
  - `data/labels/taxonomymatch_kMRI_SQ_MOAKS_BICL{00,01,03,06}.csv` and `taxonomymatch_kMRI_FNIH_SQ_MOAKS_BICL{00,01,03}.csv` are re-exports of the OAI MOAKS tables with OAI variable names.
  - Also included: nnU-Net subregion and lesion masks (`masks_aa/ab`) and question/grade mapping JSONs.
- Matching method: "most recent available MOAKS assessment was selected" per study. Side comes from DICOM SeriesDescription. [3DRK Supplementary.pdf; dcm2nifti.py] It only covers **DESS** series, so locating the matching SAG_IW_TSE series still needs the OAI contents/image03 inventory (same participant/date folder, SeriesDescription `SAG_IW_TSE_<SIDE>`).

## 5. Turning MOAKS into binary "OA present" labels
- Hunter DJ et al., "Definition of osteoarthritis on MRI: results of a Delphi exercise", Osteoarthritis Cartilage 2011;19(8):963-969, doi:10.1016/j.joca.2011.04.017 (https://pure.psu.edu/en/publications/definition-of-osteoarthritis-on-mri-results-of-a-delphi-exercise/). Summary from https://mdedge.com/content/panel-proposes-mri-role-knee-oa-diagnosis-0:
  - **Tibiofemoral OA** = both group A features, or one A plus ≥2 B.
    - A: definite osteophyte; full-thickness cartilage loss.
    - B: subchondral BML or cyst not at meniscal/ligament attachments; meniscal subluxation, maceration or degenerative (horizontal) tear; partial-thickness cartilage loss.
    - Exclusions: joint trauma in the last 6 months, inflammatory arthritis.
  - **Patellofemoral OA** = both of: definite osteophyte, and partial- or full-thickness cartilage loss, on the patella and/or anterior femur (trochlea).
  - Effusion/synovitis, BML, meniscal damage etc. alone are "not diagnostic".
- How published OAI studies turned the definition into MOAKS thresholds:
  - MacKay et al./PMC12796920 ("MRI-defined knee OA rarely occurs without cartilage damage", OAI, tibiofemoral only; https://pmc.ncbi.nlm.nih.gov/articles/PMC12796920/):
    - Definite osteophyte: "a score >0 in any of the 6 TF locations".
    - Full-thickness cartilage: "a score of 1.1, 2.1, 2.2, 3.1, 3.2, or 3.3" in any of 10 TF subregions.
    - Partial-thickness: "1.0, 2.0, or 3.0".
    - BML: size >0.
    - Meniscus: morphology score 2, 3, 5, 6, 7 or 8, or extrusion >1.
    - The same osteophyte definition is in MacKay 2022, Arthritis Care Res (https://acrjournals.onlinelibrary.wiley.com/doi/10.1002/acr.24605, from the search snippet).
  - arXiv 2209.01192 (https://arxiv.org/pdf/2209.01192) states TF OA as: Group A "definite osteophyte formation >= 2, full thickness cartilage loss >= 3"; Group B "BML or cyst >= 1, meniscal subluxation, maceration or degenerative tear >= 2, partial thickness cartilage loss >= 1". These thresholds differ from MacKay's. Treat them as one author's reading.
  - arXiv 2212.07023 converted MOAKS to binary phenotypes "according to the definitions recommended by Roemer et al." (https://arxiv.org/pdf/2212.07023).
- Synovitis or effusion as binary: Roemer's "inflammatory phenotype" uses MOAKS grade 3 in one of (intercondylar synovitis or whole-knee effusion) plus at least grade 2 in the other (https://josr-online.biomedcentral.com/articles/10.1186/s13018-024-05194-w, from the search snippet). **UNVERIFIED** that this matches the original Roemer papers, and the original MOAKS has no intercondylar synovitis item. For single features, grade ≥2 (moderate) is a common cutoff, but I found no authoritative source for "Hoffa ≥2 = synovitis present", so this is **UNVERIFIED**.
- Forum or competitor write-ups: I found none for RSNA Knee Abnormality Detection that describe the OAI/MOAKS mapping (searched Kaggle/GitHub). The competitor's exact thresholds are **UNVERIFIED**.
