# ==================== EDA 3 — the reports behind the labels ================
# Almost every label used to fit these weights was read out of a free-text
# report by a text model. This notebook ships a hand-written multilingual
# extractor for exactly that job, so it is worth seeing what it is reading.
if _EDA_TRAIN is not None and "Report" in _EDA_TRAIN.columns:
    reports = _EDA_TRAIN["Report"].fillna("").astype(str)
    words = reports.str.split().str.len()

    # Cheap language fingerprints, using stems the extractor itself keys on.
    probes = {
        "English": r"\btear\b|\bmeniscus\b|\beffusion\b",
        "Spanish/Portuguese": r"\brotura\b|\bmenisco\b|\bderrame\b",
        "Italian": r"\blesione\b|\bmenisco\b|\bversamento\b",
        "French": r"\bd.chirure\b|\bm.nisque\b|\b.panchement\b",
        "German": r"\briss\b|\bmeniskus\b|\berguss\b",
        "Turkish": r"\byirtik\b|\bmenisk.s\b|\befuzyon\b",
    }
    lowered = reports.str.lower()
    hits = {name: int(lowered.str.contains(pattern, regex=True, na=False).sum())
            for name, pattern in probes.items()}

    figure, axes = plt.subplots(1, 2, figsize=(13, 3.6))
    axes[0].hist(words[words < words.quantile(0.99)], bins=60, color=_EDA_INK)
    axes[0].set_title(f"Report length (median {words.median():.0f} words)")
    axes[0].set_xlabel("words")
    order = sorted(hits, key=hits.get, reverse=True)
    axes[1].bar(order, [hits[k] for k in order], color=_EDA_WARM)
    axes[1].set_title("Reports matching each language's finding vocabulary")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()

    empty = int((words == 0).sum())
    print(f"{empty:,} studies have no report at all "
          f"({empty / max(len(reports), 1):.1%}) — those can only be supervised by "
          f"the images.")
    print("The counts overlap because clinical vocabulary is shared across "
          "languages; they are a rough shape, not a classifier. The point is that "
          "a monolingual extractor silently drops whole slices of the training set, "
          "and a dropped study is not a wrong label — it is no label.")