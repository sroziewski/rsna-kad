import pytest

from rsna_kad.constants import LABEL_KEYS
from rsna_kad.labels.rules import (LANGS, VOCAB, extract_rules, extract_rules_detail, knee_side, normalize,
                                   split_sentences)

TR_NORMAL_TEMPLATE = ("diz eklemi içi sıvı miktarı normal. çapraz ve yan bağlar normal. medyal ve lateral menisküs "
                      "normal. eklem kıkırdakları normal. kemik iliği sinyal intensitesi normal. patellar tendon ve "
                      "kuadriseps tendonu normal. popliteal fossada kist izlenmedi.")


def test_normalize_folds_scripts():
    assert normalize("ÖÇB Yırtığı") == "ocb yirtigi"
    assert normalize("Außenmeniskus") == "aussenmeniskus"
    assert normalize("ρήξη ΠΧΣ") == "ρηξη πχσ"
    assert normalize("µηνίσκος") == normalize("μηνίσκος")


def test_split_sentences():
    s = split_sentences("ACL intact. No effusion; small Baker cyst.\nMCL normal")
    assert s == ["acl intact.", "no effusion;", "small baker cyst.", "mcl normal"]


def test_vocab_complete():
    for lang in LANGS:
        v = VOCAB[lang]
        assert set(v.terms) == set(LABEL_KEYS), lang
        assert v.negators and v.strong and v.mild and v.oa_cues, lang
        assert v.left and v.right


@pytest.mark.parametrize("lang,text,expect", [
    ("en", "There is a complete tear of the ACL. No joint effusion. Small Baker's cyst. Normal medial meniscus.",
     {"acl": "affirmed", "effusion": "negated", "baker": "affirmed", "med_men": "negated", "lat_men": "absent"}),
    ("en", "Medial and lateral menisci are intact. Bone marrow signal is normal. No fracture.",
     {"med_men": "negated", "lat_men": "negated", "contusion": "negated", "fracture": "negated"}),
    ("en", "Full-thickness cartilage loss in the medial compartment with subchondral cysts. Lateral compartment cartilage preserved.",
     {"med_oa": "affirmed", "lat_oa": "negated"}),
    ("en", "Grade 2 chondromalacia of the patella. Mucoid degeneration of the ACL.",
     {"pf_oa": "affirmed", "acl": "affirmed"}),
    ("es", "Rotura del cuerno posterior del menisco interno. Ligamentos cruzados y colaterales continuos. Sin derrame articular. Sinovitis.",
     {"med_men": "affirmed", "acl": "negated", "mcl": "negated", "effusion": "negated", "synovitis": "affirmed"}),
    ("es", "No hay alteración de señal difusa de la médula ósea. Artrosis femorotibial medial y PF.",
     {"contusion": "negated", "med_oa": "affirmed", "pf_oa": "affirmed"}),
    ("tr", "Ön çapraz bağda tam kat yırtık izlendi. Medial menisküs posterior hornunda grade II dejenerasyon. Lateral menisküs normaldir. Eklem içi efüzyon izlenmedi.",
     {"acl": "affirmed", "med_men": "affirmed", "lat_men": "negated", "effusion": "negated"}),
    ("tr", "Lateral tibial plato posteriorunda nondeplase lineer kırık ve çevresinde kemik iliği ödemi izlendi.",
     {"fracture": "affirmed", "contusion": "affirmed"}),
    ("hr", "Kompleksna ruptura stražnjeg roga medijalnog meniska. Lateralni menisk bez znakova rupture. Prednji križni ligament se prati u kontinuitetu. Manji izljev suprapatelarno.",
     {"med_men": "affirmed", "lat_men": "negated", "acl": "negated", "effusion": "affirmed"}),
    ("hr", "Hondromalacija II-III° lateralne fasete patele. Bez znakova poplitealne ciste. Nema koštanog edema.",
     {"pf_oa": "affirmed", "baker": "negated", "contusion": "negated"}),
    ("el", "Ρήξη του οπισθίου κέρατος του έσω μηνίσκου. Φυσιολογικά ελέγχονται οι πλάγιοι και χιαστοί σύνδεσμοι. Μικρή κύστη Baker. Παρατηρείται ελάχιστη ενδαρθρική συλλογή υγρού.",
     {"med_men": "affirmed", "acl": "negated", "mcl": "negated", "baker": "affirmed", "effusion": "affirmed"}),
    ("el", "Με σήμα εντός του φυσιολογικού ελέγχεται ο οστικός μυελός. Φυσιολογικά ελέγχεται ο αρθρικός χόνδρος.",
     {"contusion": "negated", "fracture": "negated", "med_oa": "negated", "pf_oa": "negated"}),
    ("bg", "МР данни за ставен излив. Частично скъсване на предната кръстна връзка. Кръстните връзки са интактни. Латералният менискус е с нормална форма. Няма данни за фрактура. Бекерова киста.",
     {"effusion": "affirmed", "acl": "affirmed", "lat_men": "negated", "fracture": "negated", "baker": "affirmed"}),
    ("bg", "Вижда се изразен контузионен костно-мозъчен едем в пателата. Хрущялът е със запазена цялост.",
     {"contusion": "affirmed", "med_oa": "negated", "pf_oa": "negated"}),
    ("de", "Horizontale Rissbildung am Innenmeniskushinterhorn. Intakte Darstellung des vorderen Kreuzbandes. Kein Gelenkerguss. Geringe Reizsynovialitis. Retropatellararthrose.",
     {"med_men": "affirmed", "acl": "negated", "effusion": "negated", "synovitis": "affirmed", "pf_oa": "affirmed"}),
    ("de", "Bone Bruise am lateralen Tibiaplateau. Keine Fraktur. Kollateralbänder durchgehend abgrenzbar.",
     {"contusion": "affirmed", "fracture": "negated", "mcl": "negated"}),
    ("nl", "Geen hydrops. Intacte voorste kruisband. Intacte mediale meniscus. Scheur laterale meniscus. Geen botoedeem. Lichte prepatellaire bursitis.",
     {"effusion": "negated", "acl": "negated", "med_men": "negated", "lat_men": "affirmed", "contusion": "negated"}),
    ("nl", "Lokaal gevorderd femoropatellair kraakbeenlijden graad 4. Kruisbanden en laterale banden intact.",
     {"pf_oa": "affirmed", "acl": "negated", "mcl": "negated"}),
    ("fr", "Pas d'épanchement articulaire. Pas de kyste poplité. Déchirure horizontale du ménisque interne. Le ligament croisé antérieur est sans particularité.",
     {"effusion": "negated", "baker": "negated", "med_men": "affirmed", "acl": "negated"}),
    ("fr", "Amincissement modéré du cartilage du compartiment interne. Contusion osseuse du plateau tibial externe. Fracture du plateau tibial.",
     {"med_oa": "affirmed", "contusion": "affirmed", "fracture": "affirmed"}),
])
def test_language_examples(lang, text, expect):
    got = extract_rules(text, lang)
    for key, state in expect.items():
        assert got[key] == state, f"{lang} {key}: got {got[key]}, expected {state}\n{text}"


def test_turkish_normal_template_negates_most_labels():
    votes, meta = extract_rules_detail(TR_NORMAL_TEMPLATE, "tr")
    n_neg = sum(v.state == "negated" for v in votes.values())
    assert n_neg >= 8, {k: v.state for k, v in votes.items()}
    assert all(v.state != "affirmed" for v in votes.values())
    assert meta["normal_template"] is True


def test_negation_window_is_local():
    # the negator is further than 5 content tokens away from the term: not negated
    got = extract_rules("No fracture of the tibia is identified, however there is a large joint effusion.", "en")
    assert got["fracture"] == "negated"
    assert got["effusion"] == "affirmed"
    far = "No significant abnormality of the patellar tendon, quadriceps tendon or extensor mechanism; there is a moderate joint effusion."
    assert extract_rules(far, "en")["effusion"] == "affirmed"


def test_severity_and_evidence():
    votes, _ = extract_rules_detail("There is a small joint effusion. Moderate synovitis.", "en")
    assert votes["effusion"].state == "affirmed" and votes["effusion"].severity == "mild"
    assert votes["synovitis"].severity == "moderate"
    assert "effusion" in votes["effusion"].evidence
    votes, _ = extract_rules_detail("ACL tear. Medial meniscus tear.", "en")
    assert votes["acl"].severity == "unspecified"


def test_affirmed_beats_negated_across_sentences():
    got = extract_rules("ACL intact. On closer inspection there is a partial tear of the ACL.", "en")
    assert got["acl"] == "affirmed"


def test_mentioned_flag_and_side():
    votes, meta = extract_rules_detail("Left knee MRI. The ACL is visualised.", "en")
    assert votes["acl"].mentioned and votes["acl"].state == "absent"
    assert meta["knee_side"] == "left"
    assert knee_side("SAĞ DİZ MRG", "tr") == "right"
    assert knee_side("MRI ΑΡΙΣΤΕΡΟΥ ΓΟΝΑΤΟΣ", "el") == "left"
    assert knee_side("left and right", "en") == "unknown"


def test_postoperative_flag():
    _, meta = extract_rules_detail("Status post ACL reconstruction with intact graft.", "en")
    assert meta["postoperative"] is True
    _, meta = extract_rules_detail("Intact ACL.", "en")
    assert meta["postoperative"] is False


def test_unknown_language_falls_back_to_english():
    assert extract_rules("No effusion.", "xx")["effusion"] == "negated"
