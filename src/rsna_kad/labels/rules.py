"""Rule voter: per-language term lists + negation windows -> state per label.

Text is normalised (NFKC, casefold, diacritics stripped, Turkish dotless i folded) so
one pattern covers spelling variants (ÖÇB/öçb -> ocb, Außenmeniskus -> aussenmeniskus,
ρήξη -> ρηξη, µ -> μ). Patterns are written in that normalised form.

Per sentence, a label is affirmed when a finding term (effusion, synovitis, ...) or a
structure term with a pathology cue (ACL + tear) is found without a negator inside the
window, negated when a negator is inside the window, and absent when nothing matches.
The window is measured in content tokens (stopwords skipped) so "no hay alteración de
señal de la médula ósea" still negates the marrow term.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from rsna_kad.constants import LABEL_KEYS

LANGS: tuple[str, ...] = ("en", "es", "tr", "hr", "el", "de", "bg", "nl", "fr")
FINDING_KEYS: tuple[str, ...] = ("effusion", "synovitis", "baker", "contusion", "fracture")
STRUCT_KEYS: tuple[str, ...] = ("acl", "mcl", "med_men", "lat_men")
OA_KEYS: tuple[str, ...] = ("med_oa", "lat_oa", "pf_oa")
WINDOW = 5
SEVERITY_WINDOW = 6

# ------------------------------------------------------------------ normalisation


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("ı", "i")
    return re.sub(r"[ \t\r\f\v]+", " ", text)


def _norm_pattern(p: str) -> str:
    # No casefold here: it would turn \W into \w. Patterns are written lower-case.
    p = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", p))
    p = "".join(ch for ch in p if unicodedata.category(ch) != "Mn")
    return p.replace("ı", "i").replace("ς", "σ")  # casefold maps final sigma to σ


_SENT_SPLIT = re.compile(r"(?<=[.;!?])\s+|\n+|\s+[-•·]\s+|(?<=\.)(?=[^\s\d])")


def split_sentences(text: str) -> list[str]:
    # "Fractures :\nAucune." - a heading and its value belong to one sentence.
    text = re.sub(r":\s*\n+\s*", ": ", normalize(text))
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _compile(patterns: Iterable[str]) -> list[re.Pattern]:
    return [re.compile(r"(?<![^\W_])" + _norm_pattern(p), re.UNICODE) for p in patterns]


# ------------------------------------------------------------------ vocabulary


@dataclass
class LangVocab:
    terms: dict[str, list[str]]                 # label key -> patterns
    both: dict[str, tuple[tuple[str, ...], list[str]]]  # group -> (keys, patterns)
    oa_cues: list[str]
    strong: list[str]                            # ligament/meniscus pathology (tear...)
    mild: list[str]                              # sub-threshold pathology (degeneration...)
    negators: list[str]
    sev_mild: list[str]
    sev_moderate: list[str]
    sev_severe: list[str]
    stopwords: set[str]
    contrast: set[str]
    left: list[str] = field(default_factory=list)
    right: list[str] = field(default_factory=list)
    postop: list[str] = field(default_factory=list)


_SHARED_POSTOP = [r"postoperat", r"post-?op\b", r"reconstruct", r"rekonstru", r"reconstru",
                  r"graft", r"greft", r"meniscectom", r"menisectom", r"meniskektom",
                  r"arthroscop", r"artroskop", r"artroscop", r"plasty\b", r"plastik",
                  r"plastie", r"plastika", r"status post", r"s/p\b", r"z\.n\.", r"suture anchor",
                  r"ankerschraub", r"tunnel"]

VOCAB: dict[str, LangVocab] = {}

VOCAB["en"] = LangVocab(
    terms={
        "acl": [r"acl\b", r"anterior cruciate", r"ant\.? cruciate"],
        "mcl": [r"mcl\b", r"medial collateral", r"tibial collateral"],
        "med_men": [r"medial menisc", r"inner menisc", r"medial(,| and| &| ?/) ?lateral menisc"],
        "lat_men": [r"lateral menisc", r"outer menisc", r"lateral(,| and| &| ?/) ?medial menisc"],
        "effusion": [r"effusion", r"joint fluid", r"intra-?articular fluid", r"hydrops",
                     r"h?a?emarthrosis", r"lipoh?a?emarthrosis", r"suprapatellar fluid"],
        "synovitis": [r"synovitis", r"synovial (thickening|proliferation|hypertrophy|enhancement)"],
        "baker": [r"baker", r"popliteal cyst"],
        "contusion": [r"contusion", r"bone bruis", r"bone marrow (o?edema|oedema)",
                      r"marrow (o?edema|oedema|signal)", r"bone o?edema", r"osseous o?edema",
                      r"trabecular (o?edema|injury)"],
        "fracture": [r"fractur"],
        "med_oa": [r"medial (tibio)?femoral", r"medial compartment", r"medial femoral condyle",
                   r"medial tibial plateau", r"medial (femoro)?tibial", r"medial joint",
                   r"medially", r"medial(?! ?(menisc|collateral|retinac|plica|horn|gastroc|patell|band|ligament|femoral condyle\w* (bone|marrow)))"],
        "lat_oa": [r"lateral (tibio)?femoral", r"lateral compartment", r"lateral femoral condyle",
                   r"lateral tibial plateau", r"lateral (femoro)?tibial", r"lateral joint",
                   r"laterally", r"lateral(?! ?(menisc|collateral|retinac|plica|horn|gastroc|patell|band|ligament))"],
        "pf_oa": [r"patello-?femoral", r"retropatellar", r"patella", r"trochlea", r"femoro-?patellar"],
    },
    both={
        "menisci": (("med_men", "lat_men"), [r"menisci", r"both menisc", r"meniscal", r"menisc"]),
        "cruciates": (("acl",), [r"cruciate"]),
        "collaterals": (("mcl",), [r"collateral"]),
        "ligaments": (("acl", "mcl"), [r"ligament"]),
        "cartilage": (("med_oa", "lat_oa", "pf_oa"), [r"cartilag", r"chondral", r"articular surface"]),
        "bone": (("contusion", "fracture"), [r"bone marrow", r"marrow signal", r"osseous structures",
                                             r"bones\b", r"bony structures", r"bone signal", r"marrow\b"]),
    },
    oa_cues=[r"osteoarthr", r"arthrosis", r"chondromalacia", r"chondropathy", r"osteophyt",
             r"cartilage (loss|thinning|defect|fissur|wear|damage|irregular|degenerat|ulcer|denud|erosion|fibrillation|softening|delamin|injury|lesion|abnormal|narrow)",
             r"(loss|thinning|defect|fissuring|fibrillation|erosion|denudation|delamination|irregularity) of (the )?(articular |hyaline )?cartilage",
             r"full[- ]thickness (cartilage|chondral|defect|fissur)", r"partial[- ]thickness (cartilage|chondral|defect|fissur)",
             r"joint space narrow", r"degenerative change", r"subchondral (cyst|sclerosis|marrow)",
             r"chondral (defect|loss|thinning|fissur|lesion|injury|delamin|abnormal|irregular)", r"bone[- ]on[- ]bone",
             r"spurring", r"cartilage is (thinned|denuded|absent|lost)", r"denuded", r"chondrosis"],
    strong=[r"tear", r"torn", r"ruptur", r"discontinu", r"disrupt", r"avuls", r"fragment", r"bucket",
            r"flap", r"displaced", r"macerat", r"truncat", r"detach", r"radial", r"horizontal", r"oblique",
            r"complex", r"lesion", r"injur", r"non-?visuali", r"not (visuali[sz]ed|seen|identified)",
            r"absent\b(?! fibers? disruption)", r"extru", r"cleavage", r"root", r"parrot", r"longitudinal", r"vertical", r"peripheral"],
    mild=[r"degenerat", r"mucoid", r"myxoid", r"sprain", r"strain", r"o?edema", r"thicken", r"intrasubstance",
          r"intra-substance", r"grade (1|i|2|ii)\b", r"signal (change|abnormal|alter|increase|intensity|heterogen)",
          r"increased signal", r"ganglion", r"cyst", r"fray", r"attenuat", r"laxity", r"lax\b", r"buckl",
          r"contusion", r"partial", r"hyperintens", r"heterogen", r"irregular", r"blunt", r"intermediate signal"],
    negators=[r"\bno\b", r"\bnot\b", r"without", r"absent\b", r"absence", r"negative", r"normal",
              r"intact", r"unremarkable", r"preserved", r"free of", r"neither", r"nor\b", r"none\b",
              r"resolved", r"maintained", r"regular", r"physiolog", r"within normal", r"healthy",
              r"no evidence", r"unchanged", r"well[- ]defined", r"good condition", r"appropriate", r"n\.a\.d"],
    sev_mild=[r"mild", r"minimal", r"minor", r"small", r"trace", r"slight", r"low[- ]grade", r"grade (1|i|2|ii)\b",
              r"early", r"incipient", r"subtle", r"superficial", r"partial[- ]thickness", r"intrasubstance",
              r"mucoid", r"tiny", r"little", r"physiologic", r"minute", r"faint", r"borderline", r"scant", r"few"],
    sev_moderate=[r"moderate", r"medium", r"intermediate", r"partial"],
    sev_severe=[r"large", r"severe", r"extensive", r"marked", r"high[- ]grade", r"complete", r"full[- ]thickness",
                r"grade (3|iii|4|iv)\b", r"advanced", r"significant", r"gross", r"massive", r"prominent",
                r"substantial", r"considerable", r"bone[- ]on[- ]bone", r"end[- ]stage", r"displaced", r"bucket", r"huge"],
    stopwords={"the", "a", "an", "of", "in", "at", "on", "to", "with", "and", "or", "is", "are", "was", "were",
               "be", "been", "there", "this", "that", "it", "its", "for", "by", "as", "from", "which", "s"},
    contrast={"but", "however", "although", "whereas", "while", "except"},
    left=[r"left"], right=[r"right"],
    postop=_SHARED_POSTOP + [r"repair", r"surgery", r"surgical"],
)

VOCAB["es"] = LangVocab(
    terms={
        "acl": [r"lca\b", r"cruzado anterior", r"l\.c\.a\b"],
        "mcl": [r"lcm\b", r"lci\b", r"lli\b", r"colateral (medial|interno|tibial)", r"lateral interno"],
        "med_men": [r"menisco (interno|medial)", r"meniscal (interno|medial)", r"menisco int\b", r"meniscos (interno|medial)"],
        "lat_men": [r"menisco (externo|lateral)", r"meniscal (externo|lateral)", r"menisco ext\b", r"meniscos (interno|medial)\w* y (externo|lateral)", r"meniscos (externo|lateral)"],
        "effusion": [r"derrame", r"liquido (articular|intraarticular|intra-articular|libre|sinovial)",
                     r"hidrartros", r"hemartros", r"lipohemartros", r"distension (capsular|de la capsula|de los recesos)",
                     r"aumento de liquido", r"liquido en (la )?(articulacion|el receso|los recesos)"],
        "synovitis": [r"sinovitis", r"engrosamiento sinovial", r"proliferacion sinovial", r"hipertrofia sinovial", r"sinovial engros"],
        "baker": [r"baker", r"quiste popliteo"],
        "contusion": [r"contusion", r"edema (oseo|de la medula osea|medular|trabecular|de medula|en la medula)",
                      r"bone bruise", r"contusion osea", r"edema en el hueso", r"edema subcondral"],
        "fracture": [r"fractura"],
        "med_oa": [r"femoro-?tibial (interno|medial|int\b)", r"compartim(i)?ento (interno|medial)",
                   r"condilo (femoral )?(interno|medial)", r"platillo tibial (interno|medial)",
                   r"tibiofemoral (interno|medial)", r"(interno|medial)(?! ?(menisc|colateral|retinac|plica|cuerno|patel))"],
        "lat_oa": [r"femoro-?tibial (externo|lateral|ext\b)", r"compartim(i)?ento (externo|lateral)",
                   r"condilo (femoral )?(externo|lateral)", r"platillo tibial (externo|lateral)",
                   r"tibiofemoral (externo|lateral)", r"(externo|lateral)(?! ?(menisc|colateral|retinac|plica|cuerno|patel|interno))"],
        "pf_oa": [r"femoro-?patelar", r"patelo-?femoral", r"rotul", r"patela", r"retropatelar", r"troclea", r"\bpf\b", r"\bfp\b"],
    },
    both={
        "menisci": (("med_men", "lat_men"), [r"meniscos", r"ambos meniscos", r"menisc"]),
        "cruciates": (("acl",), [r"cruzados", r"cruzado"]),
        "collaterals": (("mcl",), [r"colaterales", r"colateral"]),
        "ligaments": (("acl", "mcl"), [r"ligament"]),
        "cartilage": (("med_oa", "lat_oa", "pf_oa"), [r"cartilago", r"condral", r"superficies articulares"]),
        "bone": (("contusion", "fracture"), [r"medula osea", r"senal osea", r"estructuras oseas", r"huesos", r"medular", r"\bosea\b", r"\boseo\b"]),
    },
    oa_cues=[r"artrosis", r"gonartrosis", r"osteoartr", r"condromalacia", r"condropatia", r"adelgazamiento",
             r"perdida (del|de) (cartilago|grosor|espesor)", r"cartilago .{0,30}(adelgaz|perdida|defecto|fisur|irregular|ulcer|erosion|denud|lesion|degenerat|fibrilac)",
             r"defecto (condral|del cartilago|osteocondral)", r"osteofit", r"pinzamiento (femoro|articular|del espacio|de la interlinea)",
             r"cambios degenerativos", r"espesor (total|completo)", r"esclerosis subcondral", r"quiste subcondral", r"geoda",
             r"fibrilacion", r"lesion condral", r"condral", r"artropatia degenerativa", r"disminucion del espacio"],
    strong=[r"rotura", r"roto\b", r"rota\b", r"desgarro", r"ruptura", r"discontinu", r"interrup", r"lesion", r"fragment",
            r"asa de cubo", r"desinser", r"avulsion", r"radial", r"horizontal", r"oblicu", r"complej", r"desplaz", r"extru",
            r"no se (identifica|visualiza|reconoce)", r"ausencia de", r"longitudinal", r"vertical", r"periferic", r"macerac"],
    mild=[r"degenerat", r"mucoide", r"mixoide", r"esguince", r"distension", r"edema", r"engros", r"intrasustancia",
          r"intrasubstan", r"grado (1|i|2|ii)\b", r"alteracion de (la )?senal", r"aumento de senal", r"hipersenal",
          r"hiperintens", r"ganglion", r"quiste", r"deshilach", r"laxitud", r"sobrecarga", r"irregular", r"heterogen", r"parcial"],
    negators=[r"\bno\b", r"\bsin\b", r"ausencia", r"ausente", r"normal", r"intact", r"conserv", r"preserv", r"integr",
              r"continu", r"indemne", r"ni\b", r"descarta", r"adecuad", r"correct", r"habitual", r"tampoco",
              r"regular", r"fisiologic", r"morfologia y senal", r"dentro de (los )?limites", r"anodin"],
    sev_mild=[r"leve", r"minim", r"pequen", r"liger", r"discret", r"incipiente", r"grado (1|i|2|ii)\b", r"bajo grado",
              r"escas", r"sutil", r"superficial", r"mucoide", r"intrasustancia", r"fisiologic", r"minusc", r"tenue", r"laminar"],
    sev_moderate=[r"moderad", r"parcial", r"intermedi"],
    sev_severe=[r"grande", r"sever", r"important", r"extens", r"marcad", r"complet", r"avanzad", r"alto grado",
                r"grado (3|iii|4|iv)\b", r"significativ", r"abundante", r"masiv", r"acentuad", r"intens",
                r"espesor (total|completo)", r"voluminos", r"amplio", r"gran\b", r"franca?\b"],
    stopwords={"de", "la", "el", "los", "las", "del", "en", "y", "o", "a", "con", "por", "para", "que", "se", "es",
               "son", "un", "una", "al", "su", "sus", "lo", "e", "u", "hay"},
    contrast={"pero", "aunque", "salvo", "excepto", "embargo"},
    left=[r"izquierd"], right=[r"derech"],
    postop=_SHARED_POSTOP + [r"cirugia", r"quirurgic", r"intervenid", r"operad", r"postquirurgic", r"posquirurgic", r"plastia", r"injerto"],
)

VOCAB["tr"] = LangVocab(
    terms={
        "acl": [r"ocb\b", r"on capraz", r"acl\b", r"anterior krusiat", r"onc\b", r"o\.c\.b"],
        "mcl": [r"mcl\b", r"iyb\b", r"ic yan bag", r"medial kol+ateral", r"tibial kol+ateral", r"ic kol+ateral", r"medial yan bag"],
        "med_men": [r"medial menisk", r"medyal menisk", r"ic menisk", r"medial menisc", r"mm\b", r"med[iy]al ve lateral menisk", r"medial-lateral menisk"],
        "lat_men": [r"lateral menisk", r"dis menisk", r"lateral menisc", r"lm\b", r"lateral ve med[iy]al menisk", r"lateral-medial menisk"],
        "effusion": [r"efuzyon", r"effuzyon", r"sivi", r"hidrartroz", r"hemartroz", r"mayi", r"efüzyon"],
        "synovitis": [r"sinovit", r"sinovyal (kalinlasma|proliferasyon|hipertrofi)", r"sinovial (kalinlasma|proliferasyon|hipertrofi)"],
        "baker": [r"baker", r"popliteal kist"],
        "contusion": [r"kontuzyon", r"kemik ilig\w* odem", r"kemik odem", r"bone bruise", r"kemik ilig\w* sinyal",
                      r"ilik odem", r"trabekuler (odem|kontuzyon)", r"kemik iliginde odem", r"kontuzyonel", r"medullar odem"],
        "fracture": [r"kirik", r"kirig", r"fraktur"],
        "med_oa": [r"medial (femorotibial|tibiofemoral|kompartman|eklem|femoral kondil|tibial plato|femoro|kompart|kondil|plato)",
                   r"ic kompartman", r"medyal (kompartman|femorotibial|kondil|plato)", r"medial(?! ?(menisk|menisc|kol+ateral|yan|retinak|plika|horn|patel|femoral kondil\w* (kemik|ilig)))"],
        "lat_oa": [r"lateral (femorotibial|tibiofemoral|kompartman|eklem|femoral kondil|tibial plato|femoro|kompart|kondil|plato)",
                   r"dis kompartman", r"lateral(?! ?(menisk|menisc|kol+ateral|yan|retinak|plika|horn|patel|tibial plato\w* (kemik|ilig)))"],
        "pf_oa": [r"patel+ofemoral", r"patel+afemoral", r"retropatel+ar", r"patel+a", r"troklea", r"femoropatel+ar"],
    },
    both={
        "menisci": (("med_men", "lat_men"), [r"menisk", r"menisc", r"her iki menisk"]),
        "cruciates": (("acl",), [r"capraz bag", r"krusiat", r"capraz"]),
        "collaterals": (("mcl",), [r"yan bag", r"kol+ateral"]),
        "ligaments": (("acl", "mcl"), [r"\bbag(lar|da|i|lari|larda)?\b", r"ligament"]),
        "cartilage": (("med_oa", "lat_oa", "pf_oa"), [r"kikirdak", r"kondral", r"eklem yuzey"]),
        "bone": (("contusion", "fracture"), [r"kemik ilig", r"kemik yapi", r"kemik sinyal", r"kemik", r"medull"]),
    },
    oa_cues=[r"gonartroz", r"osteoartr", r"artroz", r"kondromalazi", r"kondropati",
             r"kikirdak(ta)? (kaybi|incelme|hasari|defekt|dejeneras|duzensiz|erozyon|lezyon|yuzey|kayip|kalinlik kaybi|kalinliginda azalma)",
             r"osteofit", r"dejeneratif degisiklik", r"eklem araligi(nda)? daralma", r"subkondral (kist|skleroz|odem)",
             r"kondral (defekt|lezyon|kayip)", r"tam kat", r"incelme", r"fissur", r"fibrilasyon", r"kikirdak kaybi",
             r"dejeneratif", r"artrozik", r"osteokondral"],
    strong=[r"yirtik", r"yirtig", r"ruptur", r"kopuk", r"kopma", r"devamsiz", r"kesinti", r"parcal", r"avulsiyon",
            r"kova sapi", r"deplase", r"radyal", r"radial", r"horizontal", r"oblik", r"kompleks", r"ekstruz", r"lezyon",
            r"grade (3|iii)\b", r"evre 3\b", r"izlenmemesi", r"secilememesi", r"longitudinal", r"vertikal", r"periferik", r"hasar"],
    mild=[r"dejeneras", r"dejeneratif", r"mukoid", r"miksoid", r"odem", r"kalinlas", r"intrasubstans", r"grade (1|i|2|ii)\b",
          r"evre (1|2)\b", r"sinyal (artisi|degisikligi|artimi|intensite|artis)", r"hiperintens", r"ganglion", r"kist",
          r"strain", r"sprain", r"zorlanma", r"laksite", r"burkulma", r"parsiyel", r"heterojen", r"duzensiz", r"intrameniskal"],
    # Turkish negation is a verb suffix (-medi/-mamış/-miyor/-memekte/-mez): matched generically.
    negators=[r"normal", r"dogal", r"tabii", r"saglam", r"intakt", r"yok\b", r"yoktur", r"mevcut degil", r"olagan", r"korunmus",
              r"devamlilig", r"kontinu", r"degil", r"seyrinde", r"dikkati cekmedi", r"olmadigi", r"lehine bulgu",
              r"\w+m[ae]di\b", r"\w+m[ae]mis", r"\w+m[iu]yor\b", r"\w+m[ae]mekte", r"\w+m[ae]makta", r"\w+m[ae]z\b",
              r"\w+m[ae]yen\b", r"\w+m[ae]mistir", r"\w+m[ae]digi", r"\w+m[ae]ksizin", r"\w+siz\b", r"\w+suz\b"],
    sev_mild=[r"hafif", r"minimal", r"kucuk", r"\baz\b", r"grade (1|i|2|ii)\b", r"evre (1|2)\b", r"dusuk dereceli",
              r"mukoid", r"intrasubstans", r"fizyolojik", r"minik", r"ince\b", r"eser", r"hafif-orta", r"silik", r"parsiyel"],
    sev_moderate=[r"orta", r"belirgin"],
    sev_severe=[r"ileri", r"ciddi", r"yaygin", r"genis", r"tam kat", r"grade (3|iii|4|iv)\b", r"evre (3|4)\b", r"buyuk",
                r"masif", r"agir", r"yuksek dereceli", r"komplet", r"\btam\b", r"total", r"asiri", r"bol"],
    stopwords={"ve", "ile", "bir", "bu", "da", "de", "ise", "icin", "olan", "olarak", "ya", "veya", "gibi"},
    contrast={"ancak", "fakat", "ama"},
    left=[r"\bsol\b"], right=[r"\bsag\b"],
    postop=_SHARED_POSTOP + [r"ameliyat", r"opere", r"operasyon", r"cerrahi", r"tamir", r"onarim"],
)

VOCAB["hr"] = LangVocab(
    terms={
        "acl": [r"prednj\w* (krizn|ukrsten|kriz)", r"acl\b", r"lca\b", r"pkl\b"],
        "mcl": [r"medijaln\w* kolateraln", r"unutarnj\w* kolateraln", r"unutrasnj\w* (kolateraln|bocn)", r"mcl\b",
                r"tibijaln\w* kolateraln", r"medijaln\w* (postranicn|bocn)", r"lcm\b"],
        "med_men": [r"medijaln\w* menisk", r"unutarnj\w* menisk", r"unutrasnj\w* menisk", r"medijaln\w* i lateraln\w* menisk"],
        "lat_men": [r"lateraln\w* menisk", r"vanjsk\w* menisk", r"spoljasnj\w* menisk", r"lateraln\w* i medijaln\w* menisk"],
        "effusion": [r"izljev", r"izliv", r"tekucin", r"tecnost", r"hidrops", r"hemartro", r"efuzij"],
        "synovitis": [r"sinovit", r"sinovijaln\w* (zadebljanj|hipertrofij|proliferacij)"],
        "baker": [r"baker", r"poplitealn\w* cist", r"bakerov"],
        "contusion": [r"kontuzij", r"kostan\w* edem", r"edem\w* (kostane srzi|kosti|kostan|spongioz)", r"bone bruise",
                      r"kostano-?mozdan", r"kostane srzi", r"kontuzion", r"edem\w* srzi", r"trabekularn\w* edem", r"edem\w* subhondral"],
        "fracture": [r"fraktur", r"prijelom", r"prelom", r"infrakcij"],
        "med_oa": [r"medijaln\w* (kompartm|femorotibij|tibiofemor|kondil|plato|odjeljk|dijel)", r"unutarnj\w* (kompartm|femorotibij)",
                   r"medijaln\w* femoraln\w* kondil", r"medijaln\w* platoa? tibij", r"medijalno\b",
                   r"medijaln\w*(?! ?(menisk|kolateraln|retinak|plik|rog|patel|bocn|postran))"],
        "lat_oa": [r"lateraln\w* (kompartm|femorotibij|tibiofemor|kondil|plato|odjeljk|dijel)", r"vanjsk\w* (kompartm|femorotibij)",
                   r"lateraln\w* femoraln\w* kondil", r"lateraln\w* platoa? tibij", r"lateralno\b",
                   r"lateraln\w*(?! ?(menisk|kolateraln|retinak|plik|rog|patel|bocn|postran))"],
        "pf_oa": [r"patelofemoral", r"femoropatelar", r"retropatelar", r"patel", r"trohle", r"fp zglob", r"\bfp\b"],
    },
    both={
        "menisci": (("med_men", "lat_men"), [r"menisk", r"meniska\b", r"oba meniska", r"obadva menisk"]),
        "cruciates": (("acl",), [r"krizn", r"ukrsten"]),
        "collaterals": (("mcl",), [r"kolateraln", r"postranicn", r"bocn\w* (ligament|veze)"]),
        "ligaments": (("acl", "mcl"), [r"ligament", r"veze\b", r"sveze", r"ligamenata"]),
        "cartilage": (("med_oa", "lat_oa", "pf_oa"), [r"hrskavic", r"hondraln", r"zglobn\w* ploh"]),
        "bone": (("contusion", "fracture"), [r"kostan\w* (struktur|srz)", r"kostane srzi", r"kosti\b", r"kostan\w* signal", r"kostanih struktura", r"kostan", r"kost\w*"]),
    },
    oa_cues=[r"hondromalacij", r"hondropatij", r"gonartroz", r"artroz", r"osteoartr", r"osteofit", r"degenerativn\w* promjen",
             r"stanjen\w* hrskavic", r"hrskavic\w* .{0,25}(stanjen|istanjen|osteceni|defekt|erozij|fisur|denud|nepravil|ostecen|degenerat)",
             r"defekt\w* (pune debljine|hrskavic)", r"pune debljine", r"suzenj\w* zglobn", r"subhondraln\w* (cist|skleroz)",
             r"denudacij", r"ostecenj\w* hrskavic", r"fibrilacij", r"hondraln", r"deg\.? ?promj", r"artrotsk", r"osteohondraln",
             r"stanjen", r"istanjen", r"koštane apozicij", r"kostan\w* apozicij", r"rubn\w* (osteofit|apozicij)"],
    strong=[r"ruptur", r"puknuc", r"prekid", r"diskontinu", r"lezij", r"otrgnu", r"avulzij", r"fragment", r"drska kosare",
            r"radijaln", r"horizontaln", r"kos[aeiou]\b", r"kompleksn", r"dislocir", r"ekstruz", r"puknut", r"rascjep", r"laceracij",
            r"kidanj", r"longitudinaln", r"vertikaln", r"periferno", r"ozljed", r"povred", r"nije prikazan", r"ne prikazuje se"],
    mild=[r"degenerat", r"mukoidn", r"miksoidn", r"distenzij", r"distorzij", r"edem", r"zadeblj", r"intrasupstan", r"intrasubstan",
          r"(i|ii)\.? ?(-|do)? ?(ii\.? )?stupnj", r"(i|ii)°", r"promijenjen\w* (intenzitet|signal)", r"povisen\w* (intenzitet|signal)",
          r"hiperintenz", r"ganglion", r"cist", r"uganuc", r"istegnu", r"labav", r"nehomogen", r"inhomogen", r"promjen\w* (intenziteta )?signal",
          r"signal\w* .{0,10}(promijenjen|povisen)", r"parcijaln", r"deg\.", r"degenerativn", r"zadebljan"],
    negators=[r"\bbez\b", r"\bne\b", r"nema\b", r"nije\b", r"nisu\b", r"uredn", r"uredan", r"normaln", r"intakt", r"ocuvan",
              r"odrzan", r"primjeren", r"kontinuitet", r"bez znakova", r"bez vidljiv", r"fizioloski", r"cjelovit", r"neostecen",
              r"bez osobitosti", r"b\.o\.", r"pravilan", r"pravilnog", r"pravilne", r"redovit", r"sacuvan", r"nema znakova",
              r"ne prikazuj", r"ne vid", r"ne nalaz", r"nikakv", r"niti\b", r"\bni\b", r"urednog", r"uredne", r"uredni",
              r"ocuvanog", r"homogen", r"prati se u kontinuitetu", r"se prati", r"kontinuiran"],
    sev_mild=[r"blag", r"blaz", r"diskretn", r"manj", r"minimaln", r"mal[aeio]\b", r"pocetn", r"inicijaln", r"(i|ii)\.? ?stupnj",
              r"(i|ii)°", r"i-ii", r"nezn", r"mukoidn", r"intrasupst", r"fiziolos", r"lagan", r"sitn", r"manje", r"malen", r"minim", r"parcijaln"],
    sev_moderate=[r"umjeren", r"srednj"],
    sev_severe=[r"izrazen", r"opsezn", r"uznapredoval", r"potpun", r"vec[ai]\b", r"(iii|iv)\.? ?stupnj", r"(iii|iv)°", r"iii-iv",
                r"znatn", r"obiln", r"velik", r"tesk", r"kompletn", r"pune debljine", r"masivn", r"visok\w* stupnj", r"jak[ao]?\b",
                r"izrazit", r"gotovo potpun", r"total"],
    stopwords={"i", "u", "na", "je", "su", "se", "sa", "s", "od", "do", "za", "te", "ili", "a", "da", "kao", "uz", "o", "po", "pri", "ka"},
    contrast={"ali", "no", "medutim", "osim", "vec"},
    left=[r"lijev", r"lev[aeio]g?\b"], right=[r"desn"],
    postop=_SHARED_POSTOP + [r"operac", r"operiran", r"postoperativn", r"kirurs"],
)

VOCAB["el"] = LangVocab(
    terms={
        "acl": [r"προσθι\w* χιαστ", r"πχσ\b", r"acl\b", r"προσθιου χιαστου"],
        "mcl": [r"εσω πλαγι", r"επσ\b", r"mcl\b", r"κνημιαι\w* πλαγι", r"εσω (κνημιαι|επικουρικ)"],
        "med_men": [r"εσω μηνισκ", r"εσωτερικ\w* μηνισκ", r"εσω και εξω μηνισκ"],
        "lat_men": [r"εξω μηνισκ", r"εξωτερικ\w* μηνισκ", r"εξω και εσω μηνισκ"],
        "effusion": [r"συλλογ\w* υγρου", r"συλλογη", r"ενδαρθρικ\w* συλλογ", r"υδραρθρ", r"αιμαρθρ", r"ενδαρθρικ\w* υγρ",
                     r"συλλογ", r"υγρου", r"υγρο\b"],
        "synovitis": [r"υμενιτιδ", r"υμενιτισ", r"υμενικ\w* (παχυνσ|υπερτροφ|υπερπλασ)", r"synovitis", r"σινοβιτ", r"αρθροσυνοβιτ"],
        "baker": [r"baker", r"μπεικερ", r"ιγνυακ\w* κυστ", r"κυστ\w* baker", r"κυστη του baker"],
        "contusion": [r"θλασ", r"οστικ\w* οιδημ", r"οιδημ\w* (του )?(οστικου )?μυελου", r"οιδημα μυελου", r"bone bruise",
                      r"οστεομυελικ\w* οιδημ", r"οστικο οιδημα", r"οιδημ\w* .{0,12}(μυελ|οστ)", r"μυελικ\w* οιδημ"],
        "fracture": [r"καταγμ", r"ρωγμ\w* .{0,10}(οστ|κατ)", r"καταγμα"],
        "med_oa": [r"εσω (διαμερισμ|μεσαρθρι|μηριαι|κνημιαι|κονδυλ|διαμερ)", r"εσω(?! ?(μηνισκ|πλαγι|καθεκτικ|κερα|επιγονατ))"],
        "lat_oa": [r"εξω (διαμερισμ|μεσαρθρι|μηριαι|κνημιαι|κονδυλ|διαμερ)", r"εξω(?! ?(μηνισκ|πλαγι|καθεκτικ|κερα|επιγονατ))"],
        "pf_oa": [r"επιγονατιδ", r"τροχιλ", r"προσθι\w* διαμερισμ"],
    },
    both={
        "menisci": (("med_men", "lat_men"), [r"μηνισκ"]),
        "cruciates": (("acl",), [r"χιαστ"]),
        "collaterals": (("mcl",), [r"πλαγι\w* συνδεσμ", r"πλαγιοι", r"πλαγιων", r"πλαγιους"]),
        "ligaments": (("acl", "mcl"), [r"συνδεσμ"]),
        "cartilage": (("med_oa", "lat_oa", "pf_oa"), [r"χονδρ"]),
        "bone": (("contusion", "fracture"), [r"οστικ\w* μυελ", r"μυελ", r"οστ[αω]ν?\b", r"οστικ\w* σημα", r"οστα\b", r"οστικ", r"οστεομυελ"]),
    },
    oa_cues=[r"οστεοαρθρ", r"αρθριτιδ", r"χονδροπαθ", r"χονδρομαλακ", r"οστεοφυτ", r"διαβρωσ", r"λεπτυνσ",
             r"απωλει\w* .{0,15}χονδρ", r"εκφυλι\w* .{0,20}χονδρ", r"χονδρ\w* .{0,25}(διαβρωσ|λεπτυνσ|απωλει|ελλειμ|εκφυλι|βλαβ|φθορ|ρωγμ|ανωμαλ)",
             r"ολικ\w* παχ", r"μερικ\w* παχ", r"εκφυλιστικ\w* (αλλοιωσ|οστεοαρθρ|μεταβολ)", r"στενωσ\w* .{0,15}μεσαρθρ",
             r"υποχονδρι\w* (κυστ|σκληρυνσ|οιδημ)", r"γοναρθρ", r"φθορ", r"icrs", r"χονδρικ", r"οστεοχονδρ", r"χονδρινη"],
    strong=[r"ρηξ", r"ασυνεχ", r"διακοπ", r"αποσπασ", r"κατακερματ", r"λαβη κουβα", r"οριζοντι", r"ακτινωτ", r"λοξ",
            r"συνθετ", r"μετατοπι", r"εξαρθρ", r"εξωθησ", r"βλαβ", r"τραυματ", r"κακωσ", r"δεν αναγνωριζ", r"δεν απεικονιζ",
            r"επιμηκ", r"καθετ", r"περιφερικ", r"θρυμματ", r"απουσι\w* (του )?(προσθι|συνδεσμ)"],
    mild=[r"εκφυλι", r"βλεννοειδ", r"μυξοειδ", r"διαστρεμμ", r"οιδημ", r"παχυνσ", r"ενδομηνισκ", r"ενδοουσ", r"βαθμου (1|i|2|ii)\b",
          r"(1|2)ου βαθμ", r"παθολογικ\w* σημα", r"αυξημεν\w* (σημα|εντασ)", r"γαγγλι", r"κυστ", r"διατασ", r"χαλαρ", r"ανομοιογεν",
          r"σηματ", r"σημα\b", r"μερικ\w* ρηξ", r"ετερογεν", r"ανωμαλ"],
    negators=[r"\bδεν\b", r"χωρις", r"απουσια", r"φυσιολογικ", r"ακεραι", r"ουδεμι", r"ουδεν", r"διατηρ", r"ανευ", r"απουσ",
              r"ουτε", r"\bμη\b", r"οχι", r"συνεχει", r"κατα φυση", r"\bκφ\b", r"αρνητικ", r"ελευθερ", r"υγιη", r"ομαλ",
              r"εντος (του )?φυσιολογικου", r"φ\.?κ\.?\b", r"δε\b", r"χωρις ευρηματα", r"χωρις ιδιαιτερα", r"φυσιολογικα"],
    sev_mild=[r"ηπι", r"μικρ", r"ελαχιστ", r"διακριτ", r"αρχομεν", r"αρχικ", r"βαθμου (1|i|2|ii)\b", r"(1|2)ου βαθμ",
              r"χαμηλ\w* βαθμ", r"επιφανειακ", r"μερικ\w* παχ", r"ενδομηνισκ", r"ενδοουσιακ", r"βλεννοειδ", r"φυσιολογικ",
              r"αμυδρ", r"οριακ", r"μικρου βαθμου", r"ελασσον", r"ισχν", r"μερικ"],
    sev_moderate=[r"μετρι", r"ικαν\w* ποσοτ"],
    sev_severe=[r"εκτεταμ", r"σοβαρ", r"μεγαλ", r"ευμεγεθ", r"ολικ", r"πληρ", r"προχωρημ", r"σημαντικ", r"αξιολογ",
                r"βαθμου (3|iii|4|iv)\b", r"(3|4)ου βαθμ", r"υψηλ\w* βαθμ", r"εντον", r"μαζικ", r"αδρ", r"μεγαλου βαθμου",
                r"μειζον", r"σαφ", r"icrs (iii|iv|3|4)", r"αφθον"],
    stopwords={"η", "ο", "το", "τα", "του", "της", "των", "και", "με", "σε", "στο", "στη", "στην", "στον", "απο", "για", "που",
               "οι", "ειναι", "ενα", "μια", "τον", "την", "τησ", "τουσ", "στα", "στισ", "στουσ", "τισ", "τι", "ως"},
    contrast={"αλλα", "ομως", "ωστοσο", "εκτος", "παρα"},
    left=[r"αριστερ"], right=[r"δεξι"],
    postop=_SHARED_POSTOP + [r"μετεγχειρητ", r"μεταεγχειρητ", r"μοσχευμ", r"πλαστικ", r"μηνισκεκτομ", r"αρθροσκοπ", r"χειρουργ", r"εγχειρ"],
)

VOCAB["bg"] = LangVocab(
    terms={
        "acl": [r"предн\w* кръстн", r"предн\w* кръстосан", r"пкв\b", r"acl\b", r"пкс\b", r"предната кръстна"],
        "mcl": [r"медиал\w* колатерал", r"вътрешн\w* (страничн|колатерал)", r"мкв\b", r"mcl\b", r"тибиал\w* колатерал", r"вътрешната странична"],
        "med_men": [r"медиал\w* мениск", r"вътрешн\w* мениск", r"медиал\w* и латерал\w* мениск"],
        "lat_men": [r"латерал\w* мениск", r"външн\w* мениск", r"латерал\w* и медиал\w* мениск"],
        "effusion": [r"излив", r"ставна течност", r"течност в ставата", r"хидрартроз", r"хемартроз", r"вътреставн\w* течност",
                     r"количество течност", r"течност"],
        "synovitis": [r"синовит", r"синовиал\w* (задебел|хипертроф|пролифер)"],
        "baker": [r"бейкър", r"бекер", r"baker", r"поплитеал\w* кист"],
        "contusion": [r"контуз", r"костно-?мозъч\w* (едем|оток)", r"костномозъч\w* (едем|оток)", r"едем на костния мозък",
                      r"bone bruise", r"оток на костния мозък", r"трабекулар\w* (едем|оток)", r"костен едем", r"едем\w* .{0,12}костн",
                      r"костномозъчен", r"костно-мозъчен", r"едем в .{0,25}(кондил|плато|патела)"],
        "fracture": [r"фрактур", r"счупван", r"счупен", r"инфракц"],
        "med_oa": [r"медиал\w* (компартм|отдел|фемуро-?тибиал|феморо-?тибиал|кондил|плато|страна|част|тибиофемор)", r"медианн\w* гонартроз",
                   r"медиал\w*(?! ?(мениск|колатерал|странич|ретинак|плик|рог|пател|фемуропател))"],
        "lat_oa": [r"латерал\w* (компартм|отдел|фемуро-?тибиал|феморо-?тибиал|кондил|плато|страна|част|тибиофемор)", r"латералн\w* гонартроз",
                   r"латерал\w*(?! ?(мениск|колатерал|странич|ретинак|плик|рог|пател))"],
        "pf_oa": [r"пателофемор", r"феморопател", r"ретропател", r"патела", r"трохле", r"капачк", r"пателата", r"пателарн"],
    },
    both={
        "menisci": (("med_men", "lat_men"), [r"мениск"]),
        "cruciates": (("acl",), [r"кръстн", r"кръстосан"]),
        "collaterals": (("mcl",), [r"колатерал", r"страничн\w* връзк"]),
        "ligaments": (("acl", "mcl"), [r"връзк", r"лигамент"]),
        "cartilage": (("med_oa", "lat_oa", "pf_oa"), [r"хрущял", r"хондрал", r"ставн\w* повърхност"]),
        "bone": (("contusion", "fracture"), [r"костн\w* мозък", r"костния мозък", r"кости\w*\b", r"костн\w* структур", r"костите", r"костен", r"кост\w*"]),
    },
    oa_cues=[r"гонартроз", r"артроз", r"остеоартр", r"хондромалац", r"хондропат", r"остеофит", r"изтън\w* (хрущял|ставн)",
             r"хрущял\w* .{0,30}(изтън|дефект|ерози|увред|дегенер|нарушен|неравн|загуб|липс|фисур|разкъс)", r"дегенеративн\w* промен",
             r"стеснен\w* (ставн|междуставн)", r"субхондрал\w* (кист|склероз)", r"пълна дебелина", r"цяла дебелина", r"хондрал\w* (дефект|лезия)",
             r"остеохондрал\w* (лези|дефект)", r"изтънен", r"изтъняване", r"фибрилац", r"артрозн", r"дегенеративни"],
    strong=[r"скъсван", r"руптур", r"разкъсван", r"прекъсван", r"прекъснат", r"дисконтину", r"лезия", r"увреда", r"увреждан",
            r"фрагмент", r"дръжка на кофа", r"радиал", r"хоризонтал", r"комплексн", r"дислоц", r"екструз", r"откъсван", r"авулз",
            r"скъсан", r"разкъсан", r"нарушена (цялост|непрекъснатост)", r"надлъжн", r"вертикал", r"перифер", r"травматичн"],
    mild=[r"дегенератив", r"мукоидн", r"миксоидн", r"дисторз", r"навяхван", r"едем", r"оток", r"задебел", r"интрасубстанц",
          r"вътресубстанц", r"(1|2|i|ii)\s?(-та|-ра|ва|ра)? ?степен", r"степен (1|2|i|ii)\b", r"сигнал", r"хиперинтенз",
          r"повишен сигнал", r"ганглион", r"кист", r"разтегл", r"хлабав", r"нехомоген", r"частичн", r"хетероген"],
    negators=[r"\bне\b", r"\bбез\b", r"\bняма\b", r"липс", r"нормал", r"интактн", r"запазен", r"съхранен", r"(?<!нарушена )(?<!нарушение на )цялост",
              r"без данни", r"няма данни", r"не се (установ|визуализ|намир|доказ)", r"без особености", r"физиологич", r"непроменен",
              r"нито", r"неувреден", r"здрав", r"редовн", r"правилн", r"обичайн", r"в норма", r"интактен", r"хомогенн", r"непрекъснат"],
    sev_mild=[r"лек", r"малк", r"минимал", r"дискрет", r"начал", r"незначител", r"степен (1|2|i|ii)\b", r"(1|2|i|ii)[- ]?(ва|ра|та)? ?степен",
              r"ниска степен", r"повърхност", r"частичн\w* дебелина", r"мукоидн", r"физиологич", r"слаб", r"дребн", r"неголям", r"ограничен", r"частичн"],
    sev_moderate=[r"умерен", r"среден", r"средна"],
    sev_severe=[r"изразен", r"голям", r"тежк", r"значителн", r"пълн", r"напреднал", r"обширн", r"степен (3|4|iii|iv)\b",
                r"(3|4|iii|iv)[- ]?(та|ва)? ?степен", r"висока степен", r"масивн", r"цяла дебелина", r"пълна дебелина", r"обилн",
                r"груб", r"комплетн", r"тотал", r"силно", r"изявен"],
    stopwords={"и", "в", "на", "с", "за", "от", "по", "при", "към", "до", "се", "е", "са", "да", "или", "а", "че", "със", "у", "мр"},
    contrast={"но", "обаче", "освен"},
    left=[r"\bляв", r"\bлев", r"лявата", r"лява"], right=[r"\bдесн", r"\bдясн", r"дясната", r"дясна"],
    postop=_SHARED_POSTOP + [r"оператив", r"реконструкц", r"следоператив", r"постоператив", r"опериран", r"хирург", r"артроскоп", r"пластик"],
)

VOCAB["de"] = LangVocab(
    terms={
        "acl": [r"vorder\w* kreuzband", r"vkb\b", r"acl\b", r"lca\b", r"vorderes kb", r"vkb-"],
        "mcl": [r"innenband", r"medial\w* (kollateralband|seitenband|kollateral)", r"mcl\b", r"tibial\w* kollateral",
                r"innenseitenband", r"lig\.? collaterale mediale", r"ligamentum collaterale mediale", r"mediales seitenband", r"innenbandes"],
        "med_men": [r"innenmenisk", r"innenmenisc", r"medial\w* menisk", r"medial\w* menisc", r"\bimhh\b", r"\bim-hh\b", r"\bim\b", r"innen- und aussenmenisk", r"medial\w* und lateral\w* menisk"],
        "lat_men": [r"aussenmenisk", r"aussenmenisc", r"lateral\w* menisk", r"lateral\w* menisc", r"\bamhh\b", r"\bam-hh\b", r"aussen- und innenmenisk", r"lateral\w* und medial\w* menisk"],
        "effusion": [r"\w*erguss", r"gelenkfluss", r"hamarthros", r"hydrops", r"flussigkeit", r"lipohamarthros"],
        "synovitis": [r"\w*synovitis", r"\w*synovialitis", r"synovial\w* (verdickung|proliferation|hypertrophie)", r"synovialisverdick"],
        "baker": [r"baker", r"popliteal\w* zyst", r"poplitealzyst", r"kniekehlenzyst"],
        "contusion": [r"kontusion", r"bone bruise", r"knochenodem", r"knochenmarkodem", r"knochenmarksodem", r"markodem", r"knochenmark\w*odem",
                      r"spongiosaodem", r"ossar\w* odem", r"intraossar\w* (signal|odem)", r"bone-?bruise", r"knochenprellung", r"markraumodem", r"knochenmarks?-?odem"],
        "fracture": [r"\w*fraktur", r"infraktion", r"knochenbruch", r"fissur\w* .{0,10}(kortikal|knoch|tibia|femur)", r"impaktier", r"impression"],
        "med_oa": [r"medial\w* (kompartiment|kompartment|femorotibial|femoro-?tibial|tibiofemoral|gelenkabschnitt|gelenkspalt|femurkondyl|femurcondyl|tibiaplateau|kondyl|condyl|gelenk)",
                   r"innenseitig", r"medialseitig", r"medial\w*(?! ?(menisk|menisc|kollateral|seitenband|retinak|plica|hinterhorn|vorderhorn|patell|band))"],
        "lat_oa": [r"lateral\w* (kompartiment|kompartment|femorotibial|femoro-?tibial|tibiofemoral|gelenkabschnitt|gelenkspalt|femurkondyl|femurcondyl|tibiaplateau|kondyl|condyl|gelenk)",
                   r"aussenseitig", r"lateralseitig", r"lateral\w*(?! ?(menisk|menisc|kollateral|seitenband|retinak|plica|hinterhorn|vorderhorn|patell|band))"],
        "pf_oa": [r"retropatellar", r"femoropatellar", r"patellofemoral", r"patella", r"trochlea", r"kniescheibe", r"femoro-?patellar", r"patellar\w* (knorpel|gelenk|facette)", r"patellarucklfache"],
    },
    both={
        "menisci": (("med_men", "lat_men"), [r"menisk", r"menisc", r"beide menisk"]),
        "cruciates": (("acl",), [r"kreuzband", r"kreuzbander"]),
        "collaterals": (("mcl",), [r"kollateralband", r"seitenband", r"kollateralbander", r"seitenbander", r"kollateral"]),
        "ligaments": (("acl", "mcl"), [r"bandapparat", r"bander\b", r"\bband\b", r"ligament", r"bandstrukturen"]),
        "cartilage": (("med_oa", "lat_oa", "pf_oa"), [r"knorpel", r"chondral", r"gelenkflachen"]),
        "bone": (("contusion", "fracture"), [r"knochenmark", r"knochern\w* struktur", r"knochen", r"spongiosa", r"ossar", r"markraum", r"markraumsignal", r"skelett"]),
    },
    oa_cues=[r"\w*arthrose", r"osteoarthr", r"\w*chondropathie", r"chondromalazie",
             r"knorpel\w*(schaden|defekt|verlust|ausdunn|verschmaler|lasion|irregular|fissur|ulzer|glatze|abrieb|degenerat|verdunn|unregelm|riss|delamin|erosion|schad)",
             r"knorpel\w* (schaden|defekt|verlust|ausdunn|verschmal|lasion|irregular|fissur|ulzer|glatze|abrieb|degenerat|verdunn|unregelm|riss|delamin|erosion|aufgebraucht|aufgehoben)",
             r"osteophyt", r"vollschichtig", r"degenerativ\w* verander", r"gelenkspaltverschmaler", r"subchondral\w* (zyst|sklerose|geode|odem)",
             r"geoden", r"chondral\w* (defekt|lasion)", r"arthrot", r"retropatellararthrose", r"ausdunnung", r"knorpelbelag\w* .{0,20}(ausdunn|verdunn|unregelm|defekt|schaden|aufgebraucht)",
             r"chondropathielinie", r"knorpelaufbrauch", r"aufgebraucht", r"verschmalerung des gelenkspalt", r"randzack", r"osteochondral"],
    strong=[r"riss", r"ruptur", r"eingerissen", r"zerrissen", r"durchtrenn", r"kontinuitatsunterbrech", r"diskontinu", r"lasion", r"korbhenkel",
            r"radiar", r"horizontal", r"schrag", r"komplex", r"disloz", r"extrusion", r"extrudier", r"ausriss", r"abriss", r"avulsion",
            r"fragment", r"laesion", r"lappenriss", r"einriss", r"anriss", r"partialruptur", r"teilruptur", r"verletzung", r"gerissen",
            r"nicht (mehr )?abgrenzbar", r"nicht (mehr )?nachweisbar", r"vertikal", r"longitudinal", r"peripher", r"basisnah", r"grad (3|iii)\b"],
    mild=[r"degenerat", r"mukoid", r"myxoid", r"zerrung", r"distorsion", r"dehnung", r"odem", r"verdick", r"intrasubstan", r"grad (1|i|2|ii)\b",
          r"(erst|zweit)gradig", r"signal(anhebung|alteration|veranderung|erhohung|steigerung|storung|vermehrung|inhomogen)", r"hyperintens",
          r"ganglion", r"zyst", r"aufgetrieben", r"auffaser", r"elongiert", r"laxizitat", r"elongation", r"inhomogen", r"auflocker",
          r"teilweise", r"verdickt", r"partiell", r"mukoide", r"aufgefasert", r"signalreich"],
    negators=[r"kein\w*", r"nicht\b", r"ohne", r"unauffallig", r"regelrecht", r"intakt", r"normal", r"erhalten", r"unversehrt",
              r"durchgehend", r"kontinuierlich", r"abgrenzbar", r"physiologisch", r"kein anhalt", r"kein nachweis", r"ausschluss",
              r"ausgeschlossen", r"weder", r"noch\b", r"o\.b\.", r"o\.p\.b\.", r"altersentsprechend", r"altersgerecht", r"reizlos",
              r"glatt", r"regular", r"unauffalliger", r"homogen", r"zart", r"schlank", r"glatt begrenzt", r"ungestort"],
    sev_mild=[r"gering", r"leicht", r"diskret", r"klein", r"minimal", r"beginnend", r"angedeutet", r"grad (1|i|2|ii)\b", r"(erst|zweit)gradig",
              r"geringgradig", r"niedriggradig", r"leichtgradig", r"oberflachlich", r"initial", r"dezent", r"geringfugig", r"mukoid",
              r"physiologisch", r"flau", r"wenig", r"schmal", r"kleinvolumig", r"minim", r"partiell", r"inzipient", r"maessig-gering"],
    sev_moderate=[r"massig", r"maessig", r"mittelgradig", r"moderat", r"medium"],
    sev_severe=[r"deutlich", r"ausgepragt", r"hochgradig", r"vollschichtig", r"gross", r"komplett", r"fortgeschritten", r"drittgradig",
                r"viertgradig", r"grad (3|iii|4|iv)\b", r"erheblich", r"massiv", r"relevant", r"kraftig", r"stark", r"hohergradig",
                r"schwer", r"tief", r"flachig", r"vollstandig", r"vollschicht", r"betont", r"grossvolumig", r"reichlich", r"ausgedehnt", r"pangonarthrose", r"bis auf den knochen"],
    stopwords={"der", "die", "das", "des", "dem", "den", "und", "oder", "ein", "eine", "einer", "eines", "einem", "einen", "im", "in",
               "am", "an", "auf", "bei", "mit", "von", "zu", "zum", "zur", "ist", "sind", "bzw", "sowie", "auch", "sich", "es", "als"},
    contrast={"aber", "jedoch", "allerdings", "ausser", "wobei", "hingegen"},
    left=[r"\blink"], right=[r"\brecht(?!e ?(winkl|s ?winkl))"],
    postop=_SHARED_POSTOP + [r"operativ", r"z\.n\.", r"zustand nach", r"postoperativ", r"ersatzplastik", r"teilresektion", r"reseziert", r"naht"],
)

VOCAB["nl"] = LangVocab(
    terms={
        "acl": [r"voorste kruisband", r"vkb\b", r"acl\b", r"lca\b", r"voorste kruisbandplastiek"],
        "mcl": [r"mediaal collateraal", r"mediale collateraal", r"mcl\b", r"mediale band", r"binnenband", r"mediale collaterale",
                r"tibiaal collateraal", r"ligamentum collaterale mediale", r"mediaal ligament", r"mediale zijband"],
        "med_men": [r"mediale menisc", r"binnenmenisc", r"mediaal menisc", r"mediale en laterale menisc"],
        "lat_men": [r"laterale menisc", r"buitenmenisc", r"lateraal menisc", r"laterale en mediale menisc"],
        "effusion": [r"hydrops", r"gewrichtsvocht", r"effusie", r"vocht in het gewricht", r"gewrichtseffusie", r"hemartro", r"vochtcollectie",
                     r"opzetting van (de )?(suprapatellaire )?(recessus|bursa)", r"vocht\b", r"synoviaal vocht", r"intra-?articulair vocht", r"lipohemartro", r"uitgezette recessus", r"vochtuitstorting"],
        "synovitis": [r"synovitis", r"synoviale (verdikking|proliferatie|hypertrofie)", r"synovitis"],
        "baker": [r"baker", r"popliteale cyste", r"bakercyste", r"poplitea\w* cyst", r"baker-?cyst", r"kniekuil"],
        "contusion": [r"contusie", r"botoedeem", r"bot-?oedeem", r"beenmergoedeem", r"beenmerg-?oedeem", r"bone bruise", r"botkneuzing", r"kneuzing",
                      r"oedeem in het (bot|beenmerg)", r"medullair oedeem", r"trabeculair oedeem", r"botcontusie", r"beenmergoedeem", r"oedeem\w* .{0,15}(bot|beenmerg|condyl|plateau)"],
        "fracture": [r"\w*fractuur", r"fractur", r"breuk\b", r"infractie"],
        "med_oa": [r"mediaal (femorotibiaal|femoro-?tibiaal|compartiment|tibiofemoraal|femorotibiale|tibiaplateau)",
                   r"mediale (femorotibiale|compartiment|femurcondyl|tibiaplateau|condyl|femorotibiaal|femoro-?tibiale)", r"mediaal compartiment",
                   r"media(a)?l\w*(?! ?(menisc|collateraal|collaterale|band|retinac|plica|hoorn|patel|zijband|ligament))"],
        "lat_oa": [r"lateraal (femorotibiaal|femoro-?tibiaal|compartiment|tibiofemoraal|femorotibiale|tibiaplateau)",
                   r"laterale (femorotibiale|compartiment|femurcondyl|tibiaplateau|condyl|femorotibiaal|femoro-?tibiale)", r"lateraal compartiment",
                   r"latera(a)?l\w*(?! ?(menisc|collateraal|collaterale|band|banden|retinac|plica|hoorn|patel|zijband|ligament))"],
        "pf_oa": [r"patellofemor", r"femoropatell", r"retropatell", r"patella", r"trochlea", r"patellair", r"femoro-?patell"],
    },
    both={
        "menisci": (("med_men", "lat_men"), [r"menisc"]),
        "cruciates": (("acl",), [r"kruisband"]),
        "collaterals": (("mcl",), [r"collateraal", r"collaterale", r"laterale banden", r"zijband", r"collaterale banden", r"laterale band\b"]),
        "ligaments": (("acl", "mcl"), [r"ligament", r"banden\b", r"\bband\b", r"bandapparaat"]),
        "cartilage": (("med_oa", "lat_oa", "pf_oa"), [r"kraakbeen", r"chondra", r"gewrichtsoppervlak"]),
        "bone": (("contusion", "fracture"), [r"beenmerg", r"\bbot\b", r"botstructur", r"ossaal", r"ossale", r"botten", r"botsignaal", r"medullair", r"bot"]),
    },
    oa_cues=[r"\w*artrose", r"osteoartr", r"chondropathie", r"chondromalacie", r"kraakbeenlijden",
             r"kraakbeen\w*(verlies|defect|schade|letsel|beschadig|verdunning|fissuur|fissuren|erosie|degenerat|irregular|onregelm|ulcer|lijden)",
             r"kraakbeen\w* .{0,25}(verlies|defect|schade|letsel|beschadig|verdunning|fissuur|fissuren|erosie|degenerat|irregular|onregelm|ulcer|lijden|dun|afwezig)",
             r"osteofyt", r"degeneratieve verander", r"gewrichtsspleetvernauwing", r"subchondral\w* (cyst|sclerose|geode)", r"geode",
             r"chondraal (defect|letsel)", r"volledige dikte", r"full thickness", r"onregelmatig\w* (voorkomen )?kraakbeen", r"verdunning",
             r"fibrillatie", r"chondrale", r"kraakbeenafwijking", r"chondropathie", r"gonartrose", r"artrotisch"],
    strong=[r"scheur", r"ruptuur", r"gescheurd", r"geruptureerd", r"discontinu", r"onderbroken", r"onderbreking", r"letsel", r"laesie",
            r"fragment", r"hengselscheur", r"bucket", r"radiair", r"horizontaal", r"horizontale", r"schuin", r"complex", r"verplaats",
            r"gelux", r"extrusie", r"avulsie", r"afscheuring", r"uitgescheurd", r"beschadig", r"inscheuring", r"ruptur", r"verscheur",
            r"gestoorde continu", r"verticaal", r"verticale", r"longitudinaal", r"perifeer", r"perifere", r"niet (meer )?(zichtbaar|herkenbaar|aantoonbaar)"],
    mild=[r"degenerat", r"mucoid", r"myxoid", r"verstuiking", r"distorsie", r"oedeem", r"verdikking", r"verdikt", r"intrasubstant",
          r"graad (1|i|2|ii)\b", r"signaal(verhoging|toename|verandering|stijging|alteratie|afwijking)", r"gestegen (t2-?)?signaal",
          r"verhoogd\w* signaal", r"hyperintens", r"ganglion", r"cyste", r"rafelig", r"afstomp", r"laxiteit", r"inhomogeen", r"strain", r"sprain",
          r"signaalstoornis", r"elongatie", r"verstuikt", r"gezwollen", r"zwelling", r"intra-?substanti", r"partieel", r"partiele", r"heterogeen", r"onregelmatig"],
    negators=[r"\bgeen\b", r"\bniet\b", r"zonder", r"normaal", r"normale", r"intact", r"intacte", r"ongestoord", r"gaaf", r"gave\b",
              r"behouden", r"bewaard", r"onopvallend", r"geen aanwijzingen", r"geen tekenen", r"geen argumenten", r"afwezig",
              r"fysiologisch", r"continu", r"ononderbroken", r"uitgesloten", r"noch\b", r"zonder bijzonderheden", r"geen bijzonderheden",
              r"regelmatig", r"homogeen", r"onaangetast", r"vrij van", r"ongewijzigd", r"normaal voorkomen", r"nihil"],
    sev_mild=[r"licht", r"gering", r"klein", r"minimaal", r"minimale", r"discreet", r"discrete", r"beginnend", r"graad (1|i|2|ii)\b",
              r"laaggradig", r"oppervlakkig", r"partiele dikte", r"ondiep", r"subtiel", r"weinig", r"fysiologisch", r"spoor", r"miniem",
              r"beperkt", r"geringgradig", r"mild", r"lichte", r"geringe", r"kleine", r"beginnende", r"minieme", r"fijn"],
    sev_moderate=[r"matig", r"matige", r"middelmatig"],
    sev_severe=[r"uitgesproken", r"groot", r"grote", r"ernstig", r"ernstige", r"gevorderd", r"gevorderde", r"volledig", r"volledige",
                r"graad (3|iii|4|iv)\b", r"hooggradig", r"belangrijk", r"belangrijke", r"uitgebreid", r"uitgebreide", r"massief",
                r"aanzienlijk", r"diep", r"diepe", r"compleet", r"complete", r"forse", r"fors\b", r"duidelijk", r"duidelijke", r"beduidend",
                r"significant", r"omvangrijk", r"ruim", r"veel\b", r"opvallend", r"totaal", r"totale", r"transmura", r"full-?thickness", r"lokaal gevorderd"],
    stopwords={"de", "het", "een", "en", "of", "van", "in", "op", "met", "aan", "te", "is", "zijn", "er", "wordt", "worden", "dat",
               "die", "deze", "ter", "tot", "bij", "ook", "als", "naar", "door", "voor", "over"},
    contrast={"maar", "echter", "hoewel", "behalve", "doch"},
    left=[r"\blinks", r"\blinker"], right=[r"\brechts", r"\brechter"],
    postop=_SHARED_POSTOP + [r"operat", r"postoperat", r"heelkundig", r"chirurg", r"plastiek", r"geopereerd", r"status na"],
)

VOCAB["fr"] = LangVocab(
    terms={
        "acl": [r"ligament croise anterieur", r"lca\b", r"croise anterieur", r"acl\b", r"croise ant\b"],
        "mcl": [r"ligament collateral (interne|medial|tibial)", r"lcm\b", r"lli\b", r"collateral (interne|medial)", r"ligament lateral interne", r"lcm\b"],
        "med_men": [r"menisques? (interne|medial)", r"meniscal (interne|medial)", r"menisque int\b"],
        "lat_men": [r"menisques? (externe|lateral)", r"meniscal (externe|lateral)", r"menisque ext\b", r"menisques? (interne|medial) et (externe|lateral)"],
        "effusion": [r"epanchement", r"hydarthrose", r"hemarthrose", r"liquide (intra-?articulaire|articulaire)", r"lipohemarthrose",
                     r"lame liquidienne", r"distension .{0,15}(capsul|recess|cul-de-sac)", r"epanchement"],
        "synovitis": [r"synovite", r"epaississement synovial", r"synovial\w* (epaissi|hypertroph|prolifer)", r"synovitis"],
        "baker": [r"baker", r"kyste poplite", r"kyste popliteal"],
        "contusion": [r"contusion", r"oedeme osseux", r"oedeme (de la )?moelle", r"oedeme medullaire", r"oedeme intra-?osseux", r"bone bruise",
                      r"oedeme spongieux", r"contusion osseuse", r"oedeme de l'os", r"hypersignal .{0,20}(osseux|spongieux|medullaire)",
                      r"oedeme .{0,15}(condyle|plateau|rotule|patella)", r"oedeme trabeculaire", r"oedeme .{0,10}osseu"],
        "fracture": [r"fracture", r"fractur", r"trait de fracture"],
        "med_oa": [r"compartiment (interne|medial)", r"femoro-?tibial (interne|medial)", r"condyle (femoral )?(interne|medial)",
                   r"plateau tibial (interne|medial)", r"femorotibial (interne|medial)", r"tibio-?femoral (interne|medial)",
                   r"(interne|medial)(?! ?(menisc|collateral|retinac|plica|corne|rotul|patell))"],
        "lat_oa": [r"compartiment (externe|lateral)", r"femoro-?tibial (externe|lateral)", r"condyle (femoral )?(externe|lateral)",
                   r"plateau tibial (externe|lateral)", r"femorotibial (externe|lateral)", r"tibio-?femoral (externe|lateral)",
                   r"(externe|lateral)(?! ?(menisc|collateral|retinac|plica|corne|rotul|patell|interne))"],
        "pf_oa": [r"femoro-?patellaire", r"patello-?femoral", r"rotul", r"patella", r"trochle", r"retro-?patellaire", r"patellaire", r"femoropatellaire"],
    },
    both={
        "menisci": (("med_men", "lat_men"), [r"menisque", r"meniscal", r"menisques"]),
        "cruciates": (("acl",), [r"croises", r"croise", r"pivot central"]),
        "collaterals": (("mcl",), [r"collateraux", r"collateral", r"ligaments lateraux", r"lateraux"]),
        "ligaments": (("acl", "mcl"), [r"ligament"]),
        "cartilage": (("med_oa", "lat_oa", "pf_oa"), [r"cartilag", r"chondral", r"surfaces articulaires"]),
        "bone": (("contusion", "fracture"), [r"moelle osseuse", r"moelle", r"\bos\b", r"structures osseuses", r"signal osseux", r"osseux", r"osseuse", r"medullaire", r"spongieux"]),
    },
    oa_cues=[r"arthrose", r"gonarthrose", r"osteoarthr", r"chondropathie", r"chondromalacie", r"amincissement", r"perte (de )?cartilag",
             r"cartilage .{0,30}(aminc|perte|defect|fissur|irregul|ulcer|erosion|denud|lesion|degenerat|fibrillation|usure|mise a nu)",
             r"fissur\w* .{0,15}cartilag", r"osteophyt", r"pincement", r"remaniements? degeneratif", r"changements? degeneratif", r"pleine epaisseur",
             r"toute l'epaisseur", r"epaisseur complete", r"sclerose sous-?chondrale", r"geode", r"kyste sous-?chondral",
             r"chondral\w* (defect|lesion|perte|fissur)", r"lesion chondrale", r"fibrillation", r"usure", r"mise a nu", r"os a nu", r"denudation", r"chondrale", r"arthrosique"],
    strong=[r"dechirure", r"rupture", r"rompu", r"dechir", r"discontinu", r"interruption", r"interrompu", r"desinsertion", r"avulsion",
            r"fragment", r"anse de seau", r"radiaire", r"horizontal", r"oblique", r"complexe", r"deplac", r"extrusion", r"extru", r"lesion",
            r"fissur", r"languette", r"clivage", r"arrachement", r"luxation", r"longitudinal", r"vertical", r"peripheri", r"non (visible|visualis|individualis)"],
    mild=[r"degenerat", r"mucoid", r"myxoid", r"entorse", r"distension", r"oedeme", r"epaississement", r"epaissi", r"intra-?substance",
          r"intrameniscal", r"grade (1|i|2|ii)\b", r"hypersignal", r"anomalie de signal", r"signal (anormal|augment|intermediaire)", r"ganglion",
          r"kyste", r"effiloch", r"laxite", r"distendu", r"heterogene", r"remaniement", r"elongation", r"remanie", r"partiel", r"irregul"],
    negators=[r"\bpas\b", r"\bsans\b", r"absence", r"aucun", r"normal", r"normaux", r"normale", r"intact", r"integr", r"conserve",
              r"preserve", r"respecte", r"continu", r"sans particularite", r"sans anomalie", r"\bni\b", r"\bnon\b", r"physiologique",
              r"regulier", r"reguliere", r"indemne", r"homogene", r"habituel", r"sain\b", r"saine", r"bien (visible|individualis|conserve)",
              r"rien a signaler", r"\bras\b", r"n'est pas", r"n'y a pas", r"pas d'"],
    sev_mild=[r"leger", r"legere", r"petit", r"petite", r"minime", r"discret", r"discrete", r"debutant", r"debutante", r"grade (1|i|2|ii)\b",
              r"bas grade", r"superficiel", r"superficielle", r"partiel", r"partielle", r"intra-?substance", r"mucoid", r"physiologique",
              r"fine\b", r"fin\b", r"lame\b", r"faible", r"modique", r"peu\b", r"legerement", r"minimal"],
    sev_moderate=[r"modere", r"moderee", r"moyen", r"moyenne", r"intermediaire"],
    sev_severe=[r"important", r"importante", r"severe", r"etendu", r"etendue", r"complet", r"complete", r"avance", r"avancee",
                r"volumineu", r"abondant", r"abondante", r"marque", r"marquee", r"grade (3|iii|4|iv)\b", r"haut grade", r"pleine epaisseur",
                r"toute l'epaisseur", r"massif", r"massive", r"significati", r"majeur", r"majeure", r"profond", r"profonde", r"transfixiant",
                r"transfixiante", r"large\b", r"grande", r"grand\b", r"notable", r"franc\b", r"franche", r"total", r"totale"],
    stopwords={"le", "la", "les", "l", "de", "des", "du", "d", "un", "une", "et", "ou", "a", "au", "aux", "en", "dans", "sur", "avec",
               "par", "pour", "est", "sont", "il", "y", "ce", "cette", "ces", "son", "sa", "ses", "qui", "que", "n", "s"},
    contrast={"mais", "cependant", "toutefois", "sauf", "hormis"},
    left=[r"gauche"], right=[r"droit"],
    postop=_SHARED_POSTOP + [r"post-?operatoire", r"chirurgic", r"opere", r"ligamentoplastie", r"meniscectomie", r"plastie", r"greffe", r"suture"],
)

# ------------------------------------------------------------------ compiled form


@dataclass
class _Compiled:
    terms: dict[str, list[re.Pattern]]
    both: dict[str, tuple[tuple[str, ...], list[re.Pattern]]]
    oa_cues: list[re.Pattern]
    strong: list[re.Pattern]
    mild: list[re.Pattern]
    negators: list[re.Pattern]
    sev_mild: list[re.Pattern]
    sev_moderate: list[re.Pattern]
    sev_severe: list[re.Pattern]
    stopwords: set[str]
    contrast: set[str]
    left: list[re.Pattern]
    right: list[re.Pattern]
    postop: list[re.Pattern]


_COMPILED: dict[str, _Compiled] = {}


def _get(lang: str) -> _Compiled:
    lang = lang if lang in VOCAB else "en"
    if lang not in _COMPILED:
        v = VOCAB[lang]
        _COMPILED[lang] = _Compiled(
            terms={k: _compile(p) for k, p in v.terms.items()},
            both={g: (keys, _compile(p)) for g, (keys, p) in v.both.items()},
            oa_cues=_compile(v.oa_cues), strong=_compile(v.strong), mild=_compile(v.mild),
            negators=_compile(v.negators), sev_mild=_compile(v.sev_mild),
            sev_moderate=_compile(v.sev_moderate), sev_severe=_compile(v.sev_severe),
            stopwords=set(v.stopwords), contrast=set(v.contrast),
            left=_compile(v.left), right=_compile(v.right), postop=_compile(v.postop),
        )
    return _COMPILED[lang]


# ------------------------------------------------------------------ matching


class _Sentence:
    def __init__(self, text: str, voc: _Compiled):
        self.text = text
        self.voc = voc
        self.tokens = [(m.start(), m.end(), m.group()) for m in _TOKEN.finditer(text)]
        self.content = [i for i, (_, _, w) in enumerate(self.tokens) if w not in voc.stopwords]

    def _token_at(self, pos: int) -> int:
        for i, (s, e, _) in enumerate(self.tokens):
            if e > pos:
                return i
        return max(len(self.tokens) - 1, 0)

    def window_span(self, start: int, end: int, n: int) -> tuple[int, int]:
        """Char span covering n content tokens either side of [start, end), cut at contrast words."""
        if not self.tokens:
            return start, end
        first, last = self._token_at(start), self._token_at(max(end - 1, start))
        lo_i, hi_i = first, last
        count = 0
        for i in range(first - 1, -1, -1):
            w = self.tokens[i][2]
            if w in self.voc.contrast:
                break
            if w in self.voc.stopwords:
                lo_i = i
                continue
            lo_i = i
            count += 1
            if count >= n:
                break
        count = 0
        for i in range(last + 1, len(self.tokens)):
            w = self.tokens[i][2]
            if w in self.voc.contrast:
                break
            if w in self.voc.stopwords:
                hi_i = i
                continue
            hi_i = i
            count += 1
            if count >= n:
                break
        return self.tokens[lo_i][0], self.tokens[hi_i][1]

    def any_in(self, patterns: list[re.Pattern], lo: int, hi: int) -> bool:
        seg = self.text[lo:hi]
        return any(p.search(seg) for p in patterns)

    def negated(self, start: int, end: int) -> bool:
        lo, hi = self.window_span(start, end, WINDOW)
        return self.any_in(self.voc.negators, lo, hi)

    def severity(self, start: int, end: int) -> str:
        lo, hi = self.window_span(start, end, SEVERITY_WINDOW)
        if self.any_in(self.voc.sev_mild, lo, hi):
            return "mild"
        if self.any_in(self.voc.sev_severe, lo, hi):
            return "severe"
        if self.any_in(self.voc.sev_moderate, lo, hi):
            return "moderate"
        return "unspecified"


@dataclass
class KeyVote:
    state: str = "absent"
    severity: str = "none"
    evidence: str = ""
    mentioned: bool = False


_RANK = {"absent": 0, "negated": 1, "affirmed": 2}


def _update(votes: dict[str, KeyVote], key: str, state: str, severity: str, evidence: str) -> None:
    cur = votes[key]
    cur.mentioned = True
    if _RANK[state] > _RANK[cur.state] or (state == cur.state == "affirmed" and cur.severity == "mild" and severity != "mild"):
        votes[key] = KeyVote(state, severity, evidence[:120], True)


def _finding_state(sent: _Sentence, m: re.Match, voc: _Compiled) -> tuple[str, str]:
    if sent.negated(m.start(), m.end()):
        return "negated", "none"
    return "affirmed", sent.severity(m.start(), m.end())


def _struct_state(sent: _Sentence, m: re.Match, voc: _Compiled) -> tuple[str, str] | None:
    """Ligament/meniscus mention: needs a pathology cue to be affirmed."""
    lo, hi = sent.window_span(m.start(), m.end(), WINDOW + 3)
    strong = sent.any_in(voc.strong, lo, hi)
    mild = sent.any_in(voc.mild, lo, hi)
    if sent.negated(m.start(), m.end()):
        # "no tear, mild degeneration" style: a mild cue outside the negated span still counts.
        return "negated", "none"
    if strong:
        sev = sent.severity(m.start(), m.end())
        return "affirmed", sev
    if mild:
        return "affirmed", "mild"
    return None


def extract_rules_detail(report: str, lang: str) -> tuple[dict[str, KeyVote], dict[str, object]]:
    voc = _get(lang)
    votes = {k: KeyVote() for k in LABEL_KEYS}
    postop = False
    for raw in split_sentences(report):
        sent = _Sentence(raw, voc)
        if any(p.search(raw) for p in voc.postop):
            postop = True
        seen_side: set[str] = set()
        # finding labels and side-specific structures
        for key in FINDING_KEYS + STRUCT_KEYS:
            for pat in voc.terms[key]:
                for m in pat.finditer(raw):
                    if key in STRUCT_KEYS:
                        res = _struct_state(sent, m, voc)
                        seen_side.add(key)
                        if res is None:
                            votes[key].mentioned = True
                            continue
                        state, sev = res
                    else:
                        state, sev = _finding_state(sent, m, voc)
                    _update(votes, key, state, sev, raw)
        # OA: compartment + cue in the same sentence
        oa_cue_spans = [(m.start(), m.end()) for p in voc.oa_cues for m in p.finditer(raw)]
        oa_specific: set[str] = set()
        if oa_cue_spans:
            cue_negated = all(sent.negated(s, e) for s, e in oa_cue_spans)
            for key in OA_KEYS:
                for pat in voc.terms[key]:
                    for m in pat.finditer(raw):
                        oa_specific.add(key)
                        if sent.negated(m.start(), m.end()) or cue_negated:
                            _update(votes, key, "negated", "none", raw)
                        else:
                            sev = sent.severity(m.start(), m.end())
                            if sev == "unspecified":
                                sev = sent.severity(*oa_cue_spans[0])
                            _update(votes, key, "affirmed", sev, raw)
        # generic (both-sided) mentions: negation only, unless a pathology cue is present
        for group, (keys, pats) in voc.both.items():
            for pat in pats:
                for m in pat.finditer(raw):
                    # "normal medial meniscus" must not touch the lateral one: a group term only
                    # speaks for keys none of which was named specifically in this sentence.
                    if any(k in seen_side or k in oa_specific for k in keys):
                        continue
                    targets = list(keys)
                    neg = sent.negated(m.start(), m.end())
                    if group == "cartilage":
                        # generic cartilage statement: negation applies to all compartments;
                        # an un-negated OA cue without a compartment ("gonarthrosis") too.
                        if neg or (oa_cue_spans and all(sent.negated(s, e) for s, e in oa_cue_spans)):
                            for k in targets:
                                _update(votes, k, "negated", "none", raw)
                        elif oa_cue_spans:
                            sev = sent.severity(m.start(), m.end())
                            for k in targets:
                                _update(votes, k, "affirmed", sev, raw)
                        else:
                            for k in targets:
                                votes[k].mentioned = True
                        continue
                    if group == "bone":
                        if neg:
                            for k in targets:
                                _update(votes, k, "negated", "none", raw)
                        else:
                            for k in targets:
                                votes[k].mentioned = True
                        continue
                    res = _struct_state(sent, m, voc)
                    if res is None:
                        for k in targets:
                            votes[k].mentioned = True
                        continue
                    state, sev = res
                    if group == "ligaments" and state == "affirmed":
                        continue  # "ligament injury" without naming which one: too vague
                    for k in targets:
                        _update(votes, k, state, sev, raw)
    n_neg = sum(v.state == "negated" for v in votes.values())
    n_aff = sum(v.state == "affirmed" for v in votes.values())
    meta = {"postoperative": postop, "knee_side": knee_side(report, lang),
            "normal_template": n_aff == 0 and n_neg >= 8}
    return votes, meta


def extract_rules(report: str, lang: str) -> dict[str, str]:
    """State per label key: affirmed / negated / absent (architecture §3.1)."""
    votes, _ = extract_rules_detail(report, lang)
    return {k: v.state for k, v in votes.items()}


def knee_side(report: str, lang: str) -> str:
    voc = _get(lang)
    text = normalize(report)
    n_left = sum(len(p.findall(text)) for p in voc.left)
    n_right = sum(len(p.findall(text)) for p in voc.right)
    if n_left == n_right:
        return "unknown"
    return "left" if n_left > n_right else "right"


def mentions(report: str, lang: str) -> dict[str, bool]:
    votes, _ = extract_rules_detail(report, lang)
    return {k: v.mentioned for k, v in votes.items()}
