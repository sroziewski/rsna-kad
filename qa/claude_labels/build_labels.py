import json, sys

BASE = r"D:\dev\my\rsna_kad\qa\claude_labels"
batch_path = BASE + r"\batches\es_000.jsonl"
out_path = BASE + r"\out\es_000.jsonl"

reports = {}
order = []
with open(batch_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        reports[obj["StudyInstanceUID"]] = obj["report"]
        order.append(obj["StudyInstanceUID"])

def L(state, severity, evidence):
    return {"state": state, "severity": severity, "evidence": evidence}

def empty():
    return L("absent", "none", "")

records = {}

# ---------- Report 1 ----------
uid = "1.2.826.0.1.3680043.8.498.10095687747295410396510538520594649149"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "med_men": L("negated","none","Menisco medial de morfología y señal\nconservada, sin signos de rotura."),
        "lat_men": L("affirmed","unspecified","Amputación marginal del cuerpo del menisco lateral."),
        "med_oa": L("negated","none","Cartílagos de los compartimentos femorotibiales sin alteraciones."),
        "lat_oa": L("negated","none","Cartílagos de los compartimentos femorotibiales sin alteraciones."),
        "pf_oa": L("affirmed","severe","Fina úlcera condral focal de espesor total del aspecto inferior de la vertiente medial de la tróclea\nfemoral"),
        "effusion": L("affirmed","mild","Leve derrame articular."),
        "synovitis": empty(),
        "baker": L("negated","none","No hay quistes poplíteos patológicos."),
        "contusion": L("negated","none","No hay alteraciones de señal significativas de la médula ósea."),
        "fracture": empty(),
    }
}

# ---------- Report 2 ----------
uid = "1.2.826.0.1.3680043.8.498.10112440805412493412191181061985753636"
records[uid] = {
    "knee_side": "right", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "med_men": L("negated","none","Meniscos impresionan de morfología y señal conservada, sin signos de rotura."),
        "lat_men": L("negated","none","Meniscos impresionan de morfología y señal conservada, sin signos de rotura."),
        "med_oa": L("negated","none","El resto de los\ncartílagos femorotibiales y de la tróclea femoral sin alteraciones."),
        "lat_oa": L("affirmed","severe","Úlceras condrales de espesor parcial y prácticamente total del platillo tibial lateral."),
        "pf_oa": L("affirmed","mild","Úlceras condrales de espesor\nparcial de distribución difusa de la rótula."),
        "effusion": L("negated","none","No hay significativo derrame articular ni quistes poplíteos."),
        "synovitis": empty(),
        "baker": L("negated","none","No hay significativo derrame articular ni quistes poplíteos."),
        "contusion": L("negated","none","No hay otras alteraciones de señal de la médula ósea."),
        "fracture": empty(),
    }
}

# ---------- Report 3 ----------
uid = "1.2.826.0.1.3680043.8.498.10162720374291180196973998234794414415"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": empty(), "mcl": empty(),
        "med_men": L("affirmed","unspecified","Rotura de menisco interno."),
        "lat_men": empty(),
        "med_oa": empty(),
        "lat_oa": L("affirmed","unspecified","Lesión osteocondral en la meseta tibial lateral."),
        "pf_oa": empty(),
        "effusion": empty(), "synovitis": empty(), "baker": empty(), "contusion": empty(), "fracture": empty(),
    }
}

# ---------- Report 4 ----------
uid = "1.2.826.0.1.3680043.8.498.10189392162056736163704940747328177364"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colateral lateral sin alteraciones."),
        "mcl": L("affirmed","moderate","engrosamiento del componente superficial del ligamento colateral medial con pérdida\nparcial de la continuidad sus fibras"),
        "med_men": L("negated","none","Leve aumento de señal del meniscocapsular del cuerpo del menisco medial que no impresiona representar\nuna rotura."),
        "lat_men": L("negated","none","Menisco lateral de morfología y señal conservada, sin signos de rotura."),
        "med_oa": L("negated","none","Cartílagos de los compartimentos femorotibiales y patelofemorales sin alteraciones significativas."),
        "lat_oa": L("negated","none","Cartílagos de los compartimentos femorotibiales y patelofemorales sin alteraciones significativas."),
        "pf_oa": L("negated","none","Cartílagos de los compartimentos femorotibiales y patelofemorales sin alteraciones significativas."),
        "effusion": L("affirmed","mild","Leve derrame articular."),
        "synovitis": empty(),
        "baker": L("affirmed","mild","Pequeño quiste poplíteo con leve cantidad de líquido regional."),
        "contusion": L("negated","none","No hay alteraciones de señal significativas de la médula ósea."),
        "fracture": empty(),
    }
}

# ---------- Report 5 ----------
uid = "1.2.826.0.1.3680043.8.498.10252163031705285141538200795896549203"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "med_men": L("affirmed","mild","Aumento de señal central del cuerno posterior del menisco medial que no impresiona contactar\ncon la superficie articular"),
        "lat_men": L("negated","none","El menisco lateral impresiona de morfología y señal conservada, sin signos de rotura."),
        "med_oa": L("affirmed","mild","Úlceras condrales de espesor parcial de la región central del cóndilo femoral\nmedial."),
        "lat_oa": L("affirmed","severe","Úlceras condrales de espesor parcial y prácticamente total de la zona de carga del compartimento\nfemorotibial lateral."),
        "pf_oa": L("affirmed","severe","Úlceras condrales de espesor total de la faceta lateral de la rótula con cambios óseos\nsecundarios."),
        "effusion": L("negated","none","No hay signos que sugieran significativo derrame articular o quistes poplíteos."),
        "synovitis": empty(),
        "baker": L("negated","none","No hay signos que sugieran significativo derrame articular o quistes poplíteos."),
        "contusion": empty(),
        "fracture": empty(),
    }
}

# ---------- Report 6 ----------
uid = "1.2.826.0.1.3680043.8.498.10425131723456547056365404016184663035"
records[uid] = {
    "knee_side": "left", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "med_men": L("affirmed","mild","Aumento de señal central del del cuerpo, cuerno posterior del menisco medial que no impresiona\ncontactar con la"),
        "lat_men": L("affirmed","unspecified","Menisco lateral con rotura degenerativa marginal del cuerno anterior y parte del cuerpo, sin\ndesplazamiento"),
        "med_oa": L("affirmed","moderate","Úlceras condrales de espesor parcial de distribución difusa los compartimentos femorotibiales de\npredominio medial."),
        "lat_oa": L("affirmed","moderate","Úlceras condrales de espesor parcial de distribución difusa los compartimentos femorotibiales de\npredominio medial."),
        "pf_oa": L("affirmed","severe","Prácticamente ausencia de visualización del cartílago patelofemoral de distribución difusa con\nformaciones quísticas"),
        "effusion": L("affirmed","moderate","Moderado derrame articular con sinovitis difusa."),
        "synovitis": L("affirmed","unspecified","Moderado derrame articular con sinovitis difusa."),
        "baker": empty(),
        "contusion": empty(),
        "fracture": empty(),
    }
}

# ---------- Report 7 ----------
uid = "1.2.826.0.1.3680043.8.498.10574706940531650593705137704922433109"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": empty(), "mcl": empty(),
        "med_men": L("affirmed","unspecified","Rotura de menisco interno."),
        "lat_men": empty(),
        "med_oa": empty(),
        "lat_oa": empty(),
        "pf_oa": L("affirmed","unspecified","OA patelofemoral."),
        "effusion": L("affirmed","unspecified","Derrame."),
        "synovitis": empty(),
        "baker": L("affirmed","unspecified","Quiste de Baker."),
        "contusion": empty(), "fracture": empty(),
    }
}

# ---------- Report 8 ----------
uid = "1.2.826.0.1.3680043.8.498.10783501127779906316117667790784040770"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": empty(), "mcl": empty(),
        "med_men": L("affirmed","unspecified","Rotura de menisco interno."),
        "lat_men": empty(),
        "med_oa": L("affirmed","unspecified","OA tricompartimental."),
        "lat_oa": L("affirmed","unspecified","OA tricompartimental."),
        "pf_oa": L("affirmed","unspecified","OA tricompartimental."),
        "effusion": L("affirmed","unspecified","Derrame con sinovitis."),
        "synovitis": L("affirmed","unspecified","Derrame con sinovitis."),
        "baker": L("affirmed","unspecified","Quiste de Baker con fuga."),
        "contusion": empty(), "fracture": empty(),
    }
}

# ---------- Report 9 ----------
uid = "1.2.826.0.1.3680043.8.498.11129751904857491327872353316480050290"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("affirmed","mild","Degeneración mucoide de las fibras del LCA."),
        "mcl": empty(),
        "med_men": L("affirmed","unspecified","Rotura de menisco interno."),
        "lat_men": empty(),
        "med_oa": empty(), "lat_oa": empty(),
        "pf_oa": L("affirmed","mild","Pequeña lesión osteocondral en la línea media de la superficie rotuliana."),
        "effusion": empty(), "synovitis": empty(), "baker": empty(), "contusion": empty(), "fracture": empty(),
    }
}

# ---------- Report 10 ----------
uid = "1.2.826.0.1.3680043.8.498.11468812519969607938527116557104410121"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("affirmed","severe","Rotura completa de LCA."),
        "mcl": empty(),
        "med_men": empty(), "lat_men": empty(),
        "med_oa": empty(), "lat_oa": empty(), "pf_oa": empty(),
        "effusion": L("affirmed","unspecified","Derrame."),
        "synovitis": empty(),
        "baker": L("affirmed","unspecified","Quiste de Baker."),
        "contusion": empty(), "fracture": empty(),
    }
}

# ---------- Report 11 ----------
uid = "1.2.826.0.1.3680043.8.498.11557620559191469069130827959098335840"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("affirmed","severe","Rotura espesor y ancho total de la unión del tercio medio del tercio proximal del LCA"),
        "mcl": L("negated","none","LCP y ligamentos colaterales sin\nalteraciones significativas."),
        "med_men": L("affirmed","mild","Aumento de señal central del cuerno posterior y cuerpo del menisco medial que no impresiona\ncontactar con la superficie"),
        "lat_men": L("affirmed","mild","Menisco lateral con pequeña rotura radial incompleta de la unión del cuerno posterior con el\ncuerpo, sin desplazamiento"),
        "med_oa": L("negated","none","El resto de los cartílagos femorotibiales y patelofemorales sin alteraciones."),
        "lat_oa": L("affirmed","severe","Fina úlcera condral focal de espesor total del aspecto posterior de platillo tibial\nlateral."),
        "pf_oa": L("negated","none","El resto de los cartílagos femorotibiales y patelofemorales sin alteraciones."),
        "effusion": L("affirmed","moderate","Leve a moderado derrame articular, probablemente hemartrosis."),
        "synovitis": empty(),
        "baker": empty(),
        "contusion": L("affirmed","mild","discreto edema óseo del aspecto posterior de platillo tibial lateral"),
        "fracture": empty(),
    }
}

# ---------- Report 12 ----------
uid = "1.2.826.0.1.3680043.8.498.11590096889491409910425456896187293515"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": empty(),
        "mcl": L("affirmed","mild","Rotura parcial del LCM."),
        "med_men": empty(), "lat_men": empty(),
        "med_oa": empty(), "lat_oa": empty(), "pf_oa": empty(),
        "effusion": empty(), "synovitis": empty(), "baker": empty(), "contusion": empty(), "fracture": empty(),
    }
}

# ---------- Report 13 ----------
uid = "1.2.826.0.1.3680043.8.498.11794753699425693678056056340551238033"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colateral lateral sin alteraciones."),
        "mcl": L("affirmed","mild","Leve engrosamiento de baja señal del\ncomponente superficial del ligamento colateral medial."),
        "med_men": L("negated","none","Meniscos de morfología y señal conservada, sin criterios que de rotura."),
        "lat_men": L("negated","none","Meniscos de morfología y señal conservada, sin criterios que de rotura."),
        "med_oa": L("negated","none","Cartílagos del compartimento femorotibial medial y de la tróclea femoral sin alteraciones."),
        "lat_oa": L("affirmed","mild","Úlceras condrales de espesor parcial de la zona de carga del compartimento femorotibial lateral."),
        "pf_oa": L("affirmed","mild","Condropatía rotuliana superficial de la faceta lateral."),
        "effusion": L("negated","none","No hay significativo derrame articular ni quistes poplíteos."),
        "synovitis": empty(),
        "baker": L("negated","none","No hay significativo derrame articular ni quistes poplíteos."),
        "contusion": L("negated","none","No hay alteraciones de señal significativas de la médula ósea."),
        "fracture": empty(),
    }
}

# ---------- Report 14 ----------
uid = "1.2.826.0.1.3680043.8.498.12063742398421783626420335095128808816"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": empty(), "mcl": empty(), "med_men": empty(), "lat_men": empty(),
        "med_oa": empty(), "lat_oa": empty(), "pf_oa": empty(),
        "effusion": L("affirmed","mild","Efusión leve."),
        "synovitis": empty(),
        "baker": empty(),
        "contusion": empty(), "fracture": empty(),
    }
}

# ---------- Report 15 ----------
uid = "1.2.826.0.1.3680043.8.498.12264607993101717201407955391437731134"
records[uid] = {
    "knee_side": "left", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "med_men": L("affirmed","severe","Rotura compleja del cuerpo y cuerno posterior del menisco medial con extrusión asociada y\nsegmento desplazado"),
        "lat_men": L("affirmed","mild","Fibrilación marginal del cuerpo y cuerno posterior del menisco lateral."),
        "med_oa": L("affirmed","severe","Úlceras condrales de espesor parcial y total de la zona de carga del compartimento femorotibial\nmedial con cambios óseos"),
        "lat_oa": L("negated","none","Cartílagos de del compartimento femorotibial lateral y de la tróclea femoral sin alteraciones."),
        "pf_oa": L("affirmed","mild","Úlceras condrales de espesor parcial la faceta medial de la rótula."),
        "effusion": L("affirmed","moderate","Moderado derrame articular con sinovitis difusa."),
        "synovitis": L("affirmed","unspecified","Moderado derrame articular con sinovitis difusa."),
        "baker": empty(),
        "contusion": empty(),
        "fracture": empty(),
    }
}

# ---------- Report 16 ----------
uid = "1.2.826.0.1.3680043.8.498.12448079646359892252441208258836556945"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": empty(), "mcl": empty(), "med_men": empty(), "lat_men": empty(),
        "med_oa": empty(), "lat_oa": empty(),
        "pf_oa": L("affirmed","unspecified","Condropatía rotuliana."),
        "effusion": empty(), "synovitis": empty(), "baker": empty(), "contusion": empty(), "fracture": empty(),
    }
}

# ---------- Report 17 ----------
uid = "1.2.826.0.1.3680043.8.498.12505035424093604269515328931488770819"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("affirmed","unspecified","Rotura del LCA."),
        "mcl": L("affirmed","mild","Rotura parcial del LCM."),
        "med_men": L("affirmed","unspecified","Rotura de menisco interno y lateral."),
        "lat_men": L("affirmed","unspecified","Rotura de menisco interno y lateral."),
        "med_oa": empty(), "lat_oa": empty(), "pf_oa": empty(),
        "effusion": L("affirmed","unspecified","Derrame."),
        "synovitis": empty(),
        "baker": empty(),
        "contusion": L("affirmed","unspecified","Contusiones óseas femorotibiales."),
        "fracture": empty(),
    }
}

# ---------- Report 18 ----------
uid = "1.2.826.0.1.3680043.8.498.12718966639501121738042125605376714231"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": empty(), "mcl": empty(), "med_men": empty(), "lat_men": empty(),
        "med_oa": L("affirmed","unspecified","Condropatía femorotibial medial y patelofemoral."),
        "lat_oa": empty(),
        "pf_oa": L("affirmed","unspecified","Condropatía femorotibial medial y patelofemoral."),
        "effusion": L("affirmed","unspecified","Derrame con sinovitis."),
        "synovitis": L("affirmed","unspecified","Derrame con sinovitis."),
        "baker": empty(), "contusion": empty(), "fracture": empty(),
    }
}

# ---------- Report 19 ----------
uid = "1.2.826.0.1.3680043.8.498.12765434812409174436720333864278447770"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "med_men": L("affirmed","severe","Rotura compleja con componente principal horizontal oblicuo y cuerno posterior y cuerpo del\nmenisco medial"),
        "lat_men": L("negated","none","El menisco lateral es de morfología y señal conservada, sin signos de rotura."),
        "med_oa": L("negated","none","Cartílagos de los compartimentos femorotibiales y de la tróclea femoral sin alteraciones."),
        "lat_oa": L("negated","none","Cartílagos de los compartimentos femorotibiales y de la tróclea femoral sin alteraciones."),
        "pf_oa": L("affirmed","mild","Condropatía superficial de la rótula."),
        "effusion": L("affirmed","mild","Leve derrame articular."),
        "synovitis": empty(),
        "baker": L("negated","none","No hay quistes poplíteos patológicos."),
        "contusion": L("negated","none","No hay alteraciones en significativas de la médula ósea."),
        "fracture": empty(),
    }
}

# ---------- Report 20 ----------
uid = "1.2.826.0.1.3680043.8.498.13114693814781737495494700902699619138"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": empty(),
        "mcl": empty(),
        "med_men": L("affirmed","unspecified","Rotura de menisco interno."),
        "lat_men": L("affirmed","mild","Cambios de degeneración hialina intrasustancia en el cuerpo del menisco externo."),
        "med_oa": L("affirmed","unspecified","Condropatía del compartimento medial."),
        "lat_oa": L("affirmed","unspecified","Artrosis del compartimento lateral."),
        "pf_oa": empty(),
        "effusion": L("affirmed","unspecified","Derrame articular."),
        "synovitis": empty(),
        "baker": empty(),
        "contusion": empty(),
        "fracture": empty(),
    }
}

# ---------- Report 21 ----------
uid = "1.2.826.0.1.3680043.8.498.13146466304138609448483938844487385775"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "med_men": L("negated","none","El menisco medial es de morfología y señal conservada, sin criterios de una rotura\ncategórica."),
        "lat_men": L("hedged","unspecified","probable rotura del cuerno posterior y parte de la raíz\nmeniscal posterior del menisco lateral."),
        "med_oa": L("negated","none","El resto de los cartílagos femorotibiales y de la tróclea femoral sin\nalteraciones."),
        "lat_oa": L("affirmed","severe","Úlceras condrales focales de espesor total de la región central posterior de platillo tibial lateral con\ncambios óseos"),
        "pf_oa": L("affirmed","severe","Úlceras condrales de espesor total con cambios óseos secundarios del vértice y faceta medial de\nla rótula."),
        "effusion": L("affirmed","mild","Mínimo derrame articular."),
        "synovitis": empty(),
        "baker": L("affirmed","unspecified","Quiste poplíteo polilobulado, sin signos de rotura o filtración categórica."),
        "contusion": empty(),
        "fracture": empty(),
    }
}

# ---------- Report 22 ----------
uid = "1.2.826.0.1.3680043.8.498.14400292895187181989838122627394761532"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "med_men": L("affirmed","severe","Rotura horizontal oblicuo del cuerno posterior y cuerpo del menisco medial con segmento\nparcialmente desplazado"),
        "lat_men": L("affirmed","unspecified","Rotura horizontal del cuerno anterior, cuerpo y parte del cuerno posterior del menisco lateral, sin"),
        "med_oa": L("affirmed","mild","Úlceras condrales de espesor parcial de la región central y de carga del compartimento\nfemorotibial medial."),
        "lat_oa": L("negated","none","Cartílagos del compartimento femorotibial lateral sin alteraciones."),
        "pf_oa": L("affirmed","mild","Úlceras condrales de espesor parcial de la articulación patelofemoral."),
        "effusion": L("negated","none","No hay significativo derrame articular ni quistes poplíteos."),
        "synovitis": empty(),
        "baker": L("negated","none","No hay significativo derrame articular ni quistes poplíteos."),
        "contusion": L("negated","none","No hay alteraciones de señal significativas de la médula ósea."),
        "fracture": empty(),
    }
}

# ---------- Report 23 ----------
uid = "1.2.826.0.1.3680043.8.498.15593638897292057356864060466120253309"
records[uid] = {
    "knee_side": "left", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("affirmed","mild","Transformación quística parcial del tercio proximal del LCA."),
        "mcl": L("negated","none","LCP y ligamentos colaterales dentro de límites\nnormales."),
        "med_men": L("affirmed","unspecified","Aumento de señal central del cuerno posterior y cuerpo del menisco medial que contacta la superficie\narticular inferior"),
        "lat_men": L("affirmed","severe","Rotura compleja del cuerno anterior, cuerpo y cuerno posterior del menisco lateral con mínima extrusión"),
        "med_oa": L("affirmed","severe","Úlceras condrales de espesor parcial y total de la región central de la zona de carga del cóndilo femoral\nmedial."),
        "lat_oa": L("affirmed","severe","Úlceras condrales espesor parcial de la región central del platillo tibial lateral."),
        "pf_oa": L("affirmed","severe","Úlceras condrales de espesor total de distribución difusa de la rótula."),
        "effusion": L("affirmed","mild","Leve derrame articular con discreta sinovitis."),
        "synovitis": L("affirmed","mild","Leve derrame articular con discreta sinovitis."),
        "baker": L("negated","none","No hay quistes poplíteos."),
        "contusion": L("negated","none","No hay alteraciones de señal significativas de la médula ósea."),
        "fracture": empty(),
    }
}

# ---------- Report 24 ----------
uid = "1.2.826.0.1.3680043.8.498.21991252201684723900045301917670383228"
records[uid] = {
    "knee_side": "left", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales sin alteraciones actuales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales sin alteraciones actuales."),
        "med_men": L("negated","none","Meniscos impresionan de morfología y señal conservada, sin criterios categóricos de rotura."),
        "lat_men": L("negated","none","Meniscos impresionan de morfología y señal conservada, sin criterios categóricos de rotura."),
        "med_oa": L("affirmed","moderate","Úlceras condrales de espesor parcial de la región central de la zona de carga del cóndilo femoral\nmedial."),
        "lat_oa": L("affirmed","mild","Condropatía superficial del resto de los cartílagos femorotibiales."),
        "pf_oa": L("affirmed","severe","Úlceras condrales focales de espesor total de la vertiente lateral y región central de la tróclea\nfemoral"),
        "effusion": L("affirmed","mild","Leve derrame articular con discreta sinovitis difusa."),
        "synovitis": L("affirmed","mild","Leve derrame articular con discreta sinovitis difusa."),
        "baker": L("negated","none","No hay quistes poplíteos patológicos."),
        "contusion": L("negated","none","No hay alteraciones de señal significativas de la médula ósea."),
        "fracture": empty(),
    }
}

# ---------- Report 25 ----------
uid = "1.2.826.0.1.3680043.8.498.23094733099955217022059584418468076048"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": empty(), "mcl": empty(), "med_men": empty(), "lat_men": empty(),
        "med_oa": empty(), "lat_oa": empty(),
        "pf_oa": L("affirmed","mild","Condropatía patelofemoral leve."),
        "effusion": empty(), "synovitis": empty(), "baker": empty(), "contusion": empty(), "fracture": empty(),
    }
}

# ---------- Report 26 ----------
uid = "1.2.826.0.1.3680043.8.498.25266231062774974303723359806535863810"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("affirmed","severe","Rotura completa de LCA."),
        "mcl": L("affirmed","mild","Rotura parcial del LCM."),
        "med_men": L("affirmed","unspecified","Rotura de menisco interno y lateral."),
        "lat_men": L("affirmed","unspecified","Rotura de menisco interno y lateral."),
        "med_oa": empty(), "lat_oa": empty(), "pf_oa": empty(),
        "effusion": L("affirmed","unspecified","Derrame con sinovitis."),
        "synovitis": L("affirmed","unspecified","Derrame con sinovitis."),
        "baker": empty(),
        "contusion": L("affirmed","unspecified","Contusión ósea tibial y femoral."),
        "fracture": empty(),
    }
}

# ---------- Report 27 ----------
uid = "1.2.826.0.1.3680043.8.498.28289838818003822677994912735532050038"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": empty(), "mcl": empty(),
        "med_men": L("affirmed","unspecified","Rotura de menisco interno."),
        "lat_men": empty(),
        "med_oa": empty(), "lat_oa": empty(), "pf_oa": empty(),
        "effusion": L("affirmed","unspecified","Derrame."),
        "synovitis": empty(),
        "baker": L("affirmed","unspecified","Quiste de Baker."),
        "contusion": empty(),
        "fracture": empty(),
    }
}

# ---------- Report 28 ----------
uid = "1.2.826.0.1.3680043.8.498.28490861391717560759667589681288427737"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "med_men": L("affirmed","moderate","Rotura de la raíz meniscal posterior, cuerno posterior y parte del cuerpo del menisco medial con moderada"),
        "lat_men": L("affirmed","severe","Rotura compleja con componente principal horizontal oblicuo del cuerpo y cuerno posterior del menisco\nlateral."),
        "med_oa": L("affirmed","severe","Úlceras condrales de espesor parcial y total de la zona de carga del cóndilo femoral medial."),
        "lat_oa": L("affirmed","severe","Úlceras\ncondrales de espesor parcial y total de la región central posterior de platillo tibial lateral."),
        "pf_oa": L("affirmed","severe","Avanzados cambios degenerativos patelofemorales."),
        "effusion": L("affirmed","moderate","Moderado derrame articular con sinovitis difusa."),
        "synovitis": L("affirmed","unspecified","Moderado derrame articular con sinovitis difusa."),
        "baker": empty(),
        "contusion": L("affirmed","unspecified","Edema óseo con fractura subcortical de la zona de carga del platillo tibial medial"),
        "fracture": L("affirmed","unspecified","Edema óseo con fractura subcortical de la zona de carga del platillo tibial medial"),
    }
}

# ---------- Report 29 ----------
uid = "1.2.826.0.1.3680043.8.498.29794301133358685635777478470865282350"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales continuos."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales continuos."),
        "med_men": L("negated","none","Meniscos de morfología y señal normal, sin criterios de rotura."),
        "lat_men": L("negated","none","Meniscos de morfología y señal normal, sin criterios de rotura."),
        "med_oa": L("negated","none","Cartílago de los compartimentos femorotibiales de espesor y señal conservados."),
        "lat_oa": L("negated","none","Cartílago de los compartimentos femorotibiales de espesor y señal conservados."),
        "pf_oa": L("affirmed","mild","Cartílago patelofemoral de espesor conservado con alteración difusa de señal intrasustancia, sin\ndefectos condrales"),
        "effusion": L("negated","none","No hay derrame articular."),
        "synovitis": empty(),
        "baker": L("negated","none","No se observa quiste de Baker."),
        "contusion": L("negated","none","No hay alteración de señal difusa de la médula ósea."),
        "fracture": empty(),
    }
}

# ---------- Report 30 ----------
uid = "1.2.826.0.1.3680043.8.498.31175617350083221245857813741060042777"
records[uid] = {
    "knee_side": "unknown", "postoperative": False, "normal_template": False,
    "labels": {
        "acl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "mcl": L("negated","none","Ligamentos cruzados y colaterales dentro de límites normales."),
        "med_men": L("negated","none","Meniscos de morfología y señal conservada, sin signos de rotura."),
        "lat_men": L("negated","none","Meniscos de morfología y señal conservada, sin signos de rotura."),
        "med_oa": L("negated","none","El resto de los cartílagos femorotibiales y de la\ntróclea femoral sin alteraciones."),
        "lat_oa": L("affirmed","mild","Condropatía superficial de platillo tibial lateral."),
        "pf_oa": L("affirmed","mild","Úlceras condrales de espesor parcial de distribución difusa de la\nrótula."),
        "effusion": empty(),
        "synovitis": empty(),
        "baker": empty(),
        "contusion": L("negated","none","No hay alteraciones de señal significativas de la médula ósea."),
        "fracture": empty(),
    }
}

# ---------------- validate & write ----------------
assert set(records.keys()) == set(order), f"missing/extra ids: {set(order) ^ set(records.keys())}"

errors = []
for uid in order:
    rec = records[uid]
    text = reports[uid]
    labels = rec["labels"]
    expected_keys = {"acl","mcl","med_men","lat_men","med_oa","lat_oa","pf_oa","effusion","synovitis","baker","contusion","fracture"}
    if set(labels.keys()) != expected_keys:
        errors.append(f"{uid}: label key mismatch {set(labels.keys()) ^ expected_keys}")
    for k, v in labels.items():
        st = v["state"]; sev = v["severity"]; ev = v["evidence"]
        if st not in ("affirmed","hedged","negated","absent"):
            errors.append(f"{uid}/{k}: bad state {st}")
        if sev not in ("none","mild","moderate","severe","unspecified"):
            errors.append(f"{uid}/{k}: bad severity {sev}")
        if st in ("negated","absent") and sev != "none":
            errors.append(f"{uid}/{k}: severity should be none for state {st}, got {sev}")
        if len(ev) > 120:
            errors.append(f"{uid}/{k}: evidence too long ({len(ev)})")
        if ev == "" and st != "absent":
            # allowed technically but flag for review
            pass
        if ev != "" and ev not in text:
            errors.append(f"{uid}/{k}: evidence not substring: {ev!r}")

if errors:
    print(f"FOUND {len(errors)} ERRORS:")
    for e in errors:
        print(" -", e)
    sys.exit(1)

with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    for uid in order:
        line = {"StudyInstanceUID": uid, "raw": records[uid]}
        f.write(json.dumps(line, ensure_ascii=False) + "\n")

print(f"WROTE {len(order)} records OK to {out_path}")
