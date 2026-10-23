# Offline arbiters

2 candidates, 2000 bootstrap replicates (seed 2026, studies resampled once per arbiter and reused for every candidate, so deltas are paired).

## 1. What each arbiter trusts

| arbiter | cells kept | labels scored | gold cells | label == gold | balanced (label vs gold) | gold pos rate | CI95 width (best cand.) |
|---|---|---|---|---|---|---|---|
| weak | 29,758 | 12/12 | 397 | 0.887 | 0.858 | 0.423 | 0.0091 |
| strict | 36,848 | 12/12 | 499 | 0.780 | 0.776 | 0.425 | 0.0106 |
| consensus | 28,835 | 12/12 | 405 | 0.798 | 0.781 | 0.472 | 0.0140 |
| gold58 | 696 | 12/12 | 696 | 1.000 | 1.000 | 0.345 | 0.0595 |

## 2. Scores

| candidate | weak | strict | consensus | gold58 | composite |
|---|---|---|---|---|---|
| eff_effv2s_256 | 0.9088 | 0.8388 | 0.8615 | 0.8666 | 0.8892 |
| eff_pseudo_256 | 0.9413 | 0.8767 | 0.8994 | 0.8999 | 0.9237 |

## 3. Ranking per arbiter

| arbiter | ordering (best first) |
|---|---|
| weak | eff_pseudo_256 > eff_effv2s_256 |
| strict | eff_pseudo_256 > eff_effv2s_256 |
| consensus | eff_pseudo_256 > eff_effv2s_256 |
| gold58 | eff_pseudo_256 > eff_effv2s_256 |

## 4. Paired deltas (what each arbiter can resolve)

`half-width` is half the 95 % CI of the paired delta: an arbiter cannot resolve a difference smaller than that, however many metrics agree.

| pair | weak delta [CI] P | strict delta [CI] P | consensus delta [CI] P | gold58 delta [CI] P |
|---|---|---|---|---|
| eff_effv2s_256 - eff_pseudo_256 | -0.0325 [-0.0363,-0.0289] P=0.00 | -0.0379 [-0.0413,-0.0347] P=0.00 | -0.0379 [-0.0425,-0.0337] P=0.00 | -0.0328 [-0.0538,-0.0126] P=0.00 |

| arbiter | median paired half-width | narrowest | widest |
|---|---|---|---|
| weak | 0.0037 | 0.0037 | 0.0037 |
| strict | 0.0033 | 0.0033 | 0.0033 |
| consensus | 0.0044 | 0.0044 | 0.0044 |
| gold58 | 0.0206 | 0.0206 | 0.0206 |

## 5. Per-label coverage

| label | weak | strict | consensus | gold58 |
|---|---|---|---|---|
| acl | 3,585 (560+/3025-) | 3,929 (1050+/2879-) | 3,142 (792+/2350-) | 58 (24+/34-) |
| mcl | 3,473 (132+/3341-) | 3,894 (748+/3146-) | 3,190 (483+/2707-) | 58 (9+/49-) |
| med_men | 3,475 (1564+/1911-) | 4,021 (2186+/1835-) | 3,356 (1957+/1399-) | 58 (26+/32-) |
| lat_men | 3,698 (591+/3107-) | 3,878 (1003+/2875-) | 3,041 (828+/2213-) | 58 (23+/35-) |
| med_oa | 2,241 (957+/1284-) | 3,118 (1796+/1322-) | 2,064 (1124+/940-) | 58 (15+/43-) |
| lat_oa | 1,941 (450+/1491-) | 2,754 (1140+/1614-) | 1,739 (662+/1077-) | 58 (11+/47-) |
| pf_oa | 2,590 (1259+/1331-) | 3,355 (1940+/1415-) | 2,529 (1495+/1034-) | 58 (21+/37-) |
| effusion | 2,661 (974+/1687-) | 3,945 (2637+/1308-) | 3,648 (2455+/1193-) | 58 (35+/23-) |
| synovitis | 400 (347+/53-) | 596 (532+/64-) | 486 (466+/20-) | 58 (27+/31-) |
| baker | 1,619 (531+/1088-) | 2,170 (1055+/1115-) | 1,674 (911+/763-) | 58 (12+/46-) |
| contusion | 2,070 (473+/1597-) | 2,906 (981+/1925-) | 2,181 (827+/1354-) | 58 (19+/39-) |
| fracture | 2,005 (228+/1777-) | 2,282 (208+/2074-) | 1,785 (182+/1603-) | 58 (18+/40-) |
