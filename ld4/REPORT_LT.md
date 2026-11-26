# L4 · Mažas GPT asistentas – ataskaita (tema: research, kalba LT)

## 1. Sistema ir aplinka
- **Grandinė:** užklausa (pavadinimas/raktažodžiai + vienas iš 6 klausimų šablonų) → LSA paieška (TF‑IDF 1–2 gramos → SVD 64 dim., kosinusas, top‑3) → top‑1 metaduomenų kortelė perduodama kaip kontekstas → GPT generuoja atsakymo tokenus iki `<eos>` (godus dekodavimas, riba 8 tokenai).
- **Mano kodas:** `student_model.py` (QKᵀ/√d, priežastinė kaukė prieš softmax, svertinė V suma, galvų sujungimas), `student_search.py` (saugus normalizavimas su ε, kosinusas, stabilus top‑k; nulinė užklausa → balai 0, ne NaN), `ui.html` (dvi skiltys: gyvas klausimas modeliui ir rezultatai), `experiment.py`, `repair_augment.py`, `personal_questions.py`, `make_results.py`, `report_data.py`. Abi patikros: **PASS**.
- **Modelis:** 2 dekoderio blokai, d = 96, 4 galvos, kontekstas 128, žodynas 914, **338 450 parametrų**, pradžia nuo atsitiktinių svorių (seed 42).
- **Mokymas:** Windows, Python 3.11.9, torch 2.15.0 (CPU, 1 gija), AdamW lr 0,002, batch 32, nuostolis tik atsakymo ir EOS tokenams. **10 000 žingsnių (riba), 507 s (8,4 min)**. Geriausias validavimo EM 0,91 pasiektas 9 500 žingsnyje; išsaugotas tas modelis.
- **Nuostolis** (`outputs/research_LT/loss_curve.png`): mokymo nuostolis po ~3 000 žingsnių < 0,001, validavimo krinta nuo 0,57 iki ~0,30 (žemiausias 0,296 ties 6 500), po to šiek tiek kyla iki 0,33. Didelis tarpas rodo, kad modelis dalį 2 100 mokymo porų įsimena.

## 2. Trys sąlygos ir paprasti būdai (tas pats modelis, tie patys klausimai)

| Sąlyga | Paieškos top‑1 | Atsakomų EM | Teisingas unknown | Nepagrįsti | Neužbaigti |
|---|---|---|---|---|---|
| **Validavimas** (200 kl.: 158 atsakomi, 42 neatsakomi) | | | | | |
| Be šaltinio | netaikoma | 2/158 (1,3 %) | 34/42 (81,0 %) | 8/42 | 0/200 |
| Teisinga kortelė (document_id) | netaikoma | 140/158 (88,6 %) | 42/42 (100 %) | 0/42 | 0/200 |
| Paieškos top‑1 kortelė | 188/200 (94,0 %) | 138/158 (87,3 %) | 39/42 (92,9 %) | 3/42 | 0/200 |
| Daugumos atsakymas (train) | netaikoma | 48/158 (30,4 %) | 8/42 (19,0 %) | 34/42 | 0/200 |
| Lauko skaitymo taisyklė | netaikoma | 158/158 (100 %) | 42/42 (100 %) | 0/42 | 0/200 |
| **Testas** (vieną kartą, 100 kl.: 75 / 25) | | | | | |
| Be šaltinio | netaikoma | 1/75 (1,3 %) | 18/25 (72 %) | 7/25 | 0/100 |
| Teisinga kortelė | netaikoma | 70/75 (93,3 %) | 25/25 (100 %) | 0/25 | 0/100 |
| Paieškos top‑1 kortelė | 96/100 (96,0 %) | 69/75 (92,0 %) | 25/25 (100 %) | 0/25 | 0/100 |
| Daugumos atsakymas | netaikoma | 32/75 (42,7 %) | 7/25 (28 %) | 18/25 | 0/100 |
| Lauko skaitymo taisyklė | netaikoma | 75/75 (100 %) | 25/25 (100 %) | 0/25 | 0/100 |

**Išvados.**
1. Be šaltinio GPT beveik nieko neatsako teisingai (1–2 atsakymai), todėl atsakymą jis ima iš konteksto, o ne iš atminties.
2. Su paieška validavime iš 20 neteisingų atsakomų atsakymų tik 2 kilo dėl blogos kortelės (paieškos klaida), 18 – GPT klaidos su teisinga kortele.
3. Paieškos klaidos (12/200) – daugiausia **identiški pavadinimai** (pvz., du „Machine Learning for DUNE Supernova Trigger“ su balu 1,000), kurių pagal pavadinimą atskirti neįmanoma.
4. **Taisyklė lenkia GPT**, nes kortelės formatas visiškai struktūruotas (`laukas : reikšmė ;`) – užtenka nukopijuoti reikšmę. GPT tai turi išmokti iš 2 100 pavyzdžių, o skaitmenis kopijuoja netobulai. Pagal laukus (validavimas, teisinga kortelė): type 24/24, license 27/27, year 29/30, authors 31/32, month 16/21, **references 13/24**. Teste references 8/11, visi kiti ≥ 5/6.

## 3. Pataisymas: prognozė → vienas pakeitimas → prieš/po
- **Trūkumas:** skaitiniai laukai (references 13/24, month 16/21); mokymo nuostolis ≈0, validavimo ≈0,3.
- **Prognozė (užrašyta prieš bandymą, `PREDICTION.txt`):** jei modelis turės rasti lauką pagal pavadinimą, o ne pagal įsimintą vietą, kopijavimas pagerės; tikėjausi +3…+6 p. EM su teisinga kortele, daugiausia references ir month.
- **Pakeitimas (vienas):** kiekviena mokymo kortelė pridėta dar 2 kartus su atsitiktine laukų tvarka; modelis, seed, žingsnių riba – tie patys.
- **Rezultatas (tas pats validavimas, tas pats kompiuteris):**

| Sąlyga / laukas | Bazinis | Su laukų maišymu |
|---|---|---|
| Teisinga kortelė, atsakomų EM | 140/158 (88,6 %) | **103/158 (65,2 %)** |
| Paieškos top‑1 kortelė, atsakomų EM | 138/158 (87,3 %) | 102/158 (64,6 %) |
| Teisingas unknown (teisinga kortelė) | 42/42 | 41/42 |
| Be šaltinio, atsakomų EM | 2/158 | 17/158 |
| type | 24/24 | 23/24 |
| license_reported | 27/27 | 27/27 |
| year | 29/30 | **17/30** |
| authors | 31/32 | **20/32** |
| month | 16/21 | **8/21** |
| references | 13/24 | **8/24** |

- **Paaiškinimas:** prognozė nepasitvirtino – vietoje +3…+6 p. gavau **−23,4 p.** Kategoriniai laukai (type, license) beveik nepakito, bet krito visi keturi skaitiniai. Bazinis modelis naudojo **fiksuotą laukų tvarką** kaip nuorodą, kuris skaičius yra kuris (year visada pirmas, references po authors). Sumaišius tvarką keturi skaitiniai laukai atrodo vienodai, ir lauko pavadinimą reikia susieti su po jo einančiais skaitmenimis vien turiniu – 2 blokų modeliui per tą pačią žingsnių ribą to išmokti nepavyko. Tai matyti ir be šaltinio: EM pakilo nuo 2 iki 17, nes modelis dažniau spėja reikšmę (be šaltinio authors 15/32, kai dažniausias mokymo atsakymas yra „1“), užuot atsakęs unknown. Pataisymo netaikiau, testui paliktas bazinis modelis. Paieškos top‑1 nepakito (188/200), nes paieška nekeista.

## 4. Trys konkrečios klaidos (su tikru kontekstu)
1. **GPT klaida, kortelė teisinga.** `Title : Quantum Machine Learning and Deep Learning: … ; year : 2025 ; authors : 2 ; references : 104 ; license_reported : yes ;` → „Kiek literatūros nuorodų nurodyta?“ → GPT **0**, laukta 104. Tas pats su „Improving Transportation Planning…“ (`references : 40` → **0**). Mokyme references dažniausia reikšmė yra „0“ (dauguma), o 40 ir 104 – retos, todėl modelis grįžta prie dažnos reikšmės užuot kopijavęs skaitmenis.
2. **Paieškos klaida → nepagrįstas atsakymas.** Užklausa „generative art machine learning deep learning“, klausimas apie metus. Ieškotos kortelės 10.70593/978-81-981271-4-3_5 metų lauko **nėra** (laukta unknown), bet paieška grąžino kitą dokumentą „Deep Q‑Learning“ (balas 0,909), kurio kortelėje `year : 2023 ;` → GPT **2023**. GPT veikė teisingai pagal gautą kontekstą; klaida – paieškos (trumpi raktažodžiai, panašūs žodžiai „deep learning“).
3. **Nematyta reikšmė (pakeistas faktas).** Kortelėje „RDR and Machine Learning“ `year : 2021` pakeitus į **2019**, GPT vis tiek atsako **2021**. Mokymo duomenyse metai yra tik 2020–2025, todėl metams modelis veikia kaip klasifikatorius iš matytų reikšmių, o ne kaip tikras kopijuotojas. Pakeitimus authors 2 → 7 ir type → journal‑article jis seka teisingai.

## 5. Mano 10 klausimų (validavimo dokumentai, testas nenaudotas) – 5/10 teisingi
| # | Tipas | Klausimas | Šaltinis | Laukta | GPT |
|---|---|---|---|---|---|
| 1 | šablonas | Kiek literatūros nuorodų? – Improving Transportation Planning… | 10.4018/978-1-7998-9220-5.ch184 | 40 | 0 ✗ |
| 2 | šablonas | Koks tipas? – Michel Electrons… | 10.2172/1996940 | proceedings-article | ✓ |
| 3 | šablonas | Kuris mėnuo? – Trustworthy ML with Noisy Labels | 10.1007/978-981-96-9396-2_2 | 8 | 8 ✓ |
| 4 | šablonas | Kiek nuorodų? – Quantum ML and DL Fundamentals | 10.3390/make7030075 | 104 | 0 ✗ |
| 5 | šablonas, **neatsakomas** | Kurie metai? – „generative art…“ | 10.70593/978-81-981271-4-3_5 | unknown | 2023 ✗ (paieška rado „Deep Q‑Learning“) |
| 6 | šablonas, **neatsakomas** | Kiek nuorodų? – Reducing MEG interference | 10.1016/j.mlwa.2023.100462 | unknown | unknown ✓ |
| 7 | laisvas, **neatsakomas** | Kokiais metais išleista? – Mining commit messages… | 10.1016/j.mlwa.2022.100316 | unknown | unknown ✓ |
| 8 | laisvas | Kiek autorių parašė? – RDR and Machine Learning | 10.1201/9781003126157-9 | 2 | 2 ✓ |
| 9 | laisvas, **neatsakomas** | Kas yra pirmasis autorius? – Diabetes Prediction… | 10.1145/3635638.3635643 | unknown | 10 ✗ |
| 10 | laisvas | Ar straipsnis turi licenciją? – Reducing MEG interference | 10.1016/j.mlwa.2023.100462 | yes | unknown ✗ |

Šablonai 3/6, laisvi 2/4. Klausimas apie lauką, kurio modelis nemokė („pirmasis autorius“), sukelia nepagrįstą skaičių – modelis neatpažįsta, kad toks klausimas nepalaikomas. Laisva formuluotė „Ar turi licenciją?“ neatitinka šablono, todėl modelis nesusieja jos su lauku `license_reported` ir atsako unknown. Trumpi raktažodžiai (5 klausimas) paiešką paveikia labiau negu pilnas pavadinimas.

## 6. Atkūrimas
```
.venv\Scripts\python verify_attention.py
.venv\Scripts\python verify_search.py
.venv\Scripts\python core.py --data data/research --out outputs/research_LT --lang LT --minutes 30 --dim 96 --heads 4
.venv\Scripts\python experiment.py --checkpoint outputs/research_LT/best.pt --split validation --out outputs/research_LT/exp_validation.json
.venv\Scripts\python experiment.py --checkpoint outputs/research_LT/best.pt --split test --out outputs/research_LT/exp_test.json
.venv\Scripts\python repair_augment.py --out outputs/repair_aug_s42 --seed 42
.venv\Scripts\python experiment.py --checkpoint outputs/repair_aug_s42/best.pt --split validation --out outputs/repair_aug_s42/exp_validation.json
.venv\Scripts\python personal_questions.py outputs/research_LT/best.pt outputs/research_LT/personal_questions.json
.venv\Scripts\python make_results.py
.venv\Scripts\python app.py --topic research --checkpoint outputs/research_LT/best.pt
```
Išsaugotas modelis: `outputs/research_LT/best.pt` (mano mokymo rezultatas, seed 42); atkuriamas su `core.restore(path)`. Sąsaja: `http://127.0.0.1:8904`, skiltis „Rezultatai ir pakeitimai“ skaito tuos pačius `outputs/` failus.
