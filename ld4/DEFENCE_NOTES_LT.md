# Gynimo užrašai (mokykitės patys – gynime DI negalima)

## Attention (student_model.py) – mokėkite parašyti ant lentos
```
scores  = Q @ Kᵀ / √d_head          # [b, h, t, t]; √d, kad dispersija ~1 ir softmax neprisotintų
scores[j > i] = -inf                 # kaukė PRIEŠ softmax: exp(-inf)=0, eilutė vis tiek sumuojasi į 1
weights = softmax(scores, dim=-1)
out     = weights @ V                # [b, h, t, d_head]
out     = out.transpose(1,2).reshape(b, t, h*d_head) -> self.out(...)
```
- **Kodėl ne po softmax?** Nulinant po softmax eilutė nebesumuoja į 1, o ateities balai jau paveikė vardiklį. Taip informacija iš ateities „nuteka“.
- **4 tokenų kaukė:** apatinis trikampis su įstrižaine (t1 mato t1; t4 mato t1..t4).
- **Ką tikrina verify_attention:** (1) pakeitus 3+ pozicijas, 0–2 išėjimai nesikeičia; (2) gradientas į ateities įėjimus = 0; (3) jei Q=K=0, visi balai lygūs, todėl išėjimas = ankstesnių V vidurkis ([1,2,3,4], [2,3,4,5], [3,4,5,6]). Tai patikrina ir kaukę, ir galvų sujungimo tvarką.
- **Galvos:** 4 galvos × 24 dim. = 96. Kiekviena galva turi savo dėmesio pasiskirstymą.

## Paieška (student_search.py)
- cos(q,d) = q·d / (|q|·|d|); normas apribojame ε=1e‑12, todėl nulinis vektorius duoda 0, o ne NaN.
- Rikiavimas `argsort(-score, kind='stable')`: lygūs balai išlaiko mažesnį eilutės indeksą. Būtent todėl dvynių pavadinimų atveju laimi tas, kuris documents.jsonl faile eina anksčiau.
- LSA: TF‑IDF (1–2 gramos) → TruncatedSVD 64 dim. → normalizuota. Balas yra panašumas, **ne** atsakymo patikimumas.

## Skaičiai, kuriuos reikia žinoti
338 450 parametrų · žodynas 914 · 4,6 min CPU · geriausias 5 500 žingsnis · val: be šaltinio 2/158, teisinga 133/158, paieška 132/158, top‑1 188/200 · test: 1/75, 69/75, 68/75, top‑1 96/100 · taisyklė 100 % · pataisymas 133 → 98.

## Kaip elgtis su dėstytojo pakeistu faktu (gynimo užduotis)
Prieš paleidžiant pasakykite prognozę pagal šias taisykles:
- **Kategoriniai laukai** (type, license yes/no): beveik visada nukopijuos naują reikšmę, jei ji buvo matyta mokyme.
- **year:** nukopijuos tik 2020–2025. Kitus metus (2019, 2030) pakeis į artimą matytą reikšmę → tikėtina klaida.
- **authors / month** (1–2 skaitmenys): dažniausiai teisingai.
- **references** (kelių skaitmenų, retos reikšmės): rizika gauti „0“ arba tik pirmą skaitmenį.
- **Pašalintas laukas:** turėtų atsakyti unknown (abstention 100 % su teisinga kortele).
- **Naujas žodis, kurio nėra žodyne** (pvz., type : thesis), tampa `<unk>`, todėl modelis jo nesugeneruos.
- Jei pakeistas pavadinimas: paieška gali rasti kitą kortelę. Tada pirma pažiūrėkite į top‑3 ir pasakykite, ar tai paieškos, ar GPT klaida.

## Klaidų skyrimas UI
Paryškintas laukas kontekste = ką modelis turėjo nukopijuoti. Ženklelis „nesutampa su konteksto lauku“ → GPT klaida. Jei kortelė (ID) ne ta, kurios tikėjotės → paieškos klaida. Režimas „pasirinkta kortelė“ leidžia tai patikrinti, nes paieška tada aplenkiama.

## Tikėtini klausimai
- *Kodėl taisyklė geresnė?* Struktūruotas formatas; GPT iš dalies įsimena reikšmes, o ne kopijuoja (2019 pavyzdys).
- *Kodėl pataisymas pablogino?* Pašalino pozicinę nuorodą, kuria modelis atskyrė skaitinius laukus.
- *Ar modelis haliucinuoja?* Neatsakomuose su teisinga kortele 0/42 nepagrįstų; be šaltinio 8/42 (spėja „6“ mėnesiui); su laisvu klausimu „pirmasis autorius“ → „11“.
- *Kas būtų kitas žingsnis?* Pridėti paiešką pagal DOI/autorius dvyniams atskirti; mokyme įvairinti skaičių reikšmes (sintetinės kortelės) arba naudoti kopijavimo mechanizmą (pointer).

## Ką rodyti gynime (viskas vienoje sąsajoje)
1. Prieš paleisdami (jei Windows rodo UnicodeDecodeError, kodas jau pataisytas, bet atsargumo dėlei): `$env:PYTHONUTF8=1`
2. `.venv\Scripts\python app.py --topic research --checkpoint outputs/research_LT/best.pt` → atidarykite `http://127.0.0.1:8904`
3. Skirtukas **„Klausti modelio“**: įveskite pavadinimą, pasirinkite klausimą, „Atsakyti“ – matosi 4 etapai, kontekstas, tokenai, top‑3, būsena.
   Režimas „Pasirinkta kortelė“ leidžia atskirti paieškos klaidą nuo GPT klaidos; „Be šaltinio“ – parodo, kad GPT be konteksto nežino.
4. Skirtukas **„Rezultatai ir pakeitimai“**: 3 sąlygų lentelė, modelis ir nuostolis, pataisymas prieš/po, klaidos, 10 klausimų, ką parašėte.
   Visi skaičiai skaitomi iš `outputs/` failų (adresas `/report`), ne įrašyti ranka.
