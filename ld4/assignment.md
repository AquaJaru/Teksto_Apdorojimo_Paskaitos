# 4 laboratorinis – Sukurkite savo mažą GPT asistentą

## Ką turite sukurti?
Sukurkite veikiančią klausimų–atsakymų sistemą: vartotojas nurodo, apie ką klausia, sistema suranda dokumento kortelę, o jūsų pačių išmokytas mažas GPT pateikia trumpą atsakymą. Ekrane turi matytis atsakymas, šaltinis ir modeliui perduotas tekstas.

**Pagrindinis tikslas:** suprasti, ką išmoksta modelis ir kodėl sistema suklysta. Vien paleisti pateiktą kodą nepakanka: dalį kodo parašysite patys, atliksite palyginimą ir patikrinsite savo pasiūlytą pataisymą.

## 1. Pasirinkite vieną temą
- **Python žinynas (`software`):** klausimai apie funkcijos modulį, parametrus ar tipą.
- **Universiteto kursai (`university`):** klausimai apie MIT OpenCourseWare kurso semestrą, lygį ar metus.
- **Mokslinės publikacijos (`research`):** klausimai apie publikacijos metus, autorių skaičių ar tipą.

Pakete yra visos trys temos, tačiau darbą atlikite tik su viena. Kiekvienoje yra 500 šaltinį turinčių kortelių. Tai žinyno ar katalogo įrašai, ne 500 pilnų knygų ar straipsnių. Modelio užduotis – trumpi faktiniai atsakymai iš pateiktos kortelės.

## 2. Kaip turi veikti rezultatas? Konkretus pavyzdys
Tai tikras pateiktų mokymo duomenų pavyzdys iš temos `university`. Žemiau nurodyti **laukiami atsakymai**, o ne pažadas, kad jūsų modelis visada juos sugeneruos.

1. **Vartotojas įveda kursą:** Transportation Policy, Strategy, and Management. Pasirenka klausimą: „Kuris semestras nurodytas?“
2. **Paieška atrenka kortelę:** jos pavadinimas sutampa su kursu. Sąsajoje parodykite tris geriausius kandidatus ir tikras šaltinių nuorodas. Patikrinkite, ar pirmoji kortelė tikrai tinkama.
3. **GPT gauna tekstą:** `Reference metadata. Title : Transportation Policy, Strategy, and Management ; semester : fall ; level : graduate ; materials : 0 ; topics : 5 ;` ir klausimą „Kuris semestras nurodytas?“
4. **Laukiamas atsakymas:** `fall` – rudens semestras. Atsakymą pagrindžia laukas `semester : fall`. Modelis atsakymo tokenus turi sugeneruoti pats.
5. **Patikrinimas:** jei paieška parinko ne tą kursą, tai paieškos klaida. Jei kortelė teisinga, bet GPT pateikė kitą semestrą, tai modelio klaida. Abi klaidas parodykite atskirai.

**Antras klausimas tai pačiai kortelei:** „Kurie kurso metai nurodyti?“ Laukiamas atsakymas – `unknown`, nes pateiktame tekste nėra metų lauko. Originalaus puslapio adrese metų gali būti, tačiau modelis turi remtis jam perduotu tekstu. `unknown` reiškia „pateiktame kontekste informacijos nėra“.

Šaltinis: https://ocw.mit.edu/courses/1-223j-transportation-policy-strategy-and-management-fall-2004/ . Duomenų įrašas: `ocw-3735`.

## 3. Ką gaunate ir ką atliekate patys?
| Pateikta pakete | Jūsų darbas | Kaip pasitikrinti |
| --- | --- | --- |
| Duomenys ir notebookas | Pasirinkti temą; ranka atsakyti į 2 klausimus prieš DI pagalbą | Parodyti atsakymą pagrindžiantį lauką |
| GPT ir mokymo karkasas | Užbaigti attention faile `student_model.py`; išmokyti modelį nuo nulio | `python verify_attention.py`, nuostolio grafikas, tikri modelio atsakymai |
| Dokumentų vektorių karkasas | Užbaigti kosinuso balą ir rikiavimą faile `student_search.py` | `python verify_search.py` ir 3 paieškos rezultatai |
| Serveris ir pradinė sąsaja | Užbaigti `ui.html` rezultatų rodymą | Klausimas → šaltinis → kontekstas → atsakymas |
| Vertinimo funkcijos | Parašyti trijų sąlygų palyginimą ir savo pataisymo eksperimentą | Rezultatų lentelė ir konkrečių klaidų paaiškinimas |

## 4. Atlikite darbą šia tvarka
1. Išarchyvuokite visą ZIP. Pagal `README.md` paruoškite atskirą Python aplinką. VS Code atverkite `Lab04_LT.ipynb`. Pasirinkite vieną `TOPIC`.
2. Peržiūrėkite korteles ir ranka atsakykite į 2 klausimus. Nupieškite 4 tokenų kaukę: kuriuos ankstesnius tokenus kiekviena pozicija gali matyti? Paaiškinkite, kodėl ateities atsakymas turi būti paslėptas.
3. Užbaikite attention ir paieškos TODO. Paleiskite abu patikrų failus. `NotImplementedError` iki TODO užbaigimo yra numatytas – tai nėra sugadintas paketas.
4. Išmokykite savo GPT ir išsaugokite modelį. Užrašykite parametrų skaičių, aplinką ir mokymo trukmę; pateikite mokymo bei validavimo nuostolio grafiką ir kelis sugeneruotus atsakymus.
5. Užbaikite sąsają ir atlikite žemiau aprašytą eksperimentą. Tik po validavimo užfiksuokite nustatymus ir vieną kartą atlikite galutinį testą.

GPT pradedamas nuo **atsitiktinių svorių**. Parengtų GPT/LLM svorių ar išorinių pokalbių API naudoti negalima. Paieška vietoje išmoksta dokumentų vektorius; jai taip pat nereikia parengtų neuroninių svorių. Detalūs attention, duomenų ir mokymo paaiškinimai – `TECHNICAL_DETAILS.md`.

## 5. Ką turi rodyti jūsų sąsaja?
- Lauką dokumento pavadinimui ar raktažodžiams, klausimo pasirinkimą ir mygtuką „Atsakyti“.
- GPT sugeneruotą atsakymą; aiškų pranešimą, jei informacijos nėra arba generavimas nesibaigė.
- Tris paieškos kandidatus su panašumo balais ir veikiančiomis šaltinių nuorodomis. Balas nėra atsakymo patikimumo procentas.
- Tikrą modeliui perduotą kontekstą ir bent 4 etapus: užklausa → paieška → kontekstas → GPT atsakymas. Aiškiai parodykite ir vykdymo klaidas.

Vien žalio JSON atspausdinimo nepakanka. Atsakymo negalima slapta nukopijuoti iš etalono ar pakeisti taisykle, apsimetant, kad jį sugeneravo GPT.

## 6. Eksperimentas: kas padeda, o kas trukdo?
Su **tais pačiais validavimo klausimais ir tuo pačiu modeliu** palyginkite tris sąlygas:
- Be jokio šaltinio teksto.
- Su teisinga kortele, parinkta pagal klausimo `document_id`.
- Su pirmąja paieškos atrinkta kortele.

Kiekvienai sąlygai pateikite: atsakomų klausimų tikslų sutapimą (Exact Match), teisingą `unknown` neatsakomiems, nepagrįstus atsakymus ir neužbaigtą generavimą. Atskirai apskaičiuokite, kaip dažnai paieškos pirmas rezultatas yra tinkamas dokumentas. Skaičiavimo taisyklės ir lentelės šablonas – `EXPERIMENT_RESULTS.csv` bei `TECHNICAL_DETAILS.md`.

Palyginkite dar su dviem paprastais būdais: dažniausiu mokymo atsakymu kiekvienam klausimo tipui ir atsakymo lauko nuskaitymu iš kortelės. Jei taisyklė veikia geriau už GPT, paaiškinkite kodėl – tai vertinga išvada.

**Jūsų pataisymas:** pasirinkite vieną trūkumą. Prieš bandymą užrašykite prognozę, pakeiskite vieną dalyką ir tuo pačiu validavimo rinkiniu palyginkite rezultatą prieš ir po. Galite keisti modelio dydį, mokymo nustatymą, laukų tvarką ar paieškos užklausos apdorojimą. Neigiamas rezultatas taip pat tinkamas, jei jį pagrindžiate.

Pridėkite **10 savo klausimų, bent 3 neatsakomus**. Kiekvienam nurodykite šaltinį, laukiamą ir gautą atsakymą. Atskirai pažymėkite palaikomo šablono ir laisvai suformuluotus klausimus. Testo duomenimis modelio ar paieškos nederinkite.

## 7. Ką pateikti ir ką parodyti gynime?
- Kodą: `student_model.py`, `student_search.py`, `ui.html` ir kitus savo keistus failus.
- Išsaugotą notebooką, modelio mokymo/išsaugojimo/atkūrimo instrukciją ir veikiančios sąsajos vaizdą. Išsaugotas modelis turi būti jūsų mokymo rezultatas; jo vietą nurodykite pateikime, didelių failų neviešinkite viešame Git.
- Nuostolio grafiką, mokymo trukmę, JSON/CSV rezultatų lentelę ir 10 savo klausimų bandymus.
- Pirminę prognozę, vieno pakeitimo palyginimą prieš/po ir 3 konkrečių klaidų analizę. Trumpas išvadas pateikite 1–2 puslapiuose.

**Vertinama:** ar patys įgyvendinote ir suprantate attention bei rikiavimą; ar tikrai išmokėte savo modelį; ar sąsaja sąžiningai rodo veikimą; ar eksperimentas leidžia palyginti sąlygas; ar pagrindžiate išvadas. 100 % tikslumas nėra reikalavimas. Oficialioji pažymių skalė ir terminai lieka Moodle.

Gynime be DI pagalbos išbandysite naują klausimą arba dėstytojo pakeistą kortelės faktą. Prieš paleidimą pasakysite, ko tikitės, tada paaiškinsite rastą šaltinį, GPT atsakymą ir galimą klaidą. **Neapgintas darbas neužskaitomas.** Vien kodo, ataskaitos ar veikiančio UI nepakanka; iki sėkmingo gynimo įvertinimas preliminarus, o AI įvertinimas gynimo nepakeičia.

## Darbo planas ir ryšys su paskaitomis
Pirmas 90 min. užsiėmimas: duomenys, attention, mokymas ir paieška. Antras: sąsaja, eksperimentai, pataisymas ir demonstravimas. Aplinką pasiruoškite iš anksto; individualūs gynimai gali vykti numatytais gynimų laikais. Mokymui numatykite iki 30 min., su patikromis – iki 40 min.; jei modelis konverguoja anksčiau, laukti nereikia. Kompiuterių sparta skiriasi, todėl tai nėra vienodos trukmės garantija.

7–8 paskaitos: svorių mokymas ir vertinimas; 9: transformerio attention; 10–11: mažas GPT, generavimas ir faktinės klaidos; 12: paieška ir šaltinio pateikimas modeliui. Šiame darbe sujungiate šias dalis į vieną patikrinamą sistemą.
