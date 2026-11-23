# 2 laboratorinis darbas — Rekomendacija platformai

Vardas, pavardė: Akvilė Jaruševičiūtė DIEf-24 
Data: 2026-10-22  
Python / pandas / scikit-learn versijos: Python 3.14.7 / pandas 3.0.6 / scikit-learn 1.9.1 (pagal projekto `.venv` aplinkos konfigūraciją ir įdiegtų paketų metaduomenis)

Duomenų rinkinys (vienas): `books_gutenberg_4000.csv`.  
Kūrinių po valymo: **4000** (pradinis `books_gutenberg_4000.csv` turi 4000 eilučių; po B/C kode taikomų pavadinimo tikrinimo ir dublikatų šalinimo veiksmų liko 4000 kūrinių).  
Pašalinta tuščių / pasikartojančių: **0 / 0**; duomenų faile nėra tuščių pavadinimų, o dublikatų pagal `title` po pavadinimų apkarpymo nerasta.  
Tekstinis laukas (iš kokių stulpelių ir kodėl): `subjects + bookshelves + title`. Temos ir lentynos suteikia teminį / kategorijų kontekstą, o pavadinimas prideda kūrinio identifikavimo ir teminius žodžius.  
TF-IDF nustatymai (tokenizavimas, stop-žodžiai, min_df / max_features, n-gramos): mažosios raidės (`lowercase=True`); tokenizavimo šablonas `(?u)\b\w\w+\b` (žodžiai iš bent 2 simbolių); B naudoja `stop_words="english"` ir randa **4095 požymius**; C papildomai šalina `fiction`, `drama`, `poetry`, `novel`, `novels`, `short`, `stories`, `drama film`, `fiction film` ir randa **4088 požymius**. `min_df=2`; `max_features=20000`; `sublinear_tf=True`; n-gramos nenurodytos, todėl naudojama numatytoji reikšmė `(1, 1)` (unigramos). Kadangi `drama film` ir `fiction film` tokenizuojami į atskirus žodžius, šios frazės nėra pašalinamos kaip vientisi požymiai; kodas pateikia apie tai įspėjimą.  
Sėkla ir užklausų parinkimo metodas: kode apibrėžta `SEED = 20261021`, tačiau šiame eksperimente užklausos neatsitiktinės — naudojamos penkios iš anksto nurodytos knygos. Todėl sėkla užklausų atrankai faktiškai nenaudojama.  
Tuščių tekstinių laukų po paruošimo: **0**. Tekstas sudaromas sujungiant `subjects`, `bookshelves` ir `title`; visuose 4000 įrašų po paruošimo tekstinis laukas netuščias.

## 1. Pradinė analizė (be DI)

| Užklausos kūrinys | Kokių rekomendacijų tikiuosi ir kodėl | Sunki? |
| --- | --- | --- |
| Q1: The Balkans: A History of Bulgaria—Serbia—Greece—Rumania—Turkey | Knygų apie Balkanų istoriją, Graikiją, Turkiją, Rumuniją ar Serbiją, nes pavadinimas ir temos nurodo geografinę bei istorinę sritį. | Vidutinė |
| Q2: As I Remember Recollections of American Society during the Nineteenth Century | Atsiminimų, biografijų ir XIX a. JAV visuomenės istorijos kūrinių, nes temos nurodo socialinius papročius ir konkrečias JAV vietoves. | Sunki |
| Q3: Alice in Wonderland A Dramatization of Lewis Carroll's "Alice's Adventures in Wonderland" and "Through the Looking Glass" | Kitos „Alice“ istorijos versijos, „Through the Looking-Glass“ ir vaikų fantazijos / pjesių kūriniai. | Lengva |
| Q4: The Battle of the Books, and other Short Pieces | Eseistikos, satyros ir literatūros kritikos kūrinių, nes temos nurodo anglų esė, satyrą ir ginčą tarp senųjų bei moderniųjų autorių. | Vidutinė |
| Q5: The Suffrage Cook Book | Moterų rinkimų teisės istorijos, moterų teisių ir galbūt panašaus laikotarpio visuomeninių judėjimų knygų. Kulinarijos žyma gali nukreipti rekomendacijas į receptų knygas. | Sunki |

## 2. Kontroliuojamas pakeitimas: prognozė prieš bandymą

Kas keičiama: **Run C** pradinį anglų kalbos stop-žodžių sąrašą papildo bendriniais žanro / lentynų terminais, pvz., `fiction`, `novel`, `poetry`, `drama`, `short`, `stories`. Kiti pagrindiniai TF-IDF parametrai ir tekstiniai laukai paliekami tokie patys.

Kas turėtų pagerėti: bendrinės žymos turėtų mažiau veikti knygų panašumo balus, o konkretesni temos žodžiai galėtų turėti daugiau įtakos. Tikėtina, kad rekomendacijų tvarka kai kur pasikeis.

Kas gali pablogėti: bendriniai žodžiai kartais yra prasmingi. Juos pašalinus gali sumažėti ryšys tarp panašių žanro ar formos kūrinių. Taip pat pakeitimas gali beveik nepaveikti rezultatų, jei papildomi žodžiai konkrečiuose tekstuose reti arba kitų požymių svoriai jau dominuoja.

## 3. Trys paleidimai tose pačiose užklausose

A modelis pateikia tą patį penkių daugiausiai atsisiųstų knygų sąrašą kiekvienai užklausai: **Pride and Prejudice** (198164), **Moby Dick; Or, The Whale** (164301), **The Secret of Chimneys** (133305), **The Odyssey Rendered into English prose for the use of those who cannot read the original** (120932), **The Adventures of Sherlock Holmes** (119794). Tai populiarumo, o ne turinio panašumo rekomendacijos.

| Užklausa | A · populiarumo bazinis | B · pradinis TF-IDF | C · pakeistas TF-IDF | Sutapimai su 1 dalimi |
| --- | --- | --- | --- | --- |
| Q1: The Balkans | Tas pats populiariausių knygų sąrašas; tiesioginio teminio ryšio su Balkanų istorija nematyti. | *Essays on the Latin Orient* (0.3672); *Arms and the Man* (0.2703); *The Earth and its inhabitants, Volume 1: Europe...* (0.2540); *Hero Tales and Legends of the Serbians* (0.2170); *Afghanistan* (0.1884). | *Essays on the Latin Orient* (0.3672); *Arms and the Man* (0.3002); *The Earth and its inhabitants, Volume 1: Europe...* (0.2540); *Hero Tales and Legends of the Serbians* (0.2202); *The Poems of Sappho...* (0.2123). | B: 1/5; C: 1/5; A: 0/5. *Essays on the Latin Orient* atitinka pradinį lūkestį. |
| Q2: As I Remember | Tas pats populiariausių knygų sąrašas; neatsižvelgia į atsiminimų ar JAV istorijos temą. | *A scarlet poppy, and other stories* (0.5112); *The Belle of Bowling Green* (0.4709); *A Book for a Rainy Day...* (0.4692); *And life is better* (0.4508); *The Open Boat and Other Stories* (0.4249). | *A scarlet poppy, and other stories* (0.5716); *The Belle of Bowling Green* (0.4909); *The Open Boat and Other Stories* (0.4840); *Tales of the Jazz Age* (0.4741); *And life is better* (0.4719). | B: 0/5; C: 0/5; A: 0/5. Lauktų istorijos / atsiminimų rekomendacijų pagal vertinimo sąrašą nerasta. |
| Q3: Alice in Wonderland | Tas pats populiariausių knygų sąrašas; teminis ryšys silpnas. | *Alice's Adventures in Wonderland* (0.6272); *Alice's Adventures Under Ground...* (0.5732); *Alice in Wonderland, Retold in Words of One Syllable* (0.5481); *Through the Looking-Glass* (0.5479); *Alice's Adventures in Wonderland Illustrated...* (0.5216). | *Alice's Adventures in Wonderland* (0.6539); *Alice's Adventures Under Ground...* (0.5912); *Through the Looking-Glass* (0.5717); *Alice in Wonderland, Retold in Words of One Syllable* (0.5704); *Alice's Adventures in Wonderland Illustrated...* (0.5408). | B: 0/5; C: 0/5; A: 0/5 pagal griežtą iš anksto nustatytą vertinimo sąrašą. Tačiau B ir C rezultatai akivaizdžiai atitinka „Alice“ temą; tai rodo vertinimo sąrašo ribotumą. |
| Q4: The Battle of the Books | Tas pats populiariausių knygų sąrašas; nesusijęs su esė ir satyra. | *Europe and elsewhere* (0.4429); *Bacon's Essays, and Wisdom of the Ancients* (0.4339); *The Idle Thoughts of an Idle Fellow* (0.4129); *Modern Essays* (0.3881); *Suspended Judgments: Essays on Books and Sensations* (0.3865). | *Bacon's Essays, and Wisdom of the Ancients* (0.4417); *Europe and elsewhere* (0.4260); *The Idle Thoughts of an Idle Fellow* (0.4204); *Modern Essays* (0.3951); *Suspended Judgments: Essays on Books and Sensations* (0.3935). | B: 1/5; C: 1/5; A: 0/5. *Bacon's Essays...* atitinka lūkestį; C šią rekomendaciją pakelia į pirmą vietą. |
| Q5: The Suffrage Cook Book | Tas pats populiariausių knygų sąrašas; tiesioginio ryšio su moterų rinkimų teise nėra. | *History of Woman Suffrage, Volume I* (0.6107); *The Complete Book of Cheese* (0.5960); *History of Woman Suffrage, Volume II* (0.5908); *The History of Woman Suffrage, Volume V* (0.5848); *History of Woman Suffrage, Volume III* (0.5791). | Tokios pačios penkios rekomendacijos ir tokie patys panašumo balai kaip B. | B: 4/5; C: 4/5; A: 0/5. Keturi tomai atitinka moterų rinkimų teisės temą, bet sūrio knyga tikriausiai rekomenduota dėl kulinarijos žodžių. |

Ar prognozė pasitvirtino? **Iš dalies.** C pakeitė kai kurių rekomendacijų tvarką ir balus: Q1 knygos *Arms and the Man* balas padidėjo nuo 0.2703 iki 0.3002; Q4 *Bacon's Essays, and Wisdom of the Ancients* pakilo iš antros į pirmą vietą. Tačiau Q5 rezultatai nepasikeitė, o bendras sutapimų su rankiniu vertinimo sąrašu skaičius liko toks pats pagal kiekvieną užklausą. Nauda – kai kur pasikeitė prioritetų tvarka; kaina – pagerėjimas nėra nuoseklus, o pašalinus bendrinius žodžius dalis prasmingų žanro signalų taip pat gali būti prarasta.

## 4. Trys nesėkmingos rekomendacijos

| Užklausa ir sistemos atsakymas | Kodėl skaitytojui ar žiūrovui netinka | Tikėtina priežastis | Sprendimas arba kodėl neišvengiama |
| --- | --- | --- | --- |
| Q2 *As I Remember...* → *A scarlet poppy, and other stories* | Pavadinimas savaime nerodo ryšio su XIX a. JAV visuomenės atsiminimais; rankiniame vertinimo sąraše šios rekomendacijos nebuvo numatytos. | Modelis naudoja tik pavadinimą, `subjects` ir `bookshelves`; galėjo sutapti bendriniai žodžiai ar lentynų žymos. | Pridėti arba patikslinti metaduomenis, įvertinti teminį panašumą žmogui; prireikus naudoti autorių, laikotarpio ar geografijos požymius. |
| Q5 *The Suffrage Cook Book* → *The Complete Book of Cheese* | Kulinarijos tema panaši, bet knyga apie sūrį greičiausiai neatitinka pagrindinio lūkesčio apie moterų rinkimų teisę. | Užklausoje vienu metu yra kulinarijos ir rinkimų teisės žodžių; bendras vektorius sujungia abi temas. | Sverti pagrindinę temą labiau arba atskirai vertinti temų laukus; galutines rekomendacijas patikrinti redaktoriui. |
| Q1 *The Balkans...* → *Arms and the Man* | Iš vien pavadinimo neaišku, ar kūrinys iš tikrųjų susijęs su Balkanų istorija; gali būti silpnas semantinis atitikimas. | Tekstinis vektorius remiasi žodžių sutapimu, o kosinusinis panašumas nėra semantinio tinkamumo garantija. | Patikrinti kūrinio metaduomenis / aprašymą ir, jei įmanoma, naudoti išsamesnį turinį arba žmogaus vertinimą. |

Atvejis, kur bazinis modelis atrodė geriau (arba kodėl neradote): **populiarumo modelis kai kur gali būti geresnis bendram atradimui**, nes siūlo daug skaitomas knygas, net jei jos nėra tematiškai panašios. Tačiau šiose penkiose tikslinėse užklausose A modelis nepateikė nė vienos iš rankiniu būdu numatytų rekomendacijų. B ir C palyginime C nepademonstravo visuotinio pagerėjimo: Q5 rezultatai identiški, o Q2 nė viena rekomendacija nesutapo su vertinimo sąrašu.

## 5. Rekomendacija platformai

Pasirinktas variantas ir kompromisas: **naudoti TF-IDF kaip teminio rekomendavimo pagrindą, o populiarumą – kaip atskirą papildomą rekomendacijų sąrašą.** Šiame bandyme B ir C nėra aiškaus įrodymo, kad C visada geresnis. C verta palikti kaip kontroliuojamą variantą, tačiau pasirinkimą reikėtų patvirtinti didesniu vertinimo rinkiniu ir žmogaus įvertintomis rekomendacijomis. Populiarumo modelis paprastas ir naudingas bendram atradimui, bet neatliepia konkrečios užklausos temos.

Kam dar reikia žmogaus redaktoriaus: dviprasmiškų ar kelių temų kūrinių, tokių kaip *The Suffrage Cook Book*, istorinių / geografinių temų ir rekomendacijų, kurių pavadinimas atrodo panašus, bet turinys gali skirtis. Žmogus taip pat turėtų patikrinti, ar rankinis vertinimo sąrašas išsamus, nes „Alice“ rekomendacijos atrodo tematiškai tinkamos, nors automatinis tikrinimas jas pažymi kaip nesutampančias.

## 6. Šaltiniai ir DI deklaracija

Šaltiniai: `books_gutenberg_4000.csv`, `books_clean.csv`, `books_sample_5.csv`, `run_a_results.csv`, `run_b_results.csv`, `run_c_results.csv`, `experiment_results.csv`, `2task.py`, `xperiment.py` ir laboratorinio darbo `report-template.md` iš pateikto projekto archyvo. Skaičiai ir rekomendacijų pavadinimai 3 dalyje paimti iš rezultatų CSV; duomenų dydis, pašalintų eilučių skaičius ir TF-IDF požymių skaičius patikrinti pagal duomenų failą bei kodą.

