# Techninės detalės ir metrikų apibrėžimai

## Duomenys ir modelis
Kiekvienoje temoje500 skirtingų dokumentų ID. Mokymo/validavimo/testo dalyse350/100/50 dokumentų ir2100/200/100 klausimų–atsakymų porų. Vieno dokumento klausimai lieka vienoje dalyje. Paieška gali indeksuoti visus500 nepažymėtų dokumentų, tačiau GPT svoriai ir žodynas mokomi tik mokymo poromis. Klausimų šablonai kartojasi tarp dalių: tai siaura struktūruota užduotis, ne bendro kalbos supratimo įrodymas. Kilmė ir licencijos: SOURCES.md.

Kontekstas – trumpa metaduomenų kortelė. Vienas ar du laukai pašalinti neatsakomiems klausimams sudaryti. Atsakomumą nusako būtent šis tekstas, ne informacija kitur internete. Etaloninio atsakymo modeliui kaip konteksto neperduokite.

GPT:2 decoderio blokai; software/university pradėkite d=64,2 galvos; research d=96,4 galvos. CPU mokymas nuo atsitiktinių svorių, nuostolis atsakymo ir sekos pabaigos (EOS) tokenams. Attention TODO: QKᵀ dalijimas iš vienos galvos dimensijos šaknies, ateities kaukė PRIEŠ softmax, pasverta V suma ir galvų sujungimas. Patikros tikrina ateities nematymą, gradientų izoliaciją ir žinomą ankstesnių pozicijų vidurkio atvejį. Išsaugokite ir atkurkite savo modelį.

Paieška naudoja latentinę semantinę analizę (LSA): žodžių dažnio ir atvirkštinio dokumentų dažnio (TF-IDF) vektoriai sumažinami singuliariųjų reikšmių skaidymu (SVD). Parašykite saugų normalizavimą, kosinusą ir top-k. Paieškos vektoriai ir GPT embeddingai yra skirtingose erdvėse. Nulinis vektorius neturi sukurti NaN. Pradinėje sąsajoje pasirenkamas palaikomas klausimo tipas; laisvas pokalbis nėra šio modelio tikslas.

## Metrikos: rodykite skaitiklį ir vardiklį
- retrieval_top1: teisingas pirmasis dokumentas / visi klausimai su žinomu tiksliniu dokumentu. Taikoma paieškai; be šaltinio ir su teisinga kortele ši metrika netaikoma.
- answerable_em: tiksliai su etalonu sutapęs sugeneruotas atsakymas / atsakomi klausimai. Naudokite core.metrics normalizavimą ir tikrą generavimą, ne paduoto etalono atkartojimą.
- unanswerable_abstention: sugeneruotas tiksliai unknown / neatsakomi klausimai.
- unsupported_on_unanswerable: atsakymas nei unknown, nei tuščias / neatsakomi klausimai. Tai nepagrįsti teiginiai žinomuose neatsakomuose atvejuose, ne universali haliucinacijų metrika.
- incomplete: iki generavimo ribos nepasiektas EOS / visi klausimai. Ši klaida gali sutapti su kitomis kategorijomis.
- Atsakomuose klausimuose neteisingus ne-unknown atsakymus nagrinėkite atskirai: dalį lemia paieška, dalį GPT.3 klaidų pavyzdžiuose parodykite tikrą kontekstą.

Jei pogrupio vardiklis0, žymėkite „netaikoma“, o ne0%. CSV langeliai tušti, tai nėra išmatuoti nuliai. Greta procento pateikite kiekius. Užfiksavę nustatymus tas pačias3 sąlygas pakartokite galutiniame teste. Paprastieji būdai nenaudoja validavimo/testo etalonų mokymui: daugumos atsakymas skaičiuojamas mokyme pagal klausimo lauką; laukų skaitymo taisyklė remiasi tik pateiktu kontekstu. Šie būdai atskiri palyginimai, ne slaptas GPT atsakymo pakeitimas.

Išsaugokite pirminę prognozę. Prieš/po bandyme keiskite vieną dalyką, rinkitės pagal validavimą, lyginkite tuos pačius klausimus. Testas nenaudojamas derinimui. Nurodykite sėklą, konfigūraciją ir trukmę.100% tikslumas nėra reikalavimas.
