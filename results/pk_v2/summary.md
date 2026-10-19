# PK rule v2 recomputation

Original normalisation: `str.lower()` on both gold and generation (substring test). Abstention phrases (copied exactly): `i don't know`, `i'm not sure`, `i cannot`, `don't have enough`, `not sure`, `i do not know`, `unable to`, `no information`, `cannot determine`.

v2 = gold substring of lower-cased FIRST sentence and no abstention phrase in that first sentence. First sentence = text up to first `.`/`!`/`?` followed by whitespace/end, or first newline; periods after single-letter initials and common abbreviations (Mr, Dr, St, ...) and decimals do not end a sentence. v2_wb = same, gold must match with non-word characters (or edges) on both sides.

Sanity: v1 recomputed from the stored closed_book_answer (truncated to 300 chars by the original code, so identical to original only if gold appears in the first 300 chars).

## PK rate, test split (n=1069 questions)

| model | v1 | v2 | v2_wb | v1 cached | #flips v1->v2 (F->T / T->F) | v1 train+val+test | v2 train+val+test |
|---|---|---|---|---|---|---|---|
| llama | 2.2% | 5.1% | 5.0% | 2.2% | 31 (31 / 0) | 2.3% (n=2685) | 5.5% |
| mistral | 17.4% | 17.0% | 16.5% | 17.4% | 12 (4 / 8) | 17.2% (n=2685) | 16.7% |
| qwen | 3.9% | 10.0% | 9.7% | 3.9% | 65 (65 / 0) | 3.8% (n=2685) | 9.8% |
| qwen3 | 2.0% | 13.3% | 12.7% | 2.0% | 125 (123 / 2) | 1.8% (n=2685) | 14.1% |
| qwen14 | 3.6% | 7.0% | 6.8% | 3.6% | 43 (40 / 3) | 3.7% (n=2685) | 6.9% |
| deepseek32 | 18.2% | 18.4% | 18.0% | 18.2% | 4 (3 / 1) | 18.2% (n=1069) | 18.4% |

## Quadrant counts over test conditions (Q1 = knows & sufficient, Q2 = not-knows & sufficient, Q3 = knows & insufficient, Q4 = not-knows & insufficient; 2 conditions per question)

| model | rule | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|---|
| llama | v1 | 23 | 1046 | 23 | 1046 |
| llama | v2 | 54 | 1015 | 54 | 1015 |
| mistral | v1 | 186 | 883 | 186 | 883 |
| mistral | v2 | 182 | 887 | 182 | 887 |
| qwen | v1 | 42 | 1027 | 42 | 1027 |
| qwen | v2 | 107 | 962 | 107 | 962 |
| qwen3 | v1 | 21 | 1048 | 21 | 1048 |
| qwen3 | v2 | 142 | 927 | 142 | 927 |
| qwen14 | v1 | 38 | 1031 | 38 | 1031 |
| qwen14 | v2 | 75 | 994 | 75 | 994 |
| deepseek32 | v1 | 195 | 874 | 195 | 874 |
| deepseek32 | v2 | 197 | 872 | 197 | 872 |

(Every question has one sufficient and one insufficient condition, so Q1=Q3=#knows and Q2=Q4=#not-knows.)

## v2 PK rate by question type (test)

| model | comparative (n=102) | factual (n=283) | multi_hop (n=473) | subjective (n=211) |
|---|---|---|---|---|
| llama | 7.8% | 9.2% | 4.2% | 0.0% |
| mistral | 53.9% | 16.6% | 16.9% | 0.0% |
| qwen | 32.4% | 13.4% | 7.6% | 0.0% |
| qwen3 | 38.2% | 18.7% | 10.6% | 0.0% |
| qwen14 | 14.7% | 12.4% | 5.3% | 0.0% |
| deepseek32 | 55.9% | 21.9% | 16.5% | 0.0% |

## Frontier models

results/api_models contains only claude-opus-4-7_pk_stratified.json (valid JSON; stores aggregate pk_rate=0.3498, n_pk0=723, n_pk1=389, no closed-book text), claude-opus-4-7_results.json (MALFORMED JSON: begins mid-object at `},`) and claude-opus-4-7-cot_results.json (MALFORMED JSON: error at line 100). The valid/parseable parts contain judge_scores arrays, not closed-book generations; no GPT-5.5 files exist. **The frontier PK rates (Opus 4.7 35.0%, GPT-5.5 29.7%) cannot be recomputed under v2.**

## Ten example flips per model (v1 vs v2; first 10 in test order with v1 != v2)

### llama (31 flips)

- `squad_1312` v1=False→v2=True | Q: What is the upper house of the Parliament of Victoria called? | gold: **Legislative Council** | first sentence: 'The upper house of the Parliament of Victoria is called the Legislative Council.'
- `hotpot_2303` v1=False→v2=True | Q: Are University of Virginia and Massachusetts Institute of Technology in the same country? | gold: **yes** | first sentence: 'Yes.'
- `hotpot_204` v1=False→v2=True | Q: Which actor does American Beauty and American Beauty have in common? | gold: **Kevin Spacey** | first sentence: 'Kevin Spacey.'
- `hotpot_521` v1=False→v2=True | Q: The Government of Canada, which is the federal administration of Canada, is involved in what scandal in the province of Quebec? | gold: **The sponsorship scandal** | first sentence: 'The Sponsorship Scandal.'
- `hotpot_2486` v1=False→v2=True | Q: In what type of imprisonment where the inmates Arne Berge helped rescue during operation White Buses? | gold: **concentration camp** | first sentence: 'Concentration camps.'
- `squad_10020` v1=False→v2=True | Q: Secular Arab nationalism was blamed for both the defeat of Arab troops as well as what type of stagnation? | gold: **economic** | first sentence: 'Economic stagnation.'
- `squad_9991` v1=False→v2=True | Q: When was the Muslim Brotherhood founded? | gold: **1928** | first sentence: 'The Muslim Brotherhood was founded in 1928.'
- `squad_6989` v1=False→v2=True | Q: What rates of health and social problems are in countries with high inequality? | gold: **higher** | first sentence: 'Countries with high inequality tend to have higher rates of health and social problems, including lower life expectancy, higher infant mortality rates, and increased rates of mental illness, substance abuse, and crime.'
- `hotpot_2595` v1=False→v2=True | Q: Which poet is famous for "A Shropshire Lad" and is depicted in the play "The Invention of Love"? | gold: **A. E. Housman** | first sentence: 'A. E. Housman.'
- `hotpot_544` v1=False→v2=True | Q: The American Sweetgum is the hostplant of what kind of bug? | gold: **moth** | first sentence: 'The American Sweetgum is the hostplant of the Sweetgum Ball Moth.'

### mistral (12 flips)

- `hotpot_2303` v1=True→v2=False | Q: Are University of Virginia and Massachusetts Institute of Technology in the same country? | gold: **yes** | first sentence: 'No, University of Virginia is in the United States and Massachusetts Institute of Technology is in the United States.'
- `hotpot_59` v1=True→v2=False | Q: Are both Cypress and Ajuga genera? | gold: **no** | first sentence: 'Yes, both Cypress and Ajuga are genera.'
- `squad_9255` v1=True→v2=False | Q: What does the Nederrijn change it's name to? | gold: **Lek** | first sentence: 'The Rhine'
- `hotpot_2509` v1=False→v2=True | Q: Who is younger Jenny Bae or Lionel Richie ? | gold: **Jenny Bae** | first sentence: 'Lionel Richie is younger than Jenny Bae.'
- `squad_9930` v1=True→v2=False | Q: What does Salafism in its harshest form encourage its followers to view the religion of others with? | gold: **hate** | first sentence: 'Contempt.'
- `hotpot_1277` v1=True→v2=False | Q: Is Carnegie Hall Tower located in the same city as Staten Island? | gold: **yes** | first sentence: 'No, Carnegie Hall Tower is located in New York City, while Staten Island is a borough of New York City.'
- `hotpot_2999` v1=True→v2=False | Q: Who had more career changes, Charles A. Agemian or Arshile Gorky? | gold: **Charles A. Agemian** | first sentence: 'Arshile Gorky had more career changes.'
- `hotpot_2301` v1=False→v2=True | Q: What company started the urban complex development that included the highrise building, The Harmon? | gold: **MGM Resorts International** | first sentence: 'MGM Resorts International started the urban complex development that included the highrise building, The Harmon.'
- `hotpot_276` v1=True→v2=False | Q: Which Genus has more species Eucryphia or Lepidozamia ? | gold: **Eucryphia** | first sentence: 'Lepidozamia has more species.'
- `squad_9605` v1=False→v2=True | Q: What is the parliamentary mace made of? | gold: **silver** | first sentence: 'The parliamentary mace is made of gilded silver.'

### qwen (65 flips)

- `squad_1312` v1=False→v2=True | Q: What is the upper house of the Parliament of Victoria called? | gold: **Legislative Council** | first sentence: 'The upper house of the Parliament of Victoria is called the Legislative Council.'
- `squad_10718` v1=False→v2=True | Q: What type of housing was erected in Warsaw as part of the Bricks for Warsaw process? | gold: **prefabricated** | first sentence: 'As part of the Bricks for Warsaw process, prefabricated panel apartments were erected in Warsaw.'
- `squad_208` v1=False→v2=True | Q: What branch of theoretical computer science deals with broadly classifying computational problems by difficulty and class of relationship? | gold: **Computational complexity theory** | first sentence: 'Computational complexity theory.'
- `hotpot_698` v1=False→v2=True | Q: What type of publication does St. Anthony Messenger and Revolver have in common? | gold: **magazine** | first sentence: 'Both St. Anthony Messenger and Revolver are magazines.'
- `hotpot_1032` v1=False→v2=True | Q: Which canal is located further North, Beaver and Erie Canal or the Dismal Swamp Canal? | gold: **The Beaver and Erie Canal** | first sentence: 'The Beaver and Erie Canal is located further North than the Dismal Swamp Canal.'
- `hotpot_1960` v1=False→v2=True | Q: Which is a university in Pennsylvania, University of New Haven or Carnegie Mellon University? | gold: **Carnegie Mellon University** | first sentence: 'Carnegie Mellon University is a university in Pennsylvania.'
- `squad_1515` v1=False→v2=True | Q: What northern province in France has a large protestant population? | gold: **Alsace** | first sentence: 'The northern province in France with a large Protestant population is Alsace.'
- `squad_2618` v1=False→v2=True | Q: What characteristic of oxygen causes it to form bonds with other elements? | gold: **electronegativity** | first sentence: "Oxygen's high electronegativity causes it to form bonds with other elements."
- `squad_6951` v1=False→v2=True | Q: Who wrote the book "Capital in the Twenty-First Century"? | gold: **Thomas Piketty** | first sentence: 'The book "Capital in the Twenty-First Century" was written by Thomas Piketty.'
- `hotpot_276` v1=False→v2=True | Q: Which Genus has more species Eucryphia or Lepidozamia ? | gold: **Eucryphia** | first sentence: 'Lepidozamia has more species than Eucryphia.'

### qwen3 (125 flips)

- `squad_9195` v1=False→v2=True | Q: What is Europe's largest inland port? | gold: **Duisburg** | first sentence: "Europe's largest inland port is Duisburg, Germany."
- `hotpot_653` v1=False→v2=True | Q: What kind of musicians are Mark Gaudet and Jan Axel Blomberg? | gold: **drummer** | first sentence: 'Mark Gaudet and Jan Axel Blomberg are both musicians known for their work in the heavy metal genre, with Gaudet being a member of the band Theocracy and Blomberg being the drummer for the band Enslaved.'
- `hotpot_2397` v1=False→v2=True | Q: Which refers to more breeds, Stabyhoun or Griffon Bruxellois? | gold: **Griffon Bruxellois** | first sentence: 'The Griffon Bruxellois refers to more breeds.'
- `hotpot_672` v1=False→v2=True | Q: Angelo Francesco Lavignino scored a 1965 English-language Spanish-Swiss co-produced film that was directed by and also starred who? | gold: **Orson Welles** | first sentence: 'Angelo Francesco Lavignino scored a 1965 English-language Spanish-Swiss co-produced film that was directed by and also starred Orson Welles.'
- `hotpot_525` v1=False→v2=True | Q: Are Chlidanthus and Maianthemum both flowering plants? | gold: **yes** | first sentence: 'Yes, both Chlidanthus and Maianthemum are flowering plants.'
- `hotpot_707` v1=False→v2=True | Q: Which movie was released first, 'Mean Girls' or the remake of 'April Fool's Day'? | gold: **Mean Girls** | first sentence: "The movie 'Mean Girls' was released first."
- `hotpot_1704` v1=False→v2=True | Q: The location of the area where Fort Nisqually stood is in what Washington county? | gold: **Pierce County** | first sentence: 'The location of the area where Fort Nisqually stood is in Pierce County, Washington.'
- `hotpot_310` v1=False→v2=True | Q: What is the year of the event that occured first, Making Today a Perfect Day was produced, or Frozen was produced? | gold: **2013** | first sentence: 'The event that occurred first was the production of "Frozen," which was released in 2013, while "Making Today a Perfect Day" was produced later.'
- `squad_2079` v1=False→v2=True | Q: Who developed the first commercial steam powered device? | gold: **Thomas Savery** | first sentence: 'The first commercial steam-powered device was developed by Thomas Savery in 1698 with his steam pump, which was used for draining water from mines.'
- `hotpot_2494` v1=False→v2=True | Q: The technique 'panenka' was first used by a player of what nationality against "Die Katze von Anzing"?  | gold: **Czech** | first sentence: 'The technique \'panenka\' was first used by a Czech player against "Die Katze von Anzing".'

### qwen14 (43 flips)

- `squad_9195` v1=False→v2=True | Q: What is Europe's largest inland port? | gold: **Duisburg** | first sentence: "The Europe's largest inland port is the Port of Duisburg in Germany."
- `squad_10282` v1=False→v2=True | Q: Imperialism is confused with what other term? | gold: **colonialism** | first sentence: 'Imperialism is sometimes confused with colonialism.'
- `squad_6058` v1=False→v2=True | Q: What article of the Grundgesetz grants the right to make private schools? | gold: **7** | first sentence: 'Article 7 of the Grundgesetz grants the right to make private schools.'
- `hotpot_958` v1=True→v2=False | Q: Do Bloody Mary and Sidecar share any ingredients in common? | gold: **no** | first sentence: 'Yes, Bloody Mary and Sidecar both can contain lemon juice as a common ingredient.'
- `hotpot_1131` v1=False→v2=True | Q: Are both Dziga Vertov and Roger Donaldson are involved in the film industry? | gold: **yes** | first sentence: 'Yes, both Dziga Vertov and Roger Donaldson have been involved in the film industry, with Vertov as a pioneering documentary filmmaker and Donaldson as a director and producer of various films.'
- `hotpot_1960` v1=False→v2=True | Q: Which is a university in Pennsylvania, University of New Haven or Carnegie Mellon University? | gold: **Carnegie Mellon University** | first sentence: 'Carnegie Mellon University is a university located in Pennsylvania.'
- `squad_9991` v1=False→v2=True | Q: When was the Muslim Brotherhood founded? | gold: **1928** | first sentence: 'The Muslim Brotherhood was founded in 1928.'
- `squad_2618` v1=False→v2=True | Q: What characteristic of oxygen causes it to form bonds with other elements? | gold: **electronegativity** | first sentence: "Oxygen's high electronegativity causes it to form bonds with other elements."
- `hotpot_2672` v1=False→v2=True | Q: The Shōgitai were an elite corps of which shogunate? | gold: **Tokugawa shogunate** | first sentence: 'The Shōgitai were an elite corps of the Tokugawa shogunate.'
- `hotpot_826` v1=False→v2=True | Q: Which reservation is located in Hill and Chouteau and named after the principal leader of the Montana Chippewa-Cree? | gold: **Rocky Boy's Indian Reservation** | first sentence: "The Rocky Boy's Indian Reservation is located in Hill and Chouteau counties, Montana, and is named after Chief Rocky Boy, the principal leader of the Montana Chippewa-Cree."

### deepseek32 (4 flips)

- `squad_6058` v1=False→v2=True | Q: What article of the Grundgesetz grants the right to make private schools? | gold: **7** | first sentence: 'Article 7, Paragraph 1 of the Grundgesetz grants the right to make private schools.'
- `hotpot_289` v1=True→v2=False | Q: Which came to market first, "Hey Pa! There's a Goat on the Roof" or "Poleconomy" | gold: **Hey Pa! There's a Goat on the Roof** | first sentence: '"Hey Pa!'
- `hotpot_310` v1=False→v2=True | Q: What is the year of the event that occured first, Making Today a Perfect Day was produced, or Frozen was produced? | gold: **2013** | first sentence: 'Making Today a Perfect Day was produced in 2017, while Frozen was produced in 2013.'
- `squad_2079` v1=False→v2=True | Q: Who developed the first commercial steam powered device? | gold: **Thomas Savery** | first sentence: 'Thomas Savery'

## PK-stratified AUROC (sufficient vs insufficient; question-clustered bootstrap, 2000 resamples, seed 0, 95% CI)

Overall AUROC does not depend on the PK rule. qwen3 VC = chat-nothink.jsonl `score`; qwen3 has no probe. Strata with <50 valid bootstrap resamples (single class) show CI n/a.

| model | method | overall | v1 PK=0 | v1 PK=1 | v2 PK=0 | v2 PK=1 |
|---|---|---|---|---|---|---|
| llama | Standard Probe | 0.907 [0.895, 0.917] | 0.906 [0.894, 0.917] (n=1046) | 0.924 [0.827, 0.993] (n=23) | 0.905 [0.894, 0.916] (n=1015) | 0.931 [0.877, 0.978] (n=54) |
| llama | CSP Probe | 0.904 [0.893, 0.915] | 0.904 [0.892, 0.916] (n=1046) | 0.900 [0.812, 0.978] (n=23) | 0.903 [0.891, 0.915] (n=1015) | 0.921 [0.869, 0.969] (n=54) |
| llama | Verbalized Conf. | 0.717 [0.702, 0.732] | 0.716 [0.701, 0.731] (n=1046) | 0.761 [0.656, 0.867] (n=23) | 0.716 [0.700, 0.731] (n=1015) | 0.741 [0.677, 0.808] (n=54) |
| mistral | Standard Probe | 0.910 [0.899, 0.920] | 0.903 [0.891, 0.915] (n=883) | 0.941 [0.919, 0.960] (n=186) | 0.903 [0.891, 0.915] (n=887) | 0.943 [0.921, 0.962] (n=182) |
| mistral | CSP Probe | 0.902 [0.890, 0.912] | 0.897 [0.884, 0.910] (n=883) | 0.923 [0.899, 0.946] (n=186) | 0.896 [0.883, 0.909] (n=887) | 0.928 [0.904, 0.950] (n=182) |
| mistral | Verbalized Conf. | 0.700 [0.683, 0.714] | 0.695 [0.677, 0.711] (n=883) | 0.721 [0.684, 0.756] (n=186) | 0.695 [0.678, 0.712] (n=887) | 0.720 [0.684, 0.755] (n=182) |
| qwen | Standard Probe | 0.904 [0.892, 0.915] | 0.904 [0.892, 0.915] (n=1027) | 0.897 [0.825, 0.955] (n=42) | 0.902 [0.890, 0.914] (n=962) | 0.915 [0.879, 0.947] (n=107) |
| qwen | CSP Probe | 0.914 [0.903, 0.924] | 0.913 [0.901, 0.923] (n=1027) | 0.939 [0.890, 0.980] (n=42) | 0.911 [0.899, 0.921] (n=962) | 0.941 [0.913, 0.967] (n=107) |
| qwen | Verbalized Conf. | 0.780 [0.764, 0.796] | 0.781 [0.765, 0.798] (n=1027) | 0.775 [0.686, 0.859] (n=42) | 0.778 [0.761, 0.794] (n=962) | 0.806 [0.757, 0.854] (n=107) |
| qwen3 | Verbalized Conf. | 0.781 [0.766, 0.795] | 0.781 [0.765, 0.796] (n=1048) | 0.766 [0.638, 0.886] (n=21) | 0.775 [0.758, 0.791] (n=927) | 0.824 [0.781, 0.863] (n=142) |
| qwen14 | Standard Probe | 0.940 [0.931, 0.949] | 0.939 [0.931, 0.949] (n=1031) | 0.943 [0.888, 0.987] (n=38) | 0.939 [0.930, 0.948] (n=994) | 0.951 [0.921, 0.976] (n=75) |
| qwen14 | CSP Probe | 0.949 [0.941, 0.957] | 0.949 [0.941, 0.957] (n=1031) | 0.936 [0.881, 0.982] (n=38) | 0.949 [0.940, 0.957] (n=994) | 0.949 [0.916, 0.978] (n=75) |
| qwen14 | Verbalized Conf. | 0.818 [0.804, 0.833] | 0.818 [0.804, 0.833] (n=1031) | 0.846 [0.768, 0.916] (n=38) | 0.817 [0.803, 0.833] (n=994) | 0.850 [0.794, 0.902] (n=75) |
| deepseek32 | Standard Probe | 0.949 [0.940, 0.957] | 0.948 [0.938, 0.957] (n=874) | 0.953 [0.934, 0.970] (n=195) | 0.947 [0.938, 0.956] (n=872) | 0.955 [0.936, 0.972] (n=197) |
| deepseek32 | CSP Probe | 0.946 [0.938, 0.954] | 0.947 [0.938, 0.956] (n=874) | 0.941 [0.918, 0.962] (n=195) | 0.947 [0.937, 0.955] (n=872) | 0.943 [0.921, 0.964] (n=197) |
| deepseek32 | Verbalized Conf. | 0.803 [0.788, 0.819] | 0.803 [0.786, 0.822] (n=874) | 0.801 [0.765, 0.836] (n=195) | 0.803 [0.786, 0.821] (n=872) | 0.803 [0.768, 0.838] (n=197) |
