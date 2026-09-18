# Decisioni Astra — 15/09/2026 01:10 (sessione `codex exec` nuova: il thread 01a09a09 era bloccato dal lock dell'app)

Domande poste (in sintesi) e risposte letterali di Astra (gpt-6-astra, effort max), dopo lettura di
DECISIONI-ASTRA-20260914.md, RISULTATI-GATE-BACKEND-20260914.md, README.md, DESIGN.md e configs/p1_mi300x.toml.

1. **Compilazione FlexAttention con forme variabili** (misurato: dynamic=True 0,45 s/passo, statica 0,44 s, nessuna ricompilazione).
   > Confermo `dynamic=True`, cache 64. Traccia le ricompilazioni sui cambi di lunghezza e λ; l'esaurimento della cache deve emergere esplicitamente, senza fallback silenzioso.

2. **Equivalenza testo puro** (wrapped vs nativo entro il rumore nativo SDPA/eager sullo stesso prompt, argmax 100 % su 512 token).
   > PASS numerico bf16 sul controllo riportato. Il riferimento relativo sullo stesso input è corretto. Completa anche il confronto sul prompt originario da 144 token, registrando media, p99/max e argmax; può procedere insieme a P1.

3. **Valori REQUIRED di configs/p1_mi300x.toml** (AdamW 0.9/0.999/1e-8; warmup 3 % + coseno; checkpoint ogni 100 step; selezione = primary diretta su validation, tiebreak loss fixed-crop; CE media per segmento poi somma pesata; LoRA P3 5e-5).
   > Confermo tutti i valori, incluso LoRA P3 `5e-5`; selezione diretta valida per P1. Nel trainer locale completa il tie-break fixed-crop e verifica il formato tramite parser; la validation ufficiale deve consentire **768 token finali**, mentre il default attuale è 256.

4. **P1 fixed-crop**: azione teacher nel contesto ma non supervisionata; crop teacher log-uniformi [256, 8192], 70 % positivi a posizione interna casuale (20 % parziali), 30 % casuali; 2 canali con prob 0,5 (lead_lag) e 0,6 (16 canali).
   > Confermo per P1, con azione teacher interamente esclusa dalla loss. Il 20 % è condizionato ai positivi; registra negativi e parziali **effettivi** per categoria, perché le probabilità del sampler non garantiscono quelle frequenze. Confronti multicanale a offset diversi restano richiesti in P2.

5. **Kernel lineari**: risolto con fla-core 0.5.2 (0,45 s/passo a 4k token, 29 GiB).
   > Recepito. Fissa `fla-core==0.5.2` nell'ambiente riproducibile; per il budget usa il tempo end-to-end del trainer.

6. **Ordine con 100 ore di credito**: P0 smoke → benchmark 50 passi → P1 intero → validation 520 → decidere P2.
   > Confermo l'ordine; P1 prosegue senza riavvio per queste decisioni. Dopo validation 520, P2 richiede anche il gate percettivo §12.2: vero–miglior controllo con CI95 sopra zero, categorie senza collassi e formati invalidi sotto l'1 % nelle modalità valutate. Pianifica sul credito residuo includendo P3 e valutazioni.

## Esecuzione delle risposte (15/09)

- §1: `WildcardContext.flex_fn` ora registra le ricompilazioni con motivo e imposta `fail_on_recompile_limit_hit`
  (nessun fallback eager silenzioso). In vigore dal prossimo lancio: P1 `p1-main` gira con il codice precedente
  (dynamic=True, cache 64), come autorizzato.
- §2: `tools/text_equivalence_probe.py`, eseguito in parallelo a P1 (report in `integrity/artifacts/`).
- §3: `training/train.py`: default 768 token finali, formato verificato con `parse_structured_pred`, tie-break =
  loss fixed-crop su validation (crop teacher seedati); la validation ufficiale del checkpoint finale di P1 viene
  eseguita con `training/validate.py` (768 token) e non dal ciclo interno della run in corso.
- §4: `crop_kind` (positive_full / positive_partial / negative) registrato per ogni esempio fixed-crop; per la run in
  corso viene ricostruito a posteriori da `tools/report_run.py --train-parquet`.
- §5: `droplet/bootstrap.sh` pinna `fla-core==0.5.2`.
- §6: gate percettivo §12.2 da implementare (`evaluation/perceptual_gate.py`) prima di P2.

# Decisioni Astra su P2 — 15/09/2026 01:25 (nuova sessione, effort max; ha letto DECISIONI 15/09, DESIGN §9-10.4, templates/data/model)

Domande poste: (1) teacher delle traiettorie programmatico vs LLM; (2) tipi e quote sul 50 % agentico
(proposta 15/30/20/10/15/10); (3) supervisione dell'azione errata iniettata; (4) ragionamento nel blocco `<think>`;
(5) selettore Haar cieco di L3 per le finestre non-gold; (6) selezione del checkpoint P2 sulla primary agentica;
(7) tre passate contro due. Risposte letterali:

1. **Teacher programmatico**, riproducibile con seed per ID/passata. Gold per target e pianificazione; rationale da misure sulle **sole viste già acquisite, alla risoluzione disponibile**. Tipicamente 40–100 token, massimo 384 **per decisione**; classi e confini annotati non diventano automaticamente osservazioni.

2. **Ridistribuisco: 15/25/15/10/15/10%** nell’ordine proposto, aggiungendo **10% più eventi distinti**, anche con ≥3 zoom. Quote sui casi compatibili; confronti multicanale metà sincroni, metà a offset diversi. Il finale dopo crop negativo richiede evidenza complessiva sufficiente: un vuoto locale non dimostra assenza globale.

3. **Confermo 0.2/1/1.** Errore deliberatamente iniettato: CE zero per azione e rationale introduttiva, entrambe conservate nel contesto; supervisiono revisione, recupero e finale. Un’esplorazione legittima risultata negativa mantiene azione a 1; azioni fixed-crop a zero. Serve una **maschera per turno**, oggi assente.

4. **Sì a `<think>`**, seguito da azione/finale nello stesso turno assistant; i parser ricevono soltanto il contenuto dopo `</think>`. Prima di P2 servono reasoning+action, prefisso thinking-on verificato sul tokenizer, supervisione della chiusura e conservazione della storia. Oggi `generate()` forza thinking-off e ricostruisce la storia dalla sola lista di azioni.

5. **Sì al generatore Haar**, cieco anche nella scelta del canale; nel train il gold filtra i candidati negativi rispetto a **tutti** gli intervalli dei canali richiesti. Riusa il calcolo dei punteggi/candidati: il pianificatore W1 completo sceglie invece il canale dal gold e comprende un `gold_fallback`.

6. **Confermo: primary agentica su val520, tie-break primary diretta**, con valutazione a ogni passata, invalidi <1% per modalità e nessun collasso per categoria. Budget confermati, inclusi 4096 token assistant complessivi e riserva finale 768. Riporta copertura di tutti gli eventi e direzione lead_lag; ricontrolla il gate sul prescelto, anche con azioni autonome congelate.

7. **Tre passate pianificate; nessuna riduzione automatica a due.** Gate inconcludente/fallito ⇒ diagnosi percettiva prima di P2; un PASS piccolo va confrontato con una soglia minima utile fissata prima dei risultati. Alle latenze indicate sono **9,4–15 h di training**, più ~6 h per tre valutazioni agentiche: verifica il tempo reale e riserva esplicitamente il budget per P3.

## Conseguenze implementative (da fare prima di P2, dopo il gate percettivo)

- Teacher programmatico seedato per (ID, passata): pianificazione dal gold, rationale calcolate dalle sole viste
  acquisite alla risoluzione disponibile (40–100 token, max 384 per decisione), mai classi/confini annotati come
  osservazioni.
- Quote sul 50 % agentico: finale senza tool 15 %, un zoom corretto 25 %, prima finestra errata + recupero 15 %,
  raffinamento confini 10 %, confronto due canali 15 % (metà sincroni, metà a offset), crop negativo con arresto per
  evidenza sufficiente 10 %, più eventi distinti anche con ≥3 zoom 10 %.
- Maschera di supervisione **per turno**: azione errata iniettata e sua rationale introduttiva a CE zero (restano nel
  contesto); esplorazione legittima negativa a peso 1; azioni fixed-crop a zero; revisione, recupero e finale supervisionati.
- Template: ragionamento in `<think>…</think>` seguito da azione o finale nello stesso turno assistant; parser solo sul
  contenuto dopo `</think>`; prefisso thinking-on verificato sul tokenizer; supervisione della chiusura; `generate()`
  agentico con storia reale dei turni generati (oggi forza thinking-off e ricostruisce dalla sola lista di azioni).
- Selettore Haar cieco (anche nel canale) per candidati non-gold, riusando il calcolo dei punteggi di L3
  (`l3/zoom_traces_w1.py`); nel train il gold filtra i candidati negativi rispetto a tutti gli intervalli dei canali richiesti.
- Selezione P2: primary agentica su val520 con tie-break primary diretta, valutazione a ogni passata, invalidi < 1 % per
  modalità, nessun collasso per categoria; report di copertura degli eventi e direzione lead_lag; gate percettivo
  ricontrollato sul checkpoint scelto anche con azioni autonome congelate. Budget: 4096 token assistant, riserva finale 768.
- Tre passate pianificate, nessuna riduzione automatica; gate inconcludente ⇒ diagnosi percettiva prima di P2; soglia
  minima utile fissata prima dei risultati; stima 9,4–15 h di training + ~6 h di valutazioni; riservare budget per P3.


## Decisioni Astra sulla scala dei soft token (15/09, sessione Codex `01a0a2bd`, ~03:50 locali)

Contesto misurato al checkpoint 800 di P1-a: primary localize/classify/16ch = 0 con predizioni a prior fisse; gate ridotto
r20 al passo 400 FAIL (TRUE 0.082, SUBSTITUTED 0.076, ZEROED 0.080); backbone insensibile allo shuffle dei soft token.
Cause: colonne lineari dello stem non clippate (A |z| p99 95; D1–D3 `z_band` fino a 33.410 perché la MAD di banda
collassa su segmenti piatti e il floor 1e-6·max|V| non protegge); stem Linear+SiLU senza normalizzazione, quindi la
scala arriva intatta alla fusione (norma descrittori: mediana 61, media 244, max 6.049) e il mixer è scavalcato;
proiettore GELU-terminale con componente comune positiva (coseno fra soft token 0.58 → 0.72–0.86) e norma 17–25
contro 0.66–0.73 degli embedding testuali.

Risposte letterali di Astra (D1–D5):

1. **D1.** Approvato C = 16 su tutte le colonne lineari A/D/raw, globali e locali; `asinh` calcolata prima del clipping.
   Approvato il floor di banda 1e-3·s_rec conservando la guardia numerica esistente; le bande sono già in unità
   normalizzate, quindi il floor è 1e-3 assoluto. Stessa scala nella salienza, ranking sui coefficienti non clippati.
2. **D2.** Approvata LN(512) sull'uscita della fusione prima delle somme. `meta_proj` a norma tipica 10–12 con gain
   fisso ≈ 0,25. I quattro embedding categoriali (norma 22 ciascuno, ≈ 44 in somma) vanno calibrati a somma ≈ 22:
   gain iniziale ≈ 0,5; altrimenti, normalizzando la fusione, gli identificatori diventano dominanti.
3. **D3.** Proiettore variante (i): LN → Linear 512→2560 → GELU → Linear 2560→2560, ultima lineare senza attivazione e
   bias inizializzato a zero. Gain fisso per P1, calibrato dopo tutte le correzioni su token validi del train: RMS
   iniziale ≈ 0,0137, norma ≈ 0,7. Salvato nel checkpoint; deriva monitorata (gain fisso ≠ norma fissa durante
   l'apprendimento). Una seconda lineare casuale non elimina automaticamente la componente comune: permette di
   apprendere una trasformazione che la riduca.
4. **D4.** Approvato P1-b da zero (LoRA, optimizer, scheduler, curriculum inclusi). Completare P1-a (post-validazione e
   r120 previsti). Prima del rilancio: smoke aggiornato, verifica delle scale e degli aggiornamenti prodotti dalla CE
   separatamente dalla loss ausiliaria. r20 al passo 400 è una sentinella: abort se persistono prior fisse e assenza
   di beneficio del contenuto; CI soltanto inconcludente ⇒ r120 sullo stesso checkpoint. P2 resta subordinato al
   gate completo.
5. **D5.** Stem: `LayerNorm(64)` dopo SiLU e prima dei blocchi residui, riapplicando la maschera (F2 interviene troppo
   tardi per lo squilibrio dentro i trunk). Gate: `no_category_collapse` oggi non segnala TRUE = SUBSTITUTED =
   ZEROED = 0; aggiungere un veto esplicito per le categorie principali rimaste a zero; il vantaggio aggregato
   trainato da lead_lag non deve autorizzare P2.

Conseguenze implementative: `LINEAR_CLIP = 16`, `BAND_SCALE_FLOOR = 1e-3`, `META_GAIN = 0.25`,
`EMBEDDING_INIT_STD = 0.5`, `fuse_norm` in `tokenizer.py`; `BranchStem` con LayerNorm in `deep_branches.py`;
`Projector` a due lineari con buffer `out_scale` calibrato dal trainer su righe reali del train (fresh start) e
`soft_rms` registrato per esempio nel log; veto `no_main_category_at_zero` in `perceptual_gate.py`; P1-a archiviato
come risultato negativo, P1-b lanciato con `droplet/gate_early.sh` al passo 400.


## Decisioni Astra sulla sentinella di P1-b e su P1-c (15/09, ~06:00 locali, sessione Codex `codex exec`)

Contesto: P1-b (tutte le correzioni di scala) al checkpoint 400: gate r20 FAIL con TRUE 0,059 / SUBSTITUTED 0,118 / ZEROED 0,082
(Δ −0,071, CI95 [−0,152; −0,011]); validazione rapida primary 0,2174 con prior posizionali; soft token risaliti a norma 2,0
(RMS 0,040, 3× il target) e coseno a coppie 0,92 (init 0,63): il proiettore a gain fisso impara una mappa quasi di rango 1.
Opzioni proposte: O1 far finire; O2 vincoli sull'uscita del proiettore; O3 inversione dei learning rate; O4 segmento di
"lettura" supervisionato lato LM generato dai template del teacher.

Risposte letterali di Astra (D1–D4):

1. **D1. Fermare P1-b ora.** Archiviare checkpoint 400, log e diagnostiche. Prior fissi, nessun recupero delle categorie
   percettive e contenuto dannoso soddisfano il criterio di abort già fissato. Completare O1 aggiungerebbe poco.
2. **D2. P1-c da zero: O4 + O3 + O2 parziale.** RMS per token = 0,013 senza parametri affini apprendibili; esclusa la
   centratura per vista (può cancellare informazione comune utile). Decorrelazione λ = 0,01 applicata alla media dei coseni
   al quadrato, diagonale esclusa, fra token validi entro vista. Frontend/proiettore 2e-5; LoRA 1e-4; stesso warmup e
   scheduler; reset completo inclusa LoRA. Lettura 0,5; finale 1,0; CE mediata separatamente per segmento. O4 è
   l'intervento principale: i vincoli geometrici contengono il collasso ma non insegnano cosa leggere.
3. **D3. Sentinella 400, soglie fissate prima della run.** Proseguire: Δ ≥ +0,02 assoluto, limite inferiore CI95 > 0,
   nessuna categoria principale a zero o collassata, invalidi < 1 % (definizione attuale: TRUE − massimo dei controlli per
   crop, aggregazione e bootstrap per registrazione). Abort: Δ ≤ 0 insieme ai prior fissi, oppure regressione
   significativa rispetto ai controlli. Casi non conclusivi: r120 sul checkpoint 400, proseguire solo se supera i criteri.
   Coseno medio entro vista: obiettivo ≤ 0,80; > 0,90 è allarme (diagnostica, non prova né veto contro un beneficio
   causale verificato).
4. **D4. O4 richiede due correzioni sostanziali.** Il teacher attuale (`trajectories.py:_view_reasoning`) usa estremi
   copiabili dall'header e posizioni esatte dai campioni grezzi: supervisionare soprattutto bucket energetici e confronti
   fra canali alla risoluzione realmente rappresentata, evitando precisione non osservabile. In validazione e nel gate la
   lettura deve essere generata dal modello in ciascuna condizione (una lettura gold nel contesto darebbe una
   scorciatoia al finale); allineare la generazione diretta al nuovo formato (oggi parte con thinking disabilitato).

Conseguenze implementative (P1-c): `Projector` con normalizzazione RMS per token a 0,013 senza parametri (niente gain
calibrato); penalità di decorrelazione in `forward_loss` (`training.decorrelation_weight = 0.01`); nuovo modulo
`reading.py` con la lettura programmatica alla risoluzione dei bin della vista (energie Haar per bucket, ordinamento dei
canali per energia, bin della deviazione massima; niente estremi né indici di campione); sequenze P1 dirette e fixed-crop
con turno `assistant_reasoning_final` (lettura + finale) a thinking ON, pesi lettura 0,5 / finale 1,0; `generate()` diretto
a thinking ON con taglio della lettura a 384 token e riserva finale 768; config `p1_mi300x.toml` con lr 2e-5 / 1e-4.


## Decisioni Astra su P1-c e P1-d (15/09, ~08:00 locali, sessione Codex `codex exec`)

Contesto (RISULTATI-P1C-20260915.md): P1-c fermata alla sentinella 400 (gate r20 FAIL con CI inconcludente, Δ −0,015;
quick val localize/classify 0; letture generate quasi identiche); diagnostica decisiva: CE di lettura e finale identiche
con soft token veri / mescolati / azzerati (0,1702 / 0,1706 / 0,1701; 0,3105 / 0,3109 / 0,3123): il backbone ignora i
token. Proposta: P1-d in due fasi (A con LoRA congelata e lettura compatta; B con LoRA).

Risposte letterali di Astra (D1–D4):

1. **D1. Approvato P1-d da zero, con B subordinata al superamento di A.** A: massimo 500 aggiornamenti dell'optimizer,
   controllo ogni 100; frontend+mixer+proiettore 1e-4, lettura/finale 1,0/0,5, ausiliaria occupancy 0; RMS 0,013,
   decorrelazione 0,01, warmup 3 % e coseno; LoRA congelata con `requires_grad=False` (il backward attraverso il backbone
   deve comunque raggiungere i soft token). Uscita: 0,10 nats/token misurati sui campi informativi dei bin (escludendo
   boilerplate e valori copiabili dal prefisso), su 24 registrazioni val fisse e stratificate: TRUE deve battere ENTRAMBI
   SUBST e ZERO con limite inferiore del CI95 appaiato sopra zero; richiesta anche migliore accuratezza dei bin nella
   lettura GENERATA rispetto ai controlli. A 500 senza PASS: stop e diagnosi. B: frontend 5e-5, LoRA 2e-5, mix P1 completo,
   pesi lettura/finale 0,5/1,0; sentinella 400 passi dopo lo sblocco (Δ ≥ 0,02, CI95 > 0, nessuna categoria principale a
   zero/collassata, invalidi < 1 %); inconcludente senza criteri d'abort ⇒ r120 sullo stesso checkpoint; Δ ≤ 0 con prior
   fissi ⇒ abort.
2. **D2. Approvata la lettura compatta; ranking dei canali eliminato in fase A** (concentra la supervisione sulla
   posizione; il ranking dell'energia Haar normalizzata non è esattamente deducibile dall'header). Mantenere identità del
   canale, top bin e picco; indice per bin con griglia e offset dichiarati nel prompt: `top ch_00:62,61,60; peak ch_00:62`.
   Nel gate non contare le ripetizioni come nuova evidenza. Conta l'informazione imprevedibile dal prefisso, non la
   percentuale di token numerici.
3. **D3. Sì alla diagnostica prima di A, tetto 20 minuti, con intervento causale**: con frontend del checkpoint 400 e soft
   token identici, confrontare LoRA400 attiva/disattivata a parità di prompt, maschere e λ RoPE; misurare CE sui bin e
   sensibilità dei logit; poi attenzione per testa nelle otto layer sulle posizioni che predicono i bin. Nel wrapper
   `q_proj` produce anche il gate sigmoid dell'uscita: misurare gate e contributo del segnale dopo gate/proiezione; la sola
   massa d'attenzione non dimostra il meccanismo (Jain–Wallace). L'attribuzione alla LoRA resta un'ipotesi; la CE finale
   con lettura gold beneficia già della risposta intermedia.
4. **D4. Due vincoli.** SUBST deve sostituire contenuto fra registrazioni compatibili conservando header, geometria e
   maschere: il semplice shuffle dell'ordine non basta. In B conservare i pesi appresi in A e NON riavviare il curriculum
   RoPE. Se A fallisce: probe lineari separati su uscita del mixer e soft token (congelati, train/val separati per
   registrazione): successo prima del proiettore e fallimento dopo implicano il proiettore nella perdita di
   decodificabilità; successo anche dopo concentra la diagnosi sull'interfaccia col backbone; un probe negativo da solo non
   dimostra assenza d'informazione.

Conseguenze implementative: spec `prototype.spec.p1dA.json` / `p1dB.json` e config `configs/p1dA_mi300x.toml` /
`p1dB_mi300x.toml`; `reading_format = "compact"` (`top ch:b,b,b; peak ch:b`, niente ranking) con riga "Reading grid" nel
prompt della vista (bin, larghezza, offset); LoRA congelata quando il suo lr è 0; orizzonte dello scheduler = passi massimi
della fase; `--skip-curriculum` per la fase B (λ = 1 dall'inizio) con `--init-from` dal checkpoint di A; tool
`tools/reading_probe.py` (CE sui campi dei bin TRUE/SUBST/ZERO con bootstrap appaiato + accuratezza dei bin nelle letture
generate) eseguito ogni 100 passi da `droplet/probe_watch.sh`; diagnostica causale `tools/diag_lora_causal.py`.

## Decisioni Astra su P1-e (15/09, ~10:00 locali, sessione Codex)

Contesto: fase A di P1-d FAIL su tutti i criteri (vantaggio dei token veri +0,009 CI95 [−0,014; +0,033] al passo 500;
accuratezza dei bin nelle letture generate 0,021 con token veri contro 0,035 con sostituiti; probe lineari identici
all'inizializzazione). Errata dell'autocast recepita: la LoRA è attiva e vale 0,336 nats con token veri, 0,322 con token
azzerati; nessuna evidenza di lettura del contenuto.

1. **D1. A è FAIL; fase B di P1-d e P2 restano ferme.** P1-e parte dal checkpoint A500, LoRA congelata, frontend 1e-4,
   RMS 0,013: **200 passi di sola O1 senza CE**, poi, se generalizza, **300 passi O1+O2** con CE lettura 1 e finale 0.
   O3 come ablazione successiva. Gate invariato: ≥ 0,10 nats, CI95 positivo contro entrambi i controlli, accuratezza
   generata superiore; conferma su registrazioni nuove. Correzione alla mia diagnosi: un probe negativo non dimostra
   indecodificabilità, e una O1 riuscita non garantisce che il LM sappia usare quella codifica.
2. **D2. Supervisione esatta.** Due teste LINEARI sui SOFT dopo la normalizzazione RMS, ingresso `soft / 0,013`, senza
   metadati aggiuntivi; niente MLP, niente supervisione su Z all'inizio. La seconda testa ha due uscite.
   - Posizione: centro del bin sulla griglia dichiarata, in coordinate assolute, diviso la lunghezza della registrazione.
   - Energia: `log1p(E[c,b])` con esattamente il punteggio Haar del teacher aggregato sulle bande; il teacher sceglie i
     top-3 GLOBALI sulla vista, quindi nessuna ulteriore normalizzazione per canale.
   - Picco: `log1p` del massimo scarto normalizzato dalla mediana entro il bin, come nel teacher (l'energia Haar da sola
     non supervisiona questo campo).
   MSE con bersagli standardizzati da statistiche FISSE del train; pesi posizione/energia/picco 1/1/0,5. Media prima fra
   token dello stesso bin, poi bin, canali, viste e registrazioni. Occupancy a 0, sostituita per P1-e; teste escluse
   dall'inferenza. Soglie operative di O1: errore medio di posizione ≤ 1 bin e recupero dei top-3 ≥ 0,70 su registrazioni
   escluse dall'addestramento. Nota: 0,051 è la prevalenza, il caso della balanced accuracy è 0,5. Correggere lo split
   interno del probe lineare, che oggi taglia per token e può spezzare una registrazione.
3. **D3. Sostituzione al 25 % fisso.** Donatore diverso, solo dal train; stessa lunghezza, stessa cadenza temporale,
   canali con corrispondenza esplicita, finestra/griglia e validità compatibili; nessun vincolo di categoria o gold.
   Ricostruire token e bersagli DAL SEGNALE del donatore sotto la maschera del ricevente: niente incollaggio di token
   ordinati con ancore incompatibili. Header del ricevente; lettura e ausiliarie del donatore effettivamente mostrato;
   peso del finale 0 per evitare supervisione contraddittoria. Nel gate SUBSTITUTED resta valutato sul bersaglio
   originale; misurare separatamente il recupero della lettura del donatore.
4. **D4. Decorrelazione λ 0,01 nella principale**; poi confronto con 0,1 dallo stesso checkpoint e con gli stessi batch.
   Esclusi per ora 1,0 e la centratura obbligatoria: il beneficio ZEROED→TRUE non dimostra che tutta la componente comune
   sia inutile, e ridurre il coseno può creare diversità priva di contenuto. Decide il gate.
5. **D5. Controllo geometrico sì, dopo O1**, massimo 200 passi o 30 minuti: stessi otto layer, λ RoPE fisso 0 contro 1,
   checkpoint iniziale, batch, loss e parametri allenabili identici. La fattorizzazione Q*/K* è algebricamente esatta;
   eventuali bug si verificano confrontando logit e gradienti con la formula diretta su input misti. Un vantaggio causale
   solo nel ramo nativo implica l'interfaccia wildcard; due FAIL non la assolvono.

## Decisioni Astra dopo la fase 1 di P1-e (15/09, ~11:50 locali)

Risultati riportati: gate O1 su 60 registrazioni mai addestrate, viste globali a 64 bin — MAE posizione 4,16 bin al passo
100 e 2,37 al passo 200 (caso 16); recupero top-3 0,361 e 0,378 (caso 0,047); Pearson energia 0,519 → 0,672; Pearson picco
0,602 → 0,698. FAIL sulle soglie, ma prima volta che l'informazione è decodificabile e generalizza.

1. **D1. Estensione approvata**: dal checkpoint 200, altri 600 passi (800 totali), conservando pesi, teste, statistiche e
   momenti di Adam; nuovo scheduler sui 600 aggiuntivi con warmup 18 passi, picco 1e-4 e coseno; campionamento che
   prosegue dal punto raggiunto; gate ogni 100. L'attribuzione del plateau allo scheduler resta un'ipotesi: i numeri
   giustificano l'estensione, non garantiscono ulteriori dimezzamenti.
2. **D2. Termine di ordinamento affiancato all'MSE dell'energia**, pesi esistenti invariati: perdita a coppie tipo RankNet,
   `L_rank = media log(1 + exp(s_j − s_i))`, **peso 0,2**, dove `s` è l'energia standardizzata predetta mediata fra i token
   dello stesso (canale, bin), `i` percorre i tre bin top del teacher e `j` i 16 esclusi con punteggio predetto più alto.
   Competizione sull'intera vista, tutti i canali insieme; escluse le parità energetiche; media sulle coppie, poi viste,
   poi registrazioni. Concentra la supervisione sui concorrenti che possono scalzare i top-3.
3. **D3. Soglie confermate**: MAE ≤ 1 bin e recupero top-3 ≥ 0,70 sulle globali a 64 bin per dichiarare O1 PASS. Nel codice
   il recupero top-3 aggrega usando gli indici REALI dei bin, quindi può migliorare anche con posizione decodificata male:
   per questo servono entrambi i criteri. Crop rendicontati separatamente; l'MSE del mix non determina la MAE globale.
   Il checkpoint selezionato va confermato su altre registrazioni escluse dal tuning.
4. **D4. Fase 2 in parallelo dal checkpoint 200**, come esperimento diagnostico, massimo 300 passi: LoRA congelata, O1
   mantenuta, CE lettura/finale 1/0, sostituzione 25 %, controlli ogni 100 misurando anche O1. Gate del LM invariato:
   vantaggio ≥ 0,10 nats per token sui campi informativi contro entrambi i controlli, CI95 appaiato positivo e migliore
   accuratezza dei bin generati. Un PASS dimostra l'utilità di una codifica parziale; un FAIL impone di riprovare con una
   O1 migliore prima di condannare l'interfaccia. P2 resta subordinata al gate percettivo completo.

## Decisioni Astra dopo l'interruzione di O8 (15/09, ~19:00 locali)

O8 (run lunga `p1-g`/`p1-g2`) interrotta al passo ~1520 di 3000 per il criterio pre-registrato: guadagno del vantaggio
fra i passi 1000 e 1400 pari a +0,0013 (soglia +0,01) e limite superiore dell'intervallo +0,0178 (soglia +0,05).
Vantaggio misurato su 60 registrazioni: −0,0003 (p300), +0,0052 (p500), +0,0079 (p1000), +0,0092 (p1400, primo intervallo
interamente positivo contro entrambi i controlli). Metriche O1 nella stessa run: posizione 5,85 → 3,08 → 1,174 bin,
recupero top-3 0,439, Pearson energia 0,767, picco 0,852.

1. **D1. Controllo oracle in versione B, con controlli causali**, sull'ultimo checkpoint con finale supervisionato
   precedente a O8, congelato. Stesse 60 registrazioni e stesse viste; il testo oracle contiene esclusivamente i campi dei
   bin, mai la risposta finale. Quattro condizioni: `oracle` (soft token azzerati più bin corretti come testo),
   `permuted` (azzerati più bin di un'altra registrazione, permutazione fissa senza auto-abbinamenti), `empty`
   (azzerati, nessun testo), `real` (token veri, nessun testo). Si misura la CE sui campi informativi della RISPOSTA
   FINALE, media per registrazione e bootstrap appaiato, più `primary` e formati invalidi con decoding identico.
   Oracle positivo se il CI95 di `CE_controllo − CE_oracle` è interamente positivo contro SIA `permuted` SIA `empty`, e
   il `primary` conferma un beneficio sul compito.
2. **D2. Confronto adattatore attuale contro Q-Former approvato in modo condizionato** a un oracle positivo e all'assenza
   di un beneficio utile dei token veri sullo stesso checkpoint: unica differenza architetturale, stessa inizializzazione
   del LM, stessi dati, stesso numero di soft token, stessa loss e stessi passi, **finale supervisionato in entrambi i
   bracci**. Giustifica provare una nuova interfaccia, non dimostra che il resampler risolva il problema. Un oracle
   negativo impone prima di verificare il formato del prompt.
3. **D3. Verdetto da scrivere subito**: «P1: metodo non dimostrato con questa ricetta, questi dati e questo budget».
   Lo stop segue il criterio pre-registrato; l'oracle servirà a precisare la diagnosi. O1 è il miglior risultato del
   progetto, ma 1,174 bin resta sopra la soglia di 1: progresso forte, soglia formale non raggiunta.
4. **D4. Massimali sui 145 $ residui**: 15 $ per oracle e verifiche, 110 $ per il confronto adattatore/Q-Former
   (indicativamente 55 $ per braccio), 20 $ di riserva per la valutazione conclusiva e gli imprevisti. La spesa dei 110 $
   si attiva soltanto dopo il gate dei punti 1 e 2.

## Decisioni Astra sul controllo oracle (messaggio 11, 15/09)

Domanda: nel probe implementato la cross-entropy sui campi finali era misurata su una sequenza che conteneva la lettura
programmatica **vera** teacher-forced nel blocco `<think>` in tutte e quattro le condizioni, rendendo l'informazione dei
bin disponibile ovunque e annullando il contrasto.

1. **Opzione C.** Il `<think>` è la lettura **generata autonomamente dal modello in quella condizione**, seguita dal
   finale gold teacher-forced. La cross-entropy misura soltanto i campi informativi del finale. Vanno riutilizzati i
   token integrali della lettura generata, inclusa la chiusura naturale o forzata; nessuna lettura teacher va aggiunta
   nell'assistente. L'opzione B è esclusa; l'opzione A non serve come misura principale.
2. `real`: lettura generata con soft token veri. `empty`: lettura generata con soft token azzerati. Anche `oracle` e
   `permuted` generano la propria lettura, condizionata sul rispettivo `Reading:` nel turno utente.
3. `thinking=True` confermato, con decodifica e budget identici in tutte le condizioni: 384 token di ragionamento e 768
   finali. Nessun finale gold durante la generazione; gli invalidi restano nel calcolo del primary.
4. Checkpoint congelato: **`p1-dA/step_000000000500`**, con configurazione coerente e il λ del checkpoint. È l'ultimo
   previsto da D1 con finale supervisionato. Nessuna selezione successiva fra `dA` e `c` in base all'esito dell'oracle.
5. Verdetto: entrambi i CI95 appaiati interamente positivi **e** primary dell'oracle strettamente superiore a entrambi i
   controlli, sulle stesse 60 registrazioni. Nessuna soglia aggiuntiva in nats per questo controllo diagnostico, ma
   l'ampiezza degli effetti va riportata; la soglia 0,10 della cross-entropy di lettura non si trasferisce al finale.
   Un intervallo che attraversa lo zero dà **INCONCLUDENTE**, non automaticamente negativo.

## Decisioni Astra del 16/09 (chat desktop, riportate da Jacopo): la ricetta va1 sull'architettura nuova

Domanda di Jacopo: invece di riaddestrare va1, allenare il modello nuovo con lo stesso addestramento di va1.
Risposta: «Sì. È il prossimo esperimento più sensato, e non serve riaddestrare va1 prima.»

1. **Architettura nuova invariata**: encoder profondi, rami A/D, proiettore e RoPE.
2. **Supervisione come va1**: domanda → JSON finale, `final = 1`; niente lettura dei bin, niente loss ausiliarie sui
   bin, niente ragionamento intermedio. La discretizzazione interna dell'encoder resta.
3. **Stessi dati e protocollo di va1**: cinque epoche, mantenendo comparabili batch effettivo, learning rate e
   selezione del checkpoint.
4. **Partenza pulita**: Qwen preaddestrato e nuovi moduli inizializzati, senza ereditare gli effetti delle run di
   lettura precedenti.
5. **Due esiti**. Se supera i controlli vero-sostituito abbiamo la base percettiva del modello nuovo, e si aggiungono
   dopo ragionamento e zoom. Se fallisce, si usa il checkpoint va1 già disponibile come riferimento e si introducono
   le differenze architetturali una alla volta.
6. **Si misurano sia il primary sia la dipendenza dal contenuto**: «un buon punteggio da solo non basta, dopo quanto
   visto con gli shortcut».

Astra formula così la domanda a cui finora mancava la risposta: la nuova architettura funziona sul compito originale,
senza obbligarla prima a imparare la lettura dei bin?

## Decisioni Astra del 16/09 sul gate di va1r (thread `01a09a09`, messaggio 13)

1. **Gate confermato sulle tre categorie** che l'header non può risolvere, ma su un **pannello fisso adeguato**,
   idealmente i 208 positivi di Z1-bis. Soglia Δ ≥ +0,05 con CI95 appaiato sopra lo zero **contro entrambi i
   controlli**, sostituito e azzerato: il codice ne verificava solo uno ed è stato corretto. Il primary complessivo
   resta descrittivo; i criteri dell'oracle restano invariati. Le quattro righe utili dello smoke non autorizzano a
   dire «dipendenza esattamente zero» in generale.
2. **Header: opzione C.** Invariato durante `va1r`, con un braccio successivo senza minimo e massimo e con mediana e
   scala conservate. E il confronto fra `va1` e `va1r` va fatto **sulle stesse righe di validazione**: val104 contro
   test800 non dimostra nulla.
3. **Si completano le cinque epoche**, salvo problemi tecnici: le ultime due costano circa 16,80 $ e vale la pena
   completare proprio il protocollo che si vuole verificare. Niente arresto automatico per due categorie a zero
   all'epoca 3, che ignorerebbe l'eventuale progresso percettivo multicanale. Gate adeguato all'epoca 3 e sul
   checkpoint finale selezionato.
4. **Primary comparabile senza vantaggio del contenuto significa prestazione comparabile e percezione utile non
   dimostrata.** Per attribuire una scorciatoia comune serve lo stesso gate anche su `va1`; poi serve il controllo su
   ritagli identici e fuori centro, con un percorso percettivo allenato — gli adattatori locali di `va1r` non vengono
   allenati dalla sola modalità diretta.
5. **Correzione sul conteggio dei token**, verificata: non 4.096 segmenti uniformi ma **256 token per canale**
   (64 A, 64 D di copertura, 64 D salienti, 64 grezzi), multirisoluzione e con selezione saliente. L'ipotesi della
   risoluzione insufficiente resta plausibile ma quel conto non la dimostrava.

## Decisioni Astra del 16/09 sul bersaglio percettivo (thread `01a09a09`, messaggio 14)

Dopo il confronto appaiato con `va1` e la scomposizione presenza/assenza (`RISULTATI-VA1R-20260916.md`).

1. **Replica operativa e successo percettivo vanno separati.** Il primary complessivo resta il confronto con la
   ricetta `va1`, ma sovrapporre gli intervalli non dimostra equivalenza. Per dichiarare progresso percettivo serve
   **miglioramento appaiato sui positivi delle tre categorie**, con aggregazione fissata prima e CI95 sopra lo zero,
   **più** il superamento del gate vero-contro-controlli. Il tasso di falsi positivi sui negativi va riportato
   comunque: riconoscere l'assenza non è automaticamente una scorciatoia.
2. **Il gate globale resta**, all'epoca 3 e sul checkpoint selezionato, con le tre condizioni necessarie — vero,
   sostituito, azzerato — sui 208 positivi. La debolezza di `va1` non lo rende inutile: è la verifica del modello che
   stiamo effettivamente allenando. Il controllo a ritaglio viene dopo un adattamento al crop, risponde a una domanda
   diversa e non sostituisce questo esperimento.
3. **Ordine dei sospetti**, se la localizzazione sui positivi resta a zero: apprendimento dell'associazione
   segnale-coordinate, poi il formato numerico assoluto, poi la RoPE. La supervisione del collegamento **esiste già**
   nella loss finale e non va chiamata assente, semmai troppo indiretta. Prima di tutto però va verificato che
   contenuto e coordinate siano recuperabili dagli **embedding finali**: il 93 % dimostra sovrapposizione dei
   supporti, non conservazione dell'informazione attraverso encoder e proiettore. E lo 0,213 non è il tetto di un
   lettore che combina più token e raffina i confini.
4. **Braccio header ridotto**: preparazione ora, lancio dopo `va1r`, stessa inizializzazione, stessi dati, ordine,
   budget e criterio di selezione, unica modifica la rimozione di minimo e massimo. Il suo successo richiede
   **migliore dipendenza dal contenuto e migliore prestazione sui positivi**, non una differenza nel primary
   complessivo: un calo della magnitudo da solo dimostrerebbe il costo della rimozione, non un fallimento percettivo.
5. **Due correzioni fattuali, verificate**: il primary di `localize_all_channels` è **F1 sui canali**, non IoU; e
   l'affermazione «nessuno dei due localizza mai un'anomalia multicanale» vale solo sulle 104 righe — sulle 520
   `va1` supera lo zero su 4 delle 48 righe positive, con i canali giusti.

## Decisioni Astra del 16/09 su zoom, RL e budget (thread `01a09a09`, messaggio 15)

Contesto: Jacopo dispone di **altre GPU**, quindi il tetto dei 200 $ del droplet non è più il vincolo stringente. Sua
proposta: usare il tool di zoom nella parte di RL, con penalità per turno piccola, più zoom su canali diversi.

1. **SFT prima, RL dopo.** Le GPU aggiuntive servono a continuare `va1r` **e** a preparare tracce e pilot zoom-SFT in
   parallelo. L'RL si avvia quando lettura locale e ricerca autonoma funzionano: più GPU non eliminano questa
   dipendenza. Un pilot dall'epoca 2 sarebbe esplorativo; il confronto definitivo parte dal checkpoint `va1r`
   selezionato.
2. **Tracce multiscala grossolano → fine.** Regola iniziale dichiarata: `w0 = clip(L/16, 1024, 8192)`, poi
   raffinamenti a `w/4`, minimo 256, con possibilità di cambiare canale. È un'ipotesi da verificare, non una garanzia
   di contenimento. Vanno conservati eventi fuori centro, finestre negative e recuperi dagli errori. Il contenimento
   si riporta come descrittivo: W3 ha già mostrato che può **diminuire mentre la prestazione migliora**.
3. **Proposte dai token salienti sì, ma non come unica sorgente.** Vanno confrontate con il selettore Haar di W1 e
   affiancate da esplorazione di altre finestre e canali e da recuperi. Le proposte possono essere senza gold; usare
   le annotazioni del **solo train** per verificarle e supervisionarne gli esiti è legittimo. **Precisazione: il 93 %
   è copertura fra candidati, non accuratezza della finestra effettivamente scelta.**
4. **Pilot RL, valori dichiarati prima**: massimo 6 chiamate, 8 viste canale complessive, penalità **0,005 per
   chiamata** su reward finale in scala 0–1, stesso budget massimo per il riferimento SFT. Prima di lanciarlo va
   verificato che i gruppi G=8 abbiano variabilità nel **punteggio finale**, non solo nel costo: penalizzare i turni
   non rende informativo un reward piatto. Successo: **+0,03 nella macro-media dei primary sui positivi delle tre
   categorie**, CI95 appaiato sopra lo zero contro l'SFT, gate percettivo mantenuto e falsi positivi non peggiorati
   oltre due punti percentuali. Reward penalizzato e metrica ufficiale si riportano separatamente.
5. **Punteggio ufficiale invariato.** Si riportano accanto presenza con **sensibilità e specificità**, IoU sui
   positivi di localize e F1 sui positivi del multicanale. Una lista vuota corretta è una risposta valida; una
   risposta testuale vuota o invalida resta zero.
6. **`va1r` resta a cinque epoche.** L'epoca 2 informa, non decide. Un'estensione sarà un esperimento separato,
   motivato da miglioramenti ancora in corso sui positivi e nella dipendenza dal contenuto. E il confronto dell'epoca
   1 resta **val104 contro val520**: **non si dichiara che `va1r` sia avanti.**
