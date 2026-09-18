# va1eth (17-18/09/2026): il prompt di ETH verbatim, e dove si rompe davvero

Braccio deciso da Jacopo dopo la scoperta che l'header con le statistiche era **nostro** e non del benchmark
([PROMPT-ANOMALYXL-PAPER-20260917.md](PROMPT-ANOMALYXL-PAPER-20260917.md)). Astra ha finito l'usage a metà serata;
la diagnosi finale è di Fable 5.1.

## Che cosa è `va1eth`

Il prompt di AnomalyXL/OpenTSLM **verbatim, forma compresa**: system prompt del benchmark, un blocco
`The following is channel X:` per canale con i suoi 256 token subito dopo, la domanda del dataset, la risposta
incollata da un solo spazio. **Nessun chat template**: stringa piatta, e l'unico token speciale dell'intera
sequenza è l'EOS. Jacopo ha deciso la forma piatta osservando che il multiturno lo impara P2 e che Qwen il
template ce l'ha già dal post-training: tenere il nostro involucro era la deviazione dal riferimento, non toglierlo.

Due coincidenze che non erano scontate: la quota di token per canale è `A 64 + D_coverage 64 + D_salient 64 +
raw 64` = **256**, esattamente come il riferimento; e 16 canali × 256 = 4.096 è il `max_global_ts_tokens` già
dichiarato.

Verificato prima di lanciare, su registrazioni reali ([tools/eth_layout_check.py](tools/eth_layout_check.py)): i
blocchi **partizionano** i soft token, i token del blocco `slot` sono esattamente quelli con `channel_slot == slot`
confrontati sugli embedding scritti in sequenza, `tau` segue la stessa selezione, il prompt di generazione è
**prefisso byte-esatto** della sequenza di training, e la supervisione copre esattamente `" {risposta}<|im_end|>"`.

## 1. L'informazione è nell'input, e non l'ha messa il training

[tools/token_position_probe.py](tools/token_position_probe.py). Probe lineare **post-hoc** sui soft token — gli
stessi vettori scritti nei segnaposto di `inputs_embeds` — con bersaglio «questo token copre la finestra
annotata», adattato e valutato su registrazioni disgiunte, null mescolato.

| | non addestrato | epoca 1 (4.488) |
|---|---:|---:|
| AUC sul tenuto fuori | 0,9966 | 0,9895 |
| null mescolato | 0,4114 | 0,4279 |
| errore mediano del token migliore | 0,0017 di L | 0,0018 di L |
| entro l'1 % di L | 98,9 % | 98,9 % |

300 righe, 90 tenute fuori, confronto appaiato. **A pesi non addestrati è identico**: la posizione è leggibile da
salienza e `tau` già all'inizializzazione. È un fatto sull'ingresso, non una codifica appresa — correzione di Fable,
accolta.

## 2. La regola non nasce mai

[tools/generalization_ladder.py](tools/generalization_ladder.py), piolo `positions`: un solo fondo, ruotato; le
posizioni di test si interlacciano con quelle di train e il codice **rifiuta di partire** se ne condividono una.

| | train | tenuto fuori | verdetto |
|---|---:|---:|---|
| n = 8, 800 passi | **1,0000** (8/8) | 0,0000 (0/8) | `MEMORISES ONLY` |
| n = 32, 4.000 passi | 0,6250 (20/32) | 0,0000 (0/16) | `MEMORISES ONLY` |

Il primo piolo è valido proprio perché il train arriva a 8 su 8: riproduce la memorizzazione nota e poi chiede una
cosa in più. Il tenuto-fuori è **zero a ogni valutazione**, piatto, sullo stesso fondo e con la stessa larghezza.

Taratura imparata a spese nostre: a 24 esempi e 25 esposizioni il risultato documentato era `NOT MEMORISED` con
1 su 24, e il primo tentativo a n=32 con 40 esposizioni stava in quel regime — diceva solo che non avevamo
addestrato abbastanza.

## 3. Dove arriva, con il controllo che serve

[tools/depth_probe.py](tools/depth_probe.py), teacher forcing, stato nascosto della posizione che **emette la prima
cifra**. Le chiavi JSON sono alfabetiche (`_L, end, present, start`), quindi si misura **`end`** — prima quantità
posizionale, niente da copiare — e si usa **`start`** come controllo, perché ha già `end` scritto davanti.

Il primo giro sondava solo `start` e dava +0,977: era **copiatura**, non lettura del segnale.

Traiettoria di `end` al layer 8, 300 righe, 90 tenute fuori:

| checkpoint | `end` | CI95 | `start` (controllo) |
|---|---:|---|---:|
| non addestrato | −0,080 | [−0,31; +0,16] | +0,935 |
| 2.244 | +0,187 | [−0,06; +0,41] | +0,981 |
| 3.366 | +0,257 | [+0,02; +0,47] | +0,988 |
| 4.488 | +0,342 | [+0,11; +0,54] | +0,976 |

Il canale **non esiste a init** e il training lo costruisce. `start` alto ovunque è controllo positivo: lo zero su
`end` non è un artefatto del metodo. AUC per token: 0,97 al layer 0, 0,71-0,74 dal 16 in poi.

**Limiti dichiarati.** Gli intervalli sono larghi ±0,23 e punti consecutivi non sono separabili: regge solo il
confronto init contro epoca 1. Il null è **un solo rimescolamento** e oscilla fra 0,42 e 0,61 invece di stare a
0,50 — va bene per i divari grandi, è troppo grezzo per i confronti fini. E un probe lineare non sa fare
aritmetica: potrebbe sottostimare un'informazione codificata in modo non lineare (avvertenza di Fable).

## 4. L'aritmetica, e perché lo zero è esatto

Anomalie larghe 100-250 campioni su 16.384: per IoU > 0 il centro va azzeccato entro ~0,9 % di L. I centri sono
grosso modo uniformi, sd 0,29·L. Con residuo √(1−r²)·0,29·L ≤ 0,009·L serve **r ≥ 0,9995**; il massimo osservato è
0,34. Il residuo si accorcia del 3-4 % per blocco da 1.122 passi: **19-27 epoche** contro le 5 di `va1`.

Spiega anche perché `localize` è zero **esatto** e non piccolo: sotto la quasi-esattezza la metrica non dà credito
parziale, quindi non esiste un gradiente di «ci stai andando vicino».

## 5. La diagnosi (Fable, 18/09) — verificata nel nostro codice

L'aritmetica sopra assume un readout lineare che deve stringere `r`. **Ma la via facile non è quella**: scegliere il
token saliente — già AUC 0,99, supporti dei D fini di pochi campioni — e **ricopiarne l'ancora come intero**. Il
problema è che quell'ancora non esiste in forma copiabile.

Verificato riga per riga:

- `tokenizer.py`: `META_DIM = 5`, `META_GAIN = 0.25`, e la posizione entra come
  `anchor_pos / recording_length` — **un solo scalare normalizzato**, in una `Linear(5, fusion_out)`, a un quarto
  di guadagno.
- La fase `tau` esiste nella RoPE, ma la wildcard **la cancella proprio nelle coppie testo-segnale**
  (`test_mixed_pairs_have_no_temporal_phase`): è ciò che la wildcard *è*.
- `va1`, l'unico sistema che abbia localizzato in vista globale (IoU 0,0606 sui positivi, senza header), nel token
  ha le **feature di Fourier della posizione**: `wavelet_encoder.py`, commento *«value scalar + fourier
  positions»*, `in_dim = 1 + 2 * n_pos_freqs` con `n_pos_freqs = 16`, frequenze `2^k · π`. Il DESIGN le ha
  sostituite con la RoPE.

Da una codifica multi-frequenza un LM estrae cifre quasi linearmente — è leggere un positional embedding
sinusoidale. Da uno scalare no: è un convertitore scalare→cifre, aritmetica non lineare che 4.488 esempi non
insegnano. Questo spiega insieme lo zero esatto, `start` a 0,98 dato `end` (ricopiare è facile) e le 19-27 epoche.

### Le tre leve alternative, chiuse da esperimenti nostri

| leva | perché no |
|---|---|
| frequenze RoPE apprendibili (RoPE-ViT) | P1-f: λ=0 ≡ λ=1, la geometria dell'attenzione non è il muro |
| fase di allineamento a LLM congelato (LLaVA) | già fatta: fase A di P1-d, LoRA congelata, FAIL — presuppone che il LM sappia leggere il codice, e il codice non c'è |
| supervisione per token (P1-c/P1-e) | P1-e ha reso il token decodificabile a 1,17 bin e il gate del LM è rimasto piatto: stesso muro |

La quarta lezione dalla letteratura sulle immagini **la rispettiamo già**: M-RoPE di Qwen2-VL degenera in 1D sul
testo, ed è ciò che rende possibile l'adattamento da un LLM preaddestrato. Da noi lo garantisce
`test_pure_text_equals_native_rotary_for_every_lambda`, a 1e-4 per ogni λ.

## 6. Il braccio `va1ethpos`

`prototype.spec.va1ethpos.json` e `configs/va1ethpos_mi300x.toml`: **unica differenza** da `va1eth`, 16 frequenze di
Fourier dell'ancora assoluta nel contenuto di ogni soft token, seno e coseno, **guadagno pieno** (non `META_GAIN`),
calcolate in float64 perché 2¹⁵·π radianti non sopravvivono a bf16. Spento di default
(`position_fourier_freqs = 0`).

Test di disegno, non di presenza: due posizioni distanti 137 campioni — la larghezza mediana di un'anomalia — al
fondo di una serie da 131.072 sono **identiche** nello scalare bf16 e separate di **3,85** nelle feature di Fourier.

**Criterio dichiarato prima della run**: piolo `positions` n=32, 4.000 passi, stesso seme della run
`MEMORISES ONLY`; tenuto-fuori sopra zero con il sostituito a zero ⇒ `GENERALISES`. Se resta 0 su 16, la leva è
sbagliata e lo si sa per due dollari.

### Esito: la leva resta NON TESTATA

Due implementazioni, stesso seme e stesse posizioni del braccio senza Fourier, unica differenza la codifica:

| | baseline | Fourier **sommate** dopo `fuse_norm` | Fourier **concatenate** nell'ingresso |
|---|---:|---:|---:|
| train finale | 20/32 (0,625) | 3/32 | **19/32 (0,594)** |
| tenuto fuori | 0/16 | 0/16 | **0/16** |
| verdetto | `MEMORISES ONLY` | `DOES NOT FIT` | **`MEMORISES ONLY`** |

**Il primo tentativo era mio, ed era sbagliato.** Avevo sommato le 32 feature *dopo* `fuse_norm`, insieme agli
embedding categoriali e a guadagno pieno: 32 dimensioni di ampiezza 1 sopra un vettore normalizzato, che cambiano
segno fra token adiacenti alle frequenze alte. Il train è crollato da 20/32 a 3/32 con tutto il resto identico, e
il verdetto `DOES NOT FIT` significa — per il criterio dello strumento stesso — che quel piolo **non dice nulla**
sulla generalizzazione. `va1` invece costruisce il token *da* posizione e valore insieme
(`in_dim = 1 + 2*n_pos_freqs` → `Linear → GELU → Linear`).

**Corretta l'implementazione, la leva non produce nulla.** Con la concatenazione nell'ingresso (256 → 288, prima
di ogni norma) le due curve si ricongiungono esattamente — 8/32 al passo 3.200 e 10/32 al 3.600 in entrambi i
bracci — e finiscono appaiate a 19 e 20 su 32. Il tenuto-fuori è **zero esempi risolti** in entrambi.

**Il piolo non ha la potenza per la domanda, e questo lo rende inconcludente — non negativo.** Correzione di
Fable, accolta: il baseline arriva a 20/32 solo dopo 4.000 passi, cioè **125 esposizioni per esempio**. Un regime
in cui memorizzare 32 coppie costa 125 passaggi non può far nascere una mappa da codice continuo a cinque cifre su
posizioni mai viste. Lo zero sul tenuto-fuori era atteso in **entrambi** i bracci, quindi non discrimina.

E manca il **controllo positivo del piolo**: non esiste un sistema che lo superi. L'8/8 dell'oracolo testuale
dimostrava il ciclo di addestramento e la copia, non che un modello con un codice funzionante generalizzi da 32
esempi. Il controllo giusto sarebbe `va1` stesso — Fourier più 22.440 aggiornamenti — valutato sugli stessi
tenuti fuori sintetici; non è sul nodo e non si fa con quattro dollari.

**Quindi**: la codifica non cambia l'esito a n=32, e n=32 non basta a nessuno dei due. La leva è **non testata**.
Il test che la decide è sul benchmark, non su un piolo sintetico.

**Un difetto d'implementazione riproducibile, da annotare per chiunque tocchi il tokenizer.** Sommare feature ad
alta frequenza *dopo* `fuse_norm` a guadagno pieno costa 3/32 contro 19/32 sul fit, con tutto il resto identico.
Vanno concatenate all'ingresso della fusione.

**L'ipotesi che resta in piedi, non dimostrata.** Né la selezione percettiva (AUC 0,99 sui soft token, presente già
a pesi non addestrati) né la codifica copiabile della posizione hanno prodotto la regola nei regimi provati. È
compatibile con un collo di bottiglia a valle di entrambe — scrivere cinque cifre esatte da un codice continuo — e
con il fatto che l'unica cosa che nel progetto abbia funzionato sia lo zoom, dove il modello ha un numero già
scritto nel testo a cui somma un offset piccolo. Ma non è stabilito: il piolo non aveva la potenza per stabilirlo.

## Incidenti della notte

**UFW mi ha chiuso fuori dal nodo per quaranta minuti.** `22/tcp LIMIT IN` scarta un IP che apre 6 connessioni in
30 secondi: due watcher a 30 e 45 secondi più i miei retry ci sono finiti dentro, e **ogni tentativo successivo
rinfrescava il blocco**. Il training non se n'è accorto. Tre diagnosi sbagliate prima di quella giusta — RAM (5 %),
thread (1.106 su 965.893), zombie (orfani innocui di `sleep infinity`, il PID 1 del container). La prova stava nel
journal di `sshd`: dal mio IP **nessuna riga**, mentre i bot venivano serviti regolarmente.

**Due hang della GPU in due giorni.** `amdgpu: GPU reset(2) succeeded!` alle 22:29 del 17/09; il reset uccide ogni
processo sulla scheda **senza traceback**. Persi 652 passi. La traccia non era nel log del trainer ma in quello di
un probe vicino, che ha stampato `HW Exception by GPU node-1 — reason: GPU Hang`. Lezione: quando un processo muore
senza traceback, il colpevole può aver lasciato la traccia nel log del vicino.

**Un blocco in validazione** alle 21:32: log fermo da 22 minuti, nessun file toccato, nessuna voce nuova nella cache
di compilazione, GPU al 4 % e CPU al 264 %. Ripartito con `TORCHINDUCTOR_COMPILE_THREADS=1`. Causa non accertata.
