# RISULTATI gate del backend wildcard — MI300X / ROCm (14/09/2026)

Gate P0 richiesto dalla decisione Astra del 14/09 (§2 di [DECISIONI-ASTRA-20260914.md](DECISIONI-ASTRA-20260914.md)):
verificare, **senza pesi Qwen e senza dati**, che le forme della fattorizzazione Q*/K* (Q/K a 352 a regime e
384 in curriculum, V a 256, 16 teste su 4 KV espanse esplicitamente, bf16, scala esplicita 1/16) girino in
forward e backward sul backend dichiarato con ogni altro backend disabilitato, coincidano con un riferimento
MATH fp32 su tensori piccoli (output e dQ/dK/dV, causale quadrata e query non quadrate con offset di cache),
e misurarne memoria e tempo a lunghezze rappresentative. Strumento: [tools/backend_gate.py](tools/backend_gate.py).
Report JSON e log: `integrity/artifacts/rope_p0_gate_20260914/`.

## Piattaforma

| voce | valore |
|---|---|
| nodo | DigitalOcean GPU droplet, 1× AMD Instinct MI300X VF (gfx942), 191,7 GiB VRAM, 20 vCPU, 235 GB RAM |
| stack | container `rocm:latest` fornito dall'immagine DO "PyTorch 2.6.0 / ROCm 6.4": torch `2.6.0+git684f6f2`, HIP 6.4.43483, ROCm 6.4.1, Python 3.12.10, triton 3.2.0 |
| host | Ubuntu, kernel 6.8, ROCm 6.4.1; boot NVMe 720 GB, scratch NVMe 5 TB montato su `/mnt/scratch` |
| costo | 1,99 $/h; credito AMD 200 $, scadenza 4 ottobre 2026 |

## Esito per backend

| configurazione | esito | motivo / kernel osservato |
|---|---|---|
| `EFFICIENT_ATTENTION`, Q/K 352 e 384, V 256 | **RIFIUTATO** | `sdp_utils.cpp:132` (build HIP): «Efficient attention on ROCM requires q,k,v to have the same last dimension and to be less than or equal to 256». Identico con `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`. `FLASH` e `CUDNN` non disponibili. |
| `EFFICIENT_ATTENTION`, Q/K 256, V 256 (controllo) | PASS | kernel `_efficient_attention_backward` (AOTriton); serve come riferimento di precisione e costo del kernel nativo |
| `MATH` eager, 352 e 384 | PASS numerico (è il riferimento) | `aten::bmm` + softmax materializzata: 65 GiB per una sola chiamata a L=16k |
| `FLEX` (FlexAttention/Triton), 352 e 384 non paddate | **RIFIUTATO** | `ValueError: NYI: Currently non power of 2 embedding dimension are not supported. Got E=352 and Ev=256` |
| **`FLEX` con Q/K zero-paddate a 512, V 256** | **PASS su tutti i casi richiesti** | kernel Triton generati (`triton_tem_fused_*`), nessun fallback MATH nel profiler |

Il padding è esatto: le coordinate aggiunte valgono zero sia in Q sia in K, quindi ogni logit è identico a
quello della fattorizzazione a 352/384; i gradienti sulle coordinate aggiunte vengono scartati. È una
realizzazione della **stessa funzione matematica**, non una wildcard approssimata.

## Numeri (una chiamata di attenzione, B=1, 16 teste, forward+backward, bf16)

Errore massimo assoluto contro il riferimento MATH fp32 a L=512 (causale quadrata; fra parentesi il caso
con offset di cache, 128 query su 512 chiavi):

| configurazione | output | dQ | dK | dV |
|---|---:|---:|---:|---:|
| controllo EFFICIENT 256 | 1,26e-2 (8,3e-4) | 7,0e-3 (5,5e-4) | 1,2e-2 (5,9e-4) | 3,5e-2 (1,1e-3) |
| FLEX pad512, Q/K 352 | 1,15e-2 (8,5e-4) | 8,9e-3 (5,5e-4) | 1,1e-2 (6,4e-4) | 3,5e-2 (1,0e-3) |
| FLEX pad512, Q/K 384 | 1,31e-2 (8,7e-4) | 8,7e-3 (5,0e-4) | 1,3e-2 (6,3e-4) | 3,6e-2 (1,2e-3) |

FlexAttention paddata ha la stessa precisione del kernel nativo a 256: l'errore è quello del bf16, non del
kernel. Tutte le tolleranze dichiarate nello script (output 2,5e-2, gradienti 5e-2) sono rispettate.

Memoria di picco e tempo di una chiamata (forward+backward):

| configurazione | L=4096 | L=16384 |
|---|---:|---:|
| controllo EFFICIENT 256 | 0,27 GiB, 5 ms | 1,06 GiB, 58 ms |
| MATH eager 352 | 4,36 GiB, 17 ms | 65,4 GiB, 250 ms |
| MATH eager 384 | 4,38 GiB, 17 ms | 65,5 GiB, 249 ms |
| FLEX pad512 (352) | 0,44 GiB, 17 ms | 3,50 GiB, 230 ms |
| FLEX pad512 (384) | 0,44 GiB, 17 ms | 3,52 GiB, 231 ms |

Compilazione FlexAttention: circa 5 s alla prima forma, poi riutilizzata. A 16k il kernel Triton è
quattro volte più lento del kernel nativo a 256 e alla pari con l'eager, ma usa un ventesimo della memoria:
sugli otto layer di full attention di Qwen il costo di attenzione a contesto massimo è dell'ordine di 2 s
per passo (forward+backward), contro 520 GiB di attenzione materializzata che l'eager richiederebbe senza
checkpointing per layer.

## Cosa questo gate NON dimostra

- L'esattezza algebrica della fattorizzazione (lift Q*/K* contro la regola a coppie) non è misurata qui:
  è stata verificata da Astra su CPU il 13/09 e va ripetuta su GPU nel P0 numerico con i moduli veri.
- Nessun test con `enable_gqa` nativo: l'espansione 4→16 è esplicita, come prescritto.
- Nessun test di `torch.compile` in presenza di gradient checkpointing, LoRA e forme variabili (lunghezze
  e numero di viste diversi da esempio a esempio in modalità agentica): con FlexAttention ogni forma nuova
  ricompila, e questo va gestito (forme a bucket o `dynamic=True`) prima del training.
- Nessun test dello stack ROCm sui layer linear-attention di Qwen3.5 (kernel Triton di
  flash-linear-attention): è il prossimo controllo dello smoke con modello reale.

## Decisione richiesta ad Astra

Sulla MI300X con torch 2.6/ROCm 6.4 il backend principale della decisione del 14/09 (SDPA
memory-efficient) **non è disponibile per le forme richieste e non lo diventerà con flag**: il vincolo è nel
gate C++ della build HIP. Il candidato che realizza la stessa funzione è **FlexAttention con Q/K
zero-paddate a 512** (esatto, memoria lineare, precisione del kernel nativo, misurato sopra). Si chiede:

1. approvare FlexAttention+pad512 come backend del percorso MI300X, con il vincolo che il gate numerico
   del P0 (lift contro regola a coppie, su GPU) resti obbligatorio prima del training;
2. fissare la politica di compilazione per forme variabili (bucket di lunghezza contro `dynamic=True`) e
   il numero massimo di ricompilazioni ammesse per run;
3. confermare che il percorso CUDA/A40 (Polimi) resti su SDPA memory-efficient come da decisione, e che
   i due backend siano confrontati sullo stesso riferimento numerico prima di dichiararli intercambiabili.

Nessun training è stato lanciato. Nessun peso e nessun dato sono stati usati dal gate.

## Aggiunta 15/09 — gate sul backbone reale (tools/wildcard_gate.py, tools/backbone_profile.py)

Backbone Qwen3.5-4B al pin, caricato offline, torre visiva rimossa, 8 layer di full attention wrappati per
istanza, 64 tensori LoRA (q/k/v/o dei soli 8 layer) risolti per nome. Report in
`integrity/artifacts/rope_p0_gate_20260914/` (`gate_flex_dynamic.json`, `backbone_profile.json`).

| controllo | esito | numeri |
|---|---|---|
| sequenza mista (300 soft token con tempi dentro il testo), FLEX pad512 contro MATH | PASS | argmax 98,9 %, diff media 0,032 |
| generazione con cache contro ricalcolo completo (12 token greedy) | PASS | token identici, diff logit max 0,25 |
| testo puro, wrapped contro nativo SDPA (144 token, prompt ripetuto) | sotto soglia 98 % | argmax 96,5 %, diff media 0,062 su scala 2,17 |
| **controllo di rumore** su 512 token: nativo SDPA contro nativo eager | riferimento | diff media 0,104, max 10,5, argmax 100 % |
| testo puro, wrapped MATH contro nativo eager | entro il rumore | diff media 0,103, argmax 100 % |
| testo puro, wrapped FLEX contro nativo eager | entro il rumore | diff media 0,099, argmax 100 % |

Lettura: su testo puro la wildcard coincide algebricamente con il RoPE nativo (test CPU `tests/test_wildcard.py`);
la deviazione residua sul modello reale ha la stessa grandezza di quella fra due backend nativi in bf16, quindi è
rumore dei kernel accumulato su 32 layer, non un errore della regola. Il 96,5 % del primo prompt è un caso di
posizioni a bassa confidenza (frase ripetuta sei volte); sul prompt da 512 token l'accordo è totale.

Costo osservato prima dei kernel lineari: forward+backward su 3.996 token in 17,1 s con picco 64 GiB (senza
checkpointing); i 24 layer GatedDeltaNet giravano in torch puro perché `flash-linear-attention` mancava.
Installato `fla-core` 0.5.2 (kernel Triton, verificato sulla MI300X con triton 3.2). Profilo con i kernel veri
(`backbone_profile.json`, 4.096 token di cui 1.024 soft token, forward+backward con gradienti LoRA, senza checkpointing):

| percorso | tempo per passo | picco memoria | prima chiamata (compilazione) |
|---|---:|---:|---:|
| nativo (non wrappato, SDPA flash AOTriton) | 0,34 s | 28,0 GiB | 60,7 s (kernel fla) |
| wrappato, FlexAttention `dynamic=True` | 0,45 s | 28,9 GiB | 1,5 s |
| wrappato, FlexAttention forme statiche | 0,44 s | 28,9 GiB | 7,0 s |

La wildcard costa il 30 % in più del nativo sul backbone intero; le forme dinamiche non costano nulla rispetto
a quelle statiche e non ricompilano: è la politica adottata. Il 17 s misurato prima era interamente dovuto al
fallback torch di GatedDeltaNet.

### Completamento richiesto da Astra (15/09 §2): prompt originario da 144 token

| confronto | diff media | p99 | max | argmax |
|---|---:|---:|---:|---:|
| nativo SDPA contro nativo eager (rumore di riferimento) | 0.099 | 1.27 | 16.8 | 95.1 % |
| wrapped FLEX pad512 contro nativo eager | 0.094 | 1.19 | 13.2 | 94.4 % |
| wrapped FLEX pad512 contro nativo SDPA | 0.071 | 0.89 | 10.2 | 97.9 % |

Su questo prompt (frase ripetuta sei volte, molte posizioni a bassa confidenza) persino i due backend nativi
concordano solo al 95.1 % sull'argmax: il 96,5 % osservato il 14/09 era rumore del prompt, non della
regola. Criterio (Astra §2): deviazione wrapped–nativo entro 1,25× il rumore nativo SDPA/eager sullo stesso input
(media e p99) e argmax entro 2 punti → **PASS**. Report: `integrity/artifacts/rope_p0_gate_20260914/text_equivalence_144.json`.
