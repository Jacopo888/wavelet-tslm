# Brief per Codex — punto 4, fix2: Z0 "blocco silenzioso" dopo la diagnosi del collasso a EOS

## Cosa è successo sul nodo (kit `00e873e`, run `/root/l3/zoom_20260911b`, ora fermata)

Z0 probe80: `no_zoom` 0,346; `oracle` **0,0107**; `heuristic` **0,0107**, con predizioni del
secondo turno **vuote** (primo token generato = `<|im_end|>`, 80/80). Diagnostico
`l3/tools/diag_zoom_empty.py` (val[0] e val[1], greedy):

| caso | prompt | esito |
|---|---|---|
| A | prompt va1 originale | risposta corretta come sempre |
| B | turno 2 del kit (blocco zoom con feature separate + istruzione azione + riga `Assistant action:` + `Now output…`) | EOS immediato |
| C | come B ma via `model.generate` con la finestra zero-padded di `prepare_batch` | EOS immediato |
| D | prompt originale + istruzione azione + riga `Assistant action:` + `Now output…`, **nessun blocco** | EOS immediato |
| E | prompt originale + **solo** la frase `Optional action: …` (turno 1) | EOS immediato |
| F | come B senza attention mask | EOS immediato |

Conclusione: il collasso è causato dal **testo aggiunto dopo la domanda**, non dalle feature.
va1 (LoRA SFT) ha imparato a rispondere solo dopo il testo esatto della domanda
(`… (Here L = 16384.)`); qualunque riga in più → EOS. Il percorso delle feature separate
non è smentito né confermato da Z0 così com'è. Z1 (SFT sul nuovo formato) resta necessario e
insegnerà sia l'azione al turno 1 sia la risposta al turno 2.

## Modifica richiesta (solo `l3/zoom.py`, `l3/node_l3_zoom.sh`, test, PROTOCOL)

1. **Modalità Z0 "silent"** per `oracle` e `heuristic` (eval-only su va1): la finestra entra
   come **blocco in più con intestazione nello stile dei canali originali** e il prompt resta
   **identico a va1** — nessuna frase sull'azione, nessuna riga `Assistant action:`, nessun
   `Now output…`. Intestazione proposta (Codex decide il testo esatto, deve iniziare come le
   altre): `The following is channel <name>_zoom_<s>_<e> (samples [s, e) of L of channel <name>):`.
   Feature codificate separatamente come ora (finestra a piena risoluzione). La risposta
   resta in coordinate assolute della serie originale (per `localize` c'è un solo canale
   originale; per `localize_all_channels` il canale zoomato porta il nome originale nel suo
   nome). Il gate/flag di Z0 usa queste due modalità silenziose contro `no_zoom`.
2. Tenere le modalità attuali (con testo) disponibili come `oracle_text`/`heuristic_text`
   ma **fuori** da Z0 (non spendere GPU: sappiamo che danno EOS).
3. **Z1/Z2 invariati** nel formato a due turni (istruzione + azione + blocco + `Assistant
   action:`), perché lì il formato viene appreso. Aggiungere però nel gate Z1 un contatore
   `empty_final` (risposte vuote) con soglia: ≤ 2 % sulla val520, altrimenti FAIL con motivo.
4. `l3/tools/diag_zoom_empty.py` resta come strumento (non toccarlo).
5. Test CPU: costruzione del blocco silenzioso (prompt testuale identico a va1 salvo il blocco;
   conteggio placeholder = canali + 1), e che le modalità con testo non siano usate da Z0.
6. PROTOCOL §punto 4: registrare la diagnosi (tabella sopra) e la nuova definizione di Z0.

Nessun altro cambiamento; `kaggle/patch/` intatto.
