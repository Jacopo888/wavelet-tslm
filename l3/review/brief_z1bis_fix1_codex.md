# Brief per Codex — Z1-bis fix1: il planner delle tracce non regge il caricamento del train

Stato: 12/09/2026, dopo la revisione del diff di Z1-bis (kit `3a345a1`) e un tentativo di
esecuzione della sola fase `traces` sul nodo.

## Cosa è accaduto (misurato)

`l3/zoom_traces.py::_load_train_samples` fa `[compact_trace_sample(dataset[index]) for index in
range(len(dataset))]` sulle 4.488 righe del train. `compact_trace_sample` è corretto nell'intento
(tiene solo metadati e usa `range(length)` al posto dei valori), ma **`dataset[index]` materializza
comunque tutto**: `AnomalyXLDataset.__getitem__` decodifica `channel_values` (fino a 16 × 131.072
float per riga) e costruisce il prompt completo. Il train parquet è 2,05 GB.

Esito sul nodo: la fase CPU è rimasta appesa oltre 10 minuti e **il nodo ha smesso di accettare
connessioni SSH** (ping ok, sshd non risponde) — sintomo di thrashing di memoria/IO. Nessun job
GPU era in corso, quindi non si è perso lavoro; il nodo è in recupero e ho un waiter attivo.

## Correzione richiesta (solo `l3/zoom_traces.py` + test)

Il planner ha bisogno **soltanto** di: `id`, `category`, `length`, `channel_names`, `answer`.
Nessun valore di serie. Va quindi letto il parquet per colonne, senza passare da `__getitem__`:

1. Caricare con `pyarrow.parquet.read_table(path, columns=["id", "category", "length",
   "n_channels", "channel_names", "answer"])` (o `Dataset.from_parquet(...).select_columns([...])`,
   purché **non** si tocchi `channel_values`) e costruire le righe compatte da lì.
2. Tenere gli stessi controlli attuali: conteggio righe == 4.488, sha256 del parquet == quello
   dichiarato, categorie ammesse, `series_length` coerente con `length`.
3. `time_series` resta `[range(length) for _ in channel_names]`, come oggi: il planner usa solo la
   lunghezza e i nomi dei canali.
4. Se una colonna necessaria non esiste nel parquet, **fallire chiuso** con il nome della colonna,
   senza ripiegare su `__getitem__`.
5. Test CPU: un parquet sintetico minimo (3-4 righe, tutte le categorie) che verifichi che il
   caricamento non legge `channel_values` — ad esempio scrivendo valori sentinella in quella
   colonna e controllando che il planner produca lo stesso piano se quei valori cambiano.
6. Misurare e stampare il tempo e il picco di memoria della fase (`resource.getrusage`), e
   registrarli nel summary del piano: serve a evitare la ricomparsa del problema.

Vincoli invariati: nessuna modifica a `kaggle/patch/`, al contratto delle tracce (`z1bis_v1`),
alle quote, alle soglie dei due gate o al resto del kit; il piano generato deve essere
**identico** a quello che genererebbe il codice attuale a parità di seed (le regole di
campionamento non cambiano).

## Nota operativa per il driver

`node_l3_z1bis.sh` esegue la fase `traces` con `CUDA_VISIBLE_DEVICES=""`: corretto. Aggiungere
solo un limite di guardia esplicito: se la fase supera, ad esempio, 15 minuti o un picco di
memoria dichiarato, fallire chiuso invece di trascinare il nodo.
