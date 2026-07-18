# Forecasting-2-JP

link accesso dataset nel drive: 
https://drive.google.com/drive/folders/1pDeHKnxl-YRXAQDdhigfw5QzBBjZUEkb?usp=sharing


# note varie 
dataset ETTh1 ha 7 feature numeriche, impone quindi [B, 96, 7] -> [B, 24, 7]

memoria: indicizzamo il dataframe on-the-fly tramite __getitem__. È lo standard industriale rispetto alla pre-allocazione di tutte le finestre in RAM (che esploderebbe la memoria per dataset di grandi dimensioni).

StandardScaler viene fittato esclusivamente sul set di addestramento. Applicare il fit_transform sull'intero file prima dello split invalida il progetto.   

# appunti vari
    [ 0. Titolo e keyword (max 6), come suggerito da pdf ]

dataset preprocessing
	
modificare il formato del dataset
train/test/validation split
indicare finestre temporali (HORIZONS): es. input length = 96, prediction length = 24
optimization (adam, ...)

reti 1D cnn (implementare modelli semplici per testare l’inefficacia dell’ 1D risp a 2D):

MLP
CNN 1D
LSTM

cambio architettura, lista di cose da poter cambiare, passare a CNN per efficienza 


3.1. Implementare modello 2D che si ispira a TimesNet 1D -> 2D (periodicità, fft, top-k, blocchi inception, miglioramento con residual block o dropout…)

DA FARE: Modello 2D con periodo fisso e paragonato a modelli 1D
DA FARE: Modello 2D con periodo trovato da FFT e paragonato a modelli 1D
DA FARE: Modello 2D con top-k periodi e parag. a modelli 1D
DA FARE: Modello 2D con blocchi Inception 

EXTENSION: test modello 2D: FFT vs top-k vs periodo fisso (testare se il periodo influenza la qualità della predizione) 
EXTENSION: testa se periodi espliciti portano ad un miglioramento su 2D e 1D

EXT: testare modelli 1D vs 2D, con periodo fisso, su Horizons diversi (aiuta a capire se 2D è piu utile su orizzonti corti o lunghi)
ne parliamo con Adin e capiamo quali? sarebbero decine di addestramenti e righe in tabelle di risultati
Sono d’accordo, rischiamo di complicarci la vita con le Ext.

3.2. TimesNet Ablation (What is lost):

EXTENSION: Modificare TimesNet rimuovendo dei pezzi e vedere le performance rispetto a TimesNet intero

3.1 e 3.2: prendiamo una sola architettura, togliamo un pezzo alla volta, stendiamo risultati ~ LOO ablation. non si può alla fine escludere tutto quello che non ha peggiorato, gemini: non si i contributi rimossi non sono sommabili. scrivere nel paper considerazione su questo:

LOO Ablation significa prendere l'architettura completa e funzionante, spegnere un singolo modulo (es. l'estrazione FFT), registrare il calo di performance, riattivarlo, e spegnerne un altro (es. il residual block). Non si costruisce mai un modello finale "sommando" i pezzi che non hanno peggiorato le performance, perché si ignorerebbero le dipendenze non lineari tra i layer. Eseguire l'ablazione sull’architettura CNN 2D che craftiamo (piuttosto che sul pesante TimesNet ufficiale) è la mossa corretta.
(non ho capito gemini) per LOO ablation va bene, forse conviene farlo sulla nostra CNN non su TimesNet ok si hai ragione

	3.3. Trovare miglioramenti per TimesNet

EXTENSION: complexity reduction

	3.4  Diversi dataset:
EXT: testare diversi dataset ETT…
EXT: testare dataset finanza? basso SNR pk piu randomico di ETT, fft estrae rumore. usiamo solo ETT?
Io userei ETT come principale. Proverei rapidamente a far vedere che dataset finanza funziona male perchè timesnet è limitato dal fattore periodicità, che in questo caso è basso. Può essere un’osservazione finale. 

risultati a confronto modelli 1D vs 2D: lista di metriche da fare (nel pdf finale bisogna disporre tabelle e grafici)

MSE
MAE
SMAPE (short term forecasting) che esplode a inf. o NaN per valori piccoli, controllare → oppure usare MASE? tanto MSE e MAE sono gli standard
OK MASE
Loss Training/ Validation


[5. osservazioni finali: osservazioni; cosa manca; cosa hai imparato; difficoltà incontrate]



paper a parte, possibile suddivisione lavoro:

standard PyTorch ~ documentazione TF: [B,T,C] → BHC e disaccoppiare estrazione dati da computazione della rete

B: batch size, docs ~ 32 finestre rnd da dataset, alto stabilizza stime grad ma alza vram (layer inception TimesNet) e peggiora min locali (gpt)

T: T alto obbligatorio in TimesNet pk fft vuole interperiodo a bassa freq. ma va a scalare con O(TlogT), come lo risolviamo? se usiamo ETT si fa un reshape dei periodi noti di ETT?
C: channels/features e.g. volts amps temp C=3, 

H: orizzonti futuri prodotti, invece di produrre B1C come farebbe RNN 

nn.LSTM nn.conv1d secondo la docs vuole BCT, altrimenti da problemi pythorch, si usa tf?

1
2
dataloader
pipeline dati

tensori dalla stessa forma
modello
cambiare blocco 2d e fft?

tratta data loader come black box



lista di cose da fare: 

mail per chiedere feedback sulla proposta
repo github 



Buongiorno prof. Pegoraro, Adin,

vi scriviamo per condividere la struttura del nostro Progetto 2 sul forecasting temporale, ispirato alle logiche di rappresentazione di TimesNet, e per chiedervi un rapido feedback. Vorremmo verificare che la mole di attività proposta sia adeguata agli obiettivi del progetto, evitando al tempo stesso uno scope eccessivamente ampio.

Di seguito riportiamo il piano di lavoro.

1. Dataset e metriche

Dataset principale: ETT, scelto per valutare le ipotesi alla base della rappresentazione multi-periodica dell’architettura.
Caso di studio aggiuntivo: un breve esperimento su un dataset finanziario, con l’obiettivo di analizzare il comportamento dell’estrazione dei periodi tramite FFT in un contesto caratterizzato da bassa periodicità e ridotto rapporto segnale-rumore, senza finalità di ottimizzazione prestazionale.
Metriche: MSE, MAE e MASE.

2. Baseline e sviluppo architetturale

Implementazione da zero di alcune baseline 1D (MLP, CNN causale e LSTM) come riferimento.
Sviluppo di un’architettura 2D ispirata al TimesNet (trasformazione 1D→2D).

3. Studi di ablazione

Per contenere i tempi di addestramento e limitare il numero di esperimenti, prevediamo un approccio Leave-One-Out focalizzato sull’architettura 2D, considerando:

confronto tra periodi fissati a priori (noti dal dominio ETT) ed estrazione dinamica tramite FFT, per valutarne il contributo in termini di accuratezza rispetto al costo computazionale;
confronto tra blocchi Inception e un’architettura convoluzionale più leggera, per analizzare il compromesso tra complessità e prestazioni.

4. Metodologia e suddivisione del lavoro

Lavoreremo in parallelo definendo un’interfaccia comune tra i moduli (input/output nel formato [Batch, Time, Channels]), così da consentire uno sviluppo indipendente delle due componenti.

Vi chiediamo gentilmente un parere sulla fattibilità di questo esperimento. In particolare, ritenete che condurre gli esperimenti su diverse ablazioni architetturale Leave-One-Out, con i relativi test sistematici su molteplici orizzonti di previsione, sia una scelta ragionevole per soddisfare gli obiettivi del progetto oppure è troppo oneroso? Inoltre, testare 1D vs 2D è fattibile o nuovamente rischia di essere troppo oneroso?

Grazie per il tempo dedicato e buona giornata.

Cordiali saluti,

Nicola Ranzolin
Matteo Gidoni

Secondo me, in 3., dovremmo aggiungere che lo scopo principale è testare la nostra CNN 2D, che può avere periodo fisso, top-k periodi, periodo estratto con FFT, rispetto a CNN 1D.
Per completare il progetto aggiungiamo la tecnica di Ablation LOO, togliendo elemento per elemento per vedere quali sono le componenti utili.  Un altro obiettivo è valutare un dataset finanziario.
Io scriverei cosi: 
“3. obiettivo del progetto
confrontare reti 1D con 2D che possono avere periodo fisso, periodo calcolato con FFT, top-k periodi

Come ulteriore esperimento consideriamo un approccio chiamato Ablation Leave One Out per il quale realizziamo:
confronto tra periodi fissati a priori (noti dal dominio ETT) ed estrazione dinamica tramite FFT, per valutarne il contributo in termini di accuratezza rispetto al costo computazionale;
confronto tra blocchi Inception e un’architettura convoluzionale più leggera, per analizzare il compromesso tra complessità e prestazioni.
”





Buongiorno Prof. Pegoraro, ciao Aidin,

vi scriviamo per condividere la struttura del nostro Progetto 2 sul forecasting temporale, ispirato alle logiche di rappresentazione di TimesNet, e per chiedervi un rapido feedback. Vorremmo verificare che la mole di attività proposta sia adeguata agli obiettivi del progetto, evitando al tempo stesso uno scope eccessivamente ampio. Di seguito riportiamo il piano di lavoro.

1. Dataset e Metriche

Dataset principale: ETT, per valutare oggettivamente le ipotesi alla base della rappresentazione multi-periodica.
Edge Case: un breve esperimento su un dataset finanziario. L’obiettivo non è massimizzare le performance, ma dimostrare criticamente che l’estrazione dei periodi (FFT) collassa estraendo rumore in un contesto a bassa periodicità e ridotto Signal-to-Noise Ratio.
Metriche: MSE, MAE e MASE.

2. Esperimento principale: confronto tra modelli 1D e 2D

È il core del progetto e implementeremo da zero alcune baseline 1D (MLP, CNN causale e LSTM) e le confronteremo con un’architettura 2D custom (trasformazione 1D → 2D), valutando l’impatto della rappresentazione multi-periodica. Per il modello 2D considereremo tre configurazioni:

periodo fisso noto a priori;
periodo singolo estratto tramite FFT;
top-k periodi estratti tramite FFT;
modello 2D con blocchi Inception

3. Esperimento secondario: Complexity Reduction

Una volta validata l’architettura 2D, svolgeremo un’ablazione Leave-One-Out con l’obiettivo di studiare il compromesso tra complessità computazionale e accuratezza, senza moltiplicare il numero di esperimenti. In particolare confronteremo:

periodi hardcoded dal dominio vs estrazione dinamica tramite FFT, per quantificare il beneficio rispetto all’overhead computazionale;
blocchi Inception densi vs un’architettura convoluzionale più leggera.

4. Metodologia e sviluppo

Lavoreremo in parallelo definendo un’interfaccia stretta tra i moduli: il Dataloader (trattato come black box) produrrà tensori nel formato [Batch, Time, Channels] → [Batch, Horizon, Channels], disaccoppiando completamente l’estrazione dei dati dalla computazione della rete.

Per contenere i tempi sperimentali complessivi, stavamo valutando di limitare la valutazione a uno o pochi orizzonti predittivi rappresentativi e concentrare l’analisi sul confronto tra le architetture e sulle relative ablazioni, evitando la replicazione sistematica dei test su diversi orizzonti di previsione.

Ritenete che questa sia una scelta metodologicamente appropriata?

Grazie per il tempo dedicato.

Cordiali saluti,

Nicola Ranzolin
Matteo Gidoni


