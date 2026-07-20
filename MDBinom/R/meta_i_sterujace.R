# TODO: Add comment
# 
# Author: Michał Danaj
###############################################################################



NA_substit=-.Machine$integer.max;
special_val=c(-99999999, -10000001, -10000000, -9999999,  -9999998,  -9999997, -9999996);


#' Parametry
#'
#' Parametryzacja funkcji z pakietu.
#'
#' @format Wektor o polach:
#' \describe{
#'   \item{discret_threshold}{Przy jakiej wartości unikalnych wartości zmienna numeryczna będzie traktowana jako dyskretna.}
#'   \item{special_val}{Kody różnych wartości specjalnych. Będą one traktowane jako wartości dyskretne, mimo że pozostała część zmiennej może być ciągła.}
#' }
#' @export 
numeric_var_treatment.params<-list(
		#przy jakiej wartości unikalnych wartości zmiennej numerycznej ma ją traktować jako dyskretną
		discrete_threshold=15,
		
		#liczba unikalnych wartości zmiennej kategorycznej, powyżej której nie generuje statystyk
		#w funkcji univariate_stats1. W przypadku przekroczenia, zwracany jest komunikat "Too many categorical levels".
		no_stats_threshold=15,
		
		#kody różnych wartości specjalnych. Będą one traktowane osobno, jako wartości dyskretne
		special_val=-.Machine$integer.max,
		
		#wartość do zastąpienia missing value
		NA_substit=-.Machine$integer.max,
		
		#Graniczny udział wartości, powyżej której traktujemy ją jako wartość specjalną 
		#(w sposób dyskretny, wydzieloną z pozostałych).
		separate_value_thr=0.1,		
		
		#maksymalna głębokość budowy drzewa moim algorytmem
		max_gleb=3,
		
		#minimalna liczba obserwacji - do sprawdzenia - w liściu/w węźle do podziału
		min_bucket=200,
		
		#wartość graniczna nulli. Poniżej robimy imputację, powyżej traktujemy je jako osobną grupę
		nulle_do_imp_thr=0.0

)



# TODO dorobić wyświetlanie, które role zostały wprowadzone niepoprawnie.
#' Generuje kod do zmiany roli zmiennych lub data.frame z rolami
#' 
#' Generuje kod do zmiany roli zmiennych lub data.frame ze zmienionymi rolami.
#' @param zmienne_rola \code{data.frame} z rolami zmiennych.
#' @param pattern Wyrażenie przekazane do funkcji \code{grep} filtrujące zmienne
#' @param gen_code Jeśli TRUE, wynikiem funkcji jest kod do zmiany wartości. W przeciwnym razie, zwracany
#'			jest data.frame ze zmienionymi rolami.  
#' @return 
#' 
#' @author Michał Danaj
#' @export
editVariablesRole<-function(zmienne_rola, pattern=NULL, gen_code=TRUE){
	
	rola<-c("rejected", "explanatory", "target", "keep");
	
	if (is.null(pattern))
		wynik<-edit(zmienne_rola)
	else{
		indeksy<-grep(pattern, rownames(zmienne_rola), ignore.case = TRUE)
		wynik<-zmienne_rola
		wynik[indeksy,]<-edit(zmienne_rola[indeksy,])
		
		if (any(!(wynik$rola%in%rola))){
			blad<-"Niepoprawne wartości ról!"
			#stop(rownames(zmienne_rola)[!wynik$rola%in%rola])
			stop(blad)
		}
	}
	
	#sprawdzam, których zmiennych role się zmieniły
	zmiany<- zmienne_rola$rola!=wynik$rola
	nazwy_zmienionych<-rownames(zmienne_rola)[zmiany]
	nowe_wartosci<-wynik$rola[zmiany]
	
	#dla tych zmiennych generuję kod do zmiany ich ról
	nazwy_przecinek<-paste("c('", paste(nazwy_zmienionych, collapse = "','"), "')", sep="")
	wartosci_przecinek<-paste("c('", paste(nowe_wartosci, collapse = "','"), "')", sep="")
	
	if (gen_code==TRUE){
		kod<-paste(deparse(substitute(zmienne_rola)),"[",nazwy_przecinek,",'rola']<-", wartosci_przecinek, sep="")
		cat(kod)
	}
	else
		wynik
}


#########################################

